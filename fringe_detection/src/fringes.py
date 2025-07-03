import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Avoid Qt conflicts with OpenCV
matplotlib.use('Agg')  

def crop_frames(frames, crop_region):
    (xmin, xmax), (ymin, ymax) = crop_region
    return frames[:, ymin:ymax, xmin:xmax]

def subtract_min(frames):
    """Subtract minimum frame and ensure non-negative values."""
    # Calculate minimum in float64 for precision
    min_frame = np.min(frames, axis=0)
    
    # Save min frame visualization
    plt.imshow(min_frame.squeeze(), cmap='gray')
    plt.colorbar()
    plt.title('Min Frame')
    plt.savefig("min_frame.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Subtract min and handle negative values efficiently
    frames_float = frames.astype(np.float64) - min_frame

    max_val = np.amax(frames_float)
    frames_float *= 255.0 / max_val 

    return frames_float

def preprocess_video(input_path):
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    height, width = first_frame.shape
    
    frames = np.empty((total_frames, height, width), dtype=np.uint8)
    frames[0] = first_frame
    
    frame_idx = 1
    with tqdm(total=total_frames-1, desc="Loading frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale immediately
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames[frame_idx] = frame
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    frames = frames[..., np.newaxis]

    frames = subtract_min(frames)
    
    return frames, fps

def video_from_frames(frames, output_path, fps=30):
    height, width = frames.shape[1:3]
    frames_bgr = np.repeat(frames, 3, axis=3)  # Convert to BGR by repeating the single channel
    print(frames_bgr.shape)

    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))
    
    with tqdm(total=len(frames_bgr), desc="Writing video") as pbar:
        for frame in frames_bgr:
            out.write(frame.astype(np.uint8))
            pbar.update(1)
    
    out.release()

def get_phase_cross_correlation_shifts(frames, upsample_factor=100):
    if len(frames.shape) == 4 and frames.shape[3] == 1:
        # Remove channel dimension for processing
        frames_2d = frames.squeeze(axis=3)
    else:
        frames_2d = frames
    
    relative_shifts = [np.array([0.0, 0.0])]
    absolute_shifts = [np.array([0.0, 0.0])]
    
    # Convert frames to float64 once for better precision
    frames_float = frames_2d.astype(np.float64)
    
    # Process consecutive frames with progress bar
    with tqdm(total=len(frames_float)-1, desc="Computing shifts") as pbar:
        for i in range(len(frames_float)-1):
            reference = frames_float[i]
            current = frames_float[i+1]
            
            # Calculate phase correlation
            shift, _, _ = phase_cross_correlation(
                reference, current, 
                upsample_factor=upsample_factor
            )
            
            relative_shifts.append(shift)
            absolute_shifts.append(absolute_shifts[-1] + shift)
            pbar.update(1)
    
    return np.array(absolute_shifts), np.array(relative_shifts)


def make_pretty_plot(total_shifts):
    plt.figure(figsize=(10, 5))
    plt.plot(total_shifts, label="Total Cumulative Shift")
    plt.xlabel("Frame Index")
    plt.ylabel("Total Shift (a.u.)")
    plt.title("Total Phase Correlation Shift Over Time")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    # vlines at 400 and 800
    plt.axvline(x=400, color='r', linestyle='--', label='Pump On')
    plt.axvline(x=800, color='g', linestyle='--', label='Pump Off')
    # annotate
    plt.annotate('Pump On', xy=(410, max(total_shifts) * 0.9), color='red', fontsize=12)
    plt.annotate('Pump Off', xy=(810, max(total_shifts) * 0.9), color='green', fontsize=12)
    plt.savefig("total_shifts.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    vid = r"data/1ul_pump_on_off_2.mp4"
    out = r"data/1ul_pump_on_off_2_cropped.mp4"
    crop_region = None
    crop_regions = [((0, 400), (0, 100)),
                    ]
    
    print("Preprocessing video...")
    frames, fps = preprocess_video(vid)
    frames = frames[:1250]  # Limit to first 1250 frames as ROI
    print(f"Processed {len(frames)} frames at {fps:.1f} FPS")
    
    print("Saving processed video...")
    video_from_frames(frames, out, fps=int(fps))
    
    # Analyze shifts on cropped region for efficiency
    print("Computing phase correlation shifts...")

    frames_cropped = [crop_frames(frames, crop_region) for crop_region in crop_regions]
    shifts = []
    for cropped_frames in frames_cropped:
        absolute_shifts, relative_shifts = get_phase_cross_correlation_shifts(cropped_frames)
        shifts.append(absolute_shifts)

    # calculate total shift (pythagoras)
    total_shifts = [np.linalg.norm(shift, axis=1) for shift in shifts]
    total_shifts = np.sum(total_shifts, axis=0)

    make_pretty_plot(total_shifts)
    
   