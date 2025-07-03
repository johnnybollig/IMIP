import subprocess
import os
import time

def record_video(output='video.h264', duration=5, gain=1, exposure_time=10000):
    # Input validation
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if exposure_time < 1 or exposure_time > 6000000:
        raise ValueError("Exposure time must be between 1 and 6000000 microseconds")
    
    # Check if output directory exists, create if needed
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert exposure time from microseconds to seconds for rpicam-vid
    exposure_seconds = exposure_time / 1000000.0
    
    # Build rpicam-vid command
    cmd = [
        'rpicam-vid',
        '--output', output,
        '--width', '1920',
        '--height', '1080',
        '--framerate', '30',
        '--gain', str(gain),
        '--shutter', str(exposure_time),  # rpicam-vid uses microseconds
    ]
    
    try:
        print(f"Recording {output} for {duration} seconds...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run rpicam-vid
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Recording finished.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"rpicam-vid error (exit code {e.returncode}): {e.stderr}")
        raise RuntimeError(f"Video recording failed: {e.stderr}")
    except FileNotFoundError:
        print("rpicam-vid not found. Make sure you're running on a Raspberry Pi with camera support.")
        raise RuntimeError("rpicam-vid command not available")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Set your parameters here
if __name__ == "__main__":
    output_file = "video.h264"
    duration_seconds = 5
    gain = 1            
    exposure_us = 100       # In microseconds (1ms = 1000us)

    record_video(output_file, duration_seconds, gain, exposure_us)
