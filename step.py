# Falling back to a simple stepper motor control script using RPi.GPIO
# Don't have enbough cables to use the stepper motor driver with the dedicated library

import RPi.GPIO as GPIO
import time

# Pin setup
DIR_PIN = 20
STEP_PIN = 21
ENABLE_PIN = 19  # Optional but recommended

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, STEP_PIN, ENABLE_PIN], GPIO.OUT)

# Enable the driver (pull EN pin LOW)
GPIO.output(ENABLE_PIN, GPIO.LOW)

def move_motor(steps, clockwise=True, delay=0.001):
    GPIO.output(DIR_PIN, GPIO.HIGH if clockwise else GPIO.LOW)
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)

try:
    move_motor(200, True)   # 200 steps clockwise
    move_motor(200, False)  # 200 steps counter-clockwise
finally:
    GPIO.output(ENABLE_PIN, GPIO.HIGH)  # Disable driver
    GPIO.cleanup()
