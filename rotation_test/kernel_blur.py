import cv2
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

def load_data(filename):
    data = pd.read_csv(filename).sort_values(by=['#timestamp [ns]']).rename(columns={' p_RS_R_x [m]':'p_x', ' p_RS_R_y [m]':'p_y', ' p_RS_R_z [m]':'p_z', ' q_RS_w []':'q_w', ' q_RS_x []':'q_x', ' q_RS_y []':'q_y', ' q_RS_z []':'q_z' })
    timestamps = data['#timestamp [ns]']
    positions = []
    quaternions = []
    for _, row in data.iterrows(): 
        positions.append([row.p_x, row.p_y, row.p_z])
        quaternions.append(Quaternion(row.q_w, row.q_x, row.q_y, row.q_z))

    return timestamps, positions, quaternions


def get_quaternion(image:str, n:int=4): 
    timestamp = int(image)
    dt = 30000
    qs = []
    for i in range(n): 
        timestamp += i*dt 
        time_diff = np.array(timestamps) - timestamp

        sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0])

        t1 = timestamps[sign_switch_idx]
        t2 = timestamps[sign_switch_idx + 1]

        assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
        assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

    

        q1 = quaternions[sign_switch_idx]
        q2 = quaternions[sign_switch_idx + 1]
        q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1))
        qs.append(q_)
    #q_ /= q_
    #q_= quaternions[sign_switch_idx]
    #q0 = q_
    #q_ = q0/q0
    #q_ /= q_

    return qs, sign_switch_idx

image1_name = '1520530327700094100'
image2_name = '1520530327800097100'
timestamps, position, quaternions = load_data('data.csv')

img = cv2.imread(f'{image1_name}.png', cv2.IMREAD_UNCHANGED)
qs, id = get_quaternion(image1_name)
# Define the kernel size and direction of motion
kernel_size = 15
motion_directions = [np.rad2deg(q[0]) for q in qs]  # in degrees


# Define the exposure time for each image in seconds
exposure_times = [0.0008, 0.0008, 0.0008, 0.0008]

# Calculate the motion blur kernel based on the kernel size and direction of motion
kernels = []
for direction in motion_directions:
    kernel = np.zeros((kernel_size, kernel_size))
    center = int((kernel_size - 1) / 2)
    angle = np.deg2rad(direction)
    sin_val = np.sin(angle)
    cos_val = np.cos(angle)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if abs(i - center) * cos_val + abs(j - center) * sin_val < center:
                kernel[i, j] = 1
    kernel /= kernel.sum()
    kernels.append(kernel)

# Simulate motion blur for each image in the sequence
blurred_sequence = []
image = img
frames = []
frames.append(image)

for i, exp in enumerate(exposure_times): 
    # Convert the exposure time to a blur factor
    blur_factor = int(exp * 1000 / kernel_size) * 2 + 1

    # Apply the motion blur kernel to the image
    blurred = cv2.filter2D(image, -1, kernels[i]) * blur_factor

    # Normalize the blurred image to 8-bit grayscale
    blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    frames.append(blurred)
    image = blurred

# Display the blurred image sequence
for i, frame in enumerate(frames):
    cv2.imshow(f'Blurred image {i}', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()