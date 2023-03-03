import numpy as np
import pandas as pd 
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm 
import os 

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------    ALL THIS IS SEQUENCE DEPENDENT ! --------------------------------------------------------------------------------------------------------------------------------------------------

def load_gt_data(filename):
    data = pd.read_csv(filename).sort_values(by=['#timestamp [ns]']).rename(columns={' p_RS_R_x [m]':'p_x', ' p_RS_R_y [m]':'p_y', ' p_RS_R_z [m]':'p_z', ' q_RS_w []':'q_w', ' q_RS_x []':'q_x', ' q_RS_y []':'q_y', ' q_RS_z []':'q_z' })
    timestamps = data['#timestamp [ns]']
    positions = []
    quaternions = []
    for _, row in data.iterrows(): 
        positions.append([row.p_x, row.p_y, row.p_z])
        quaternions.append(Quaternion(row.q_w, row.q_x, row.q_y, row.q_z))

    return timestamps, positions, quaternions

def get_exosure_hash(filename): 
    timestamps = []
    exp_times = []

    with open(filename) as f:
        next(f)
        for line in f: 
            data = line.split()
            timestamps.append(int(data[0])) # timestamps are given in ns 
            exp_times.append(float(data[2])*1e6) # exposure times in ms, convert to ns by multiplying by 1e6 

    timestamps = np.array(timestamps)
    exp_times = np.array(exp_times)

    hash = {key: value for key, value in zip(timestamps, exp_times)}
    return hash

exposure_times = get_exosure_hash('times.txt')
timestamps, position, quaternions = load_gt_data('/home/cm2113/workspace/thesis/rotation_test/data.csv')

FX = 190.97847715128717
FY = 190.9733070521226
CX = 254.93170605935475
CY = 256.8974428996504

CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
CAM_D = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
DIM = (512,512)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# want a method to compute the maximum number of frames between two images depending on the choosen dt 
def compute_dt(timestamp:int): 
    exposure_time = exposure_times[timestamp]
    return exposure_time/2 

def get_quaternions(timestamp:int, dt:float, n:int): 
    qs = []
    for i in range(n):
        timestamp += i*dt
        time_diff = timestamps-timestamp
        sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0]) 

        t1 = timestamps[sign_switch_idx]
        t2 = timestamps[sign_switch_idx + 1]

        assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
        assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

        q1 = quaternions[sign_switch_idx]

        q2 = quaternions[sign_switch_idx + 1]
        q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1))
        qs.append(q_.normalised)
    return qs 

def get_quaternion(timestamp:int): 
    time_diff = np.array(timestamps) - timestamp

    sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0])

    t1 = timestamps[sign_switch_idx]
    t2 = timestamps[sign_switch_idx + 1]

    assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
    assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

    q1 = quaternions[sign_switch_idx]
    q2 = quaternions[sign_switch_idx + 1]
    q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1)).normalised

    return q_

def compute_frames_KRK(timestamp:int, qs:list[Quaternion]): 
    image = cv2.imread(f'{PATH}{str(timestamp)}.png', cv2.IMREAD_UNCHANGED)
    frames = []

    frames.append(image) 
    for i in range(len(qs)-1): 
        R1 = qs[i].rotation_matrix
        R2 = qs[i+1].rotation_matrix 

        H, _ = cv2.findHomography(np.array([[0,0,1],[0,1,1],[1,1,1],[1,0,1]]),
                            np.array([[0,0,1],[0,1,1],[1,1,1],[1,0,1]]).dot(R2).dot(np.linalg.inv(R1)))

        x_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)
        y_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)

        for y in range(DIM[1]):
            for x in range(DIM[0]): 
                p0 = np.array([x, y, 1])

                p = H @ p0

                x_map[y,x] = p[0]
                y_map[y,x] = p[1]

        image = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frames.append(image)
    return frames

def create_frames(timestamp1:int, timestamp0:int): 
    # for now use dt = 1/4*exp_time 
    dt = compute_dt(timestamp1)
    n = 10 # the number of frames on EACH side

    #qs_forward = get_quaternions(timestamp1, dt, n+1)
    #qs_backward = get_quaternions(timestamp1, -dt, n+1)

    #forward_frames = compute_frames_KRK(timestamp1, qs_forward)
    #backward_frames = compute_frames_KRK(timestamp1, qs_backward)
    #backward_frames = backward_frames[::-1] 
    qs = create_list_of_quaternions(timestamp1, timestamp0)


    frames = compute_frames_KRK(timestamp1, qs) #backward_frames # + forward_frames[1:] 

    avg_image = frames[0]
    for i in range(len(frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(frames[i], alpha, avg_image, beta, 0.0)

    cv2.imwrite(f'{RESULT_PATH}/{str(timestamp1)}_avg.png', avg_image)

def nextnonexistent(f):
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew

def create_list_of_quaternions(timestamp1:int, timestamp2:int): 
    q1 = get_quaternion(timestamp1)
    q2 = get_quaternion(timestamp2)
    
    # get the mid quaternion of the two 
    q_mid = Quaternion.slerp(q1,q2)
    qs = get_mid_quaternion(q1,q_mid)
    return qs

def get_mid_quaternion(q1:Quaternion, q2:Quaternion, tol:float=1e-3): 
    if Quaternion.absolute_distance(q1,q2) < tol:
        return [q1, q2]
    q_mid = Quaternion.slerp(q1, q2).normalised
    left_points = get_mid_quaternion(q1, q_mid)
    right_points = get_mid_quaternion(q_mid, q2)
    return left_points + right_points[1:]

PATH = './image_sequence/' 
RESULT_PATH = './results'
RESULT_PATH = nextnonexistent(RESULT_PATH)
print(RESULT_PATH)
os.mkdir(RESULT_PATH)

def main():
    #image0 = 1520530327700094100
    #image1 = 1520530327750096100
    #image2 = 1520530327800097100
    #frames = [image0, image1, image2]

    frames = sorted(os.listdir(os.path.join(PATH)))

    for i in tqdm(range(len(frames)-1)): 
        timestamp0 = int(frames[i].replace(".png", ""))
        timestamp1 = int(frames[i+1].replace(".png", ""))
        create_frames(timestamp1, timestamp0)

main()

