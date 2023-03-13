import numpy as np
import pandas as pd 
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm 
import os 
import concurrent.futures
import multiprocessing

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

#exposure_times = get_exosure_hash('times.txt')
#timestamps, position, quaternions = load_gt_data('/home/cm2113/workspace/thesis/rotation_test/data.csv')
timestamps, position, quaternions = load_gt_data('/home/cm2113/workspace/thesis/datasets/tumvi_room1_512_16/data.csv')
FX = 254.93170605935475 # 190.97847715128717
FY = 256.8974428996504 # 190.9733070521226
CX = 190.97847715128717 # 254.93170605935475
CY = 190.9733070521226 # 256.8974428996504

CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
CAM_D = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
DIM = (512,512)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        q_delta = qs[i+1]/qs[i]  
        q_delta = q_delta.normalised 
        
        #if i == 0: 
        #    print(f'q_delta angle: {q_delta.angle} q_delta axis: {q_delta.axis}')
        
        R_delta = q_delta.rotation_matrix #R2 @ np.linalg.inv(R1)

        x_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)
        y_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)

        for y in range(DIM[1]):
            for x in range(DIM[0]): 
                # undistort point
                p0 = np.array([x, y], dtype=np.float32).reshape(1,1,2)
                p0 = cv2.fisheye.undistortPoints(p0, CAM_K, CAM_D)
                p0 = np.array([p0[0][0][0], p0[0][0][1], 1])

                p = R_delta @ p0
                
                # distort back point 
                _p = np.array([p[0], p[1]]).reshape(1,1,2)
                _p = cv2.fisheye.distortPoints(_p, CAM_K, CAM_D)

                x_map[y,x] = _p[0][0][0]
                y_map[y,x] = _p[0][0][1]

        image = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frames.append(image)
    return frames

def create_frames(timestamp1:int, qs:list[Quaternion], results_path): 
    frames = compute_frames_KRK(timestamp1, qs) #backward_frames # + forward_frames[1:] 

    avg_image = frames[0]
    for i in range(len(frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(frames[i], alpha, avg_image, beta, 0.0)

    cv2.imwrite(f'{results_path}/{str(timestamp1)}_avg.png', avg_image)

def nextnonexistent(f):
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew

def create_list_of_quaternions(timestamp1:int, timestamp2:int, dt:float): 
    q1 = get_quaternion(timestamp1)
    q2 = get_quaternion(timestamp1-dt)
    qs = get_mid_quaternion(q1,q2)
    return qs

def get_mid_quaternion(q1:Quaternion, q2:Quaternion, tol:float=1e-3): 
    if Quaternion.absolute_distance(q1,q2) < tol:
        return [q1, q2]
    q_mid = Quaternion.slerp(q1, q2).normalised
    left_points = get_mid_quaternion(q1, q_mid)
    right_points = get_mid_quaternion(q_mid, q2)
    return left_points + right_points[1:]

PATH = '/home/cm2113/workspace/thesis/datasets/tumvi_room1_512_16/data/' 
#RESULT_PATH = './results'
#RESULT_PATH = nextnonexistent(RESULT_PATH)
#print(RESULT_PATH)
#os.mkdir(RESULT_PATH)

def create_frames_parallel(frame,quats, results_path):
    timestamp = int(frame.replace(".png", ""))
    create_frames(timestamp, quats[timestamp], results_path)

def main():
    frames = sorted(os.listdir(os.path.join(PATH)))
    divisions = [2]
    top_result_path = '/home/cm2113/workspace/thesis/results/tumvi_room1_blur'
    if not os.path.exists(top_result_path):
        os.mkdir(top_result_path)

    for div in divisions: 
        results_path = f'{top_result_path}/tumvi_room1_512_16_blur_{div}'
        results_path = nextnonexistent(results_path)
        os.mkdir(results_path)

        quats = {}
        print(f'[INFO] Create list of quaternions for each image timeinterval devided by {div}, remember i=0 is the quats for the second image in the sequence')
        for i in tqdm(range(len(frames)-1)): 
            timestamp0 = int(frames[i].replace(".png", ""))
            timestamp1 = int(frames[i+1].replace(".png", ""))
            dt = (timestamp1-timestamp0)/div
            qs = create_list_of_quaternions(timestamp1, timestamp0, dt)
            quats[timestamp1] = qs
            
        print('[INFO] creating frames')
        with multiprocessing.Pool() as pool:
            results = []
            for frame in frames[1:]:
                result = pool.apply_async(create_frames_parallel, args=(frame,quats,results_path,))
                results.append(result)
            for result in results:
                result.wait()

    """for i in tqdm(range(len(frames)-1)): 
        timestamp = int(frames[i+1].replace(".png", ""))
        create_frames(timestamp, quats[i])"""


if __name__ == '__main__': 
    # test distorsion model: 
    test_distorsion = False 
    if test_distorsion: 
        image = cv2.imread('./image_sequence/1520530327850099100.png', cv2.IMREAD_UNCHANGED)
        x_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)
        y_map = np.zeros((DIM[1], DIM[0]), dtype=np.float32)
        cv2.imshow('original', image)
        for y in range(DIM[1]):
            for x in range(DIM[0]): 
                # undistort point
                p0 = np.array([x, y], dtype=np.float32).reshape(1,1,2)
                p0 = cv2.fisheye.undistortPoints(p0, CAM_K, CAM_D)
                p0 = np.array([p0[0][0][0], p0[0][0][1], 1])
                
                p = p0
                #p = R_delta @ p0
                
                # distort back point 
                _p = np.array([p[0], p[1]]).reshape(1,1,2)
                _p = cv2.fisheye.distortPoints(_p, CAM_K, CAM_D)

                x_map[y,x] = _p[0][0][0]
                y_map[y,x] = _p[0][0][1]

        image = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow('back to undistorted', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    main()


