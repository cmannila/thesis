import shutil
import numpy as np
import pandas as pd 
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm 
import os 
import multiprocessing
from load_camera import camera 
import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------    ALL THIS IS SEQUENCE DEPENDENT ! --------------------------------------------------------------------------------------------------------------------------------------------------
DATASETS_PATH = '/home/cm2113/workspace/Datasets'
CAM_SUB_PATH = '/dso/camchain.yaml'
TIME_SUB_PATH = '/mav0/mocap0/data.csv'
FRAMES_SUB_PATH = '/mav0/cam0/data/'


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

#timestamps, position, quaternions = load_gt_data(f'{DATASETS_PATH}{DATA_SUB_PATH}{TIME_SUB_PATH}')

class data_loader: 
    def __init__(self, data_sub_path:str, top_result_path:str): 
        # load essential data 
        self.timestamps, self.positions, self.quaternions = load_gt_data(f'{DATASETS_PATH}{data_sub_path}{TIME_SUB_PATH}') 
        self.cam = camera(f'{DATASETS_PATH}{data_sub_path}{CAM_SUB_PATH}')
        self.top_result_path = top_result_path
        if not os.path.exists(self.top_result_path):
            os.mkdir(self.top_result_path)
        self.dataset_name = os.path.basename(os.path.normpath(data_sub_path))
        self.result_path = None 
        self.long_data_path = f'{DATASETS_PATH}{data_sub_path}{FRAMES_SUB_PATH}'
    
    def __str__(self): 
        return f'Timestamps: {self.timestamps[:10]}, Camera: {self.cam}, Dataset_name: {self.dataset_name}'
    
    def set_result_path(self, result_path:str):
        self.result_path = result_path 

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_quaternions(timestamp:int, dt:float, n:int, dl:data_loader): 
    qs = []
    for i in range(n):
        timestamp += i*dt
        time_diff = dl.timestamps-timestamp
        sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0]) 

        t1 = dl.timestamps[sign_switch_idx]
        t2 = dl.timestamps[sign_switch_idx + 1]

        assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
        assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

        q1 = dl.quaternions[sign_switch_idx]

        q2 = dl.quaternions[sign_switch_idx + 1]
        q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1))
        qs.append(q_.normalised)
    return qs 

def get_quaternion(timestamp:int, dl:data_loader): 
    time_diff = np.array(dl.timestamps) - timestamp

    sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0])

    t1 = dl.timestamps[sign_switch_idx]
    t2 = dl.timestamps[sign_switch_idx + 1]

    assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
    assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

    q1 = dl.quaternions[sign_switch_idx]
    q2 = dl.quaternions[sign_switch_idx + 1]
    q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1)).normalised

    return q_

def compute_frames_KRK(timestamp:int, qs:list[Quaternion], dl:data_loader): 
    image = cv2.imread(f'{dl.long_data_path}{str(timestamp)}.png', cv2.IMREAD_UNCHANGED)
    frames = []

    frames.append(image) 
    for i in range(len(qs)-1):
        q_delta = qs[i+1]/qs[i]  
        q_delta = q_delta.normalised 
        
        R_delta = q_delta.rotation_matrix #R2 @ np.linalg.inv(R1)

        x_map = np.zeros((dl.cam.DIM[1], dl.cam.DIM[0]), dtype=np.float32)
        y_map = np.zeros((dl.cam.DIM[1], dl.cam.DIM[0]), dtype=np.float32)

        for y in range(dl.cam.DIM[1]):
            for x in range(dl.cam.DIM[0]): 
                # undistort point
                p0 = np.array([x, y], dtype=np.float32).reshape(1,1,2)
                p0 = cv2.fisheye.undistortPoints(p0, dl.cam.K, dl.cam.D)
                p0 = np.array([p0[0][0][0], p0[0][0][1], 1])

                p = np.linalg.inv(R_delta) @ p0
                
                # distort back point 
                _p = np.array([p[0], p[1]]).reshape(1,1,2)
                _p = cv2.fisheye.distortPoints(_p, dl.cam.K, dl.cam.D)

                x_map[y,x] = _p[0][0][0]
                y_map[y,x] = _p[0][0][1]

        image = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frames.append(image)
    return frames

def create_frames(timestamp1:int, qs:list[Quaternion], dl:data_loader): 
    frames = compute_frames_KRK(timestamp1, qs, dl) #backward_frames # + forward_frames[1:] 

    avg_image = frames[0]
    for i in range(len(frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(frames[i], alpha, avg_image, beta, 0.0)

    cv2.imwrite(f'{dl.results_path}/{str(timestamp1)}_avg.png', avg_image)

def nextnonexistent(f):
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew

def create_list_of_quaternions(timestamp1:int, timestamp2:int, dt:float, dl:data_loader): 
    q1 = get_quaternion(timestamp1, dl)
    q2 = get_quaternion(timestamp1-dt, dl)
    qs = get_mid_quaternion(q1,q2)
    return qs

def get_mid_quaternion(q1:Quaternion, q2:Quaternion, tol:float=1e-3): 
    if Quaternion.absolute_distance(q1,q2) < tol:
        return [q1, q2]
    q_mid = Quaternion.slerp(q1, q2).normalised
    left_points = get_mid_quaternion(q1, q_mid)
    right_points = get_mid_quaternion(q_mid, q2)
    return left_points + right_points[1:]

def create_frames_parallel(frame, quats, dl):
    timestamp = int(frame.replace(".png", ""))
    create_frames(timestamp, quats[timestamp], dl)


def main():
    parser = argparse.ArgumentParser(description='Blur image sequence using rotational data')
    parser.add_argument('data', default='/tumvi/room/dataset-room1_512_16', type=str, help='the data folder in the dataset path, default=/tumvi/room/dataset-room1_512_16')
    parser.add_argument('output', default='/home/cm2113/workspace/thesis/results/tumvi_room1_blur', type=str, help='the path in which the resulting frames should be stored')
    parser.add_argument('div', default=[4], type=list[int], help='')

    args = parser.parse_args()
    data_sub_path = args.data 
    top_result_path = args.output 
    divisions = args.div 

    # load data object
    dl = data_loader(data_sub_path=data_sub_path, top_result_path=top_result_path)

    # load frames from data folder 
    frames = sorted(os.listdir(os.path.join(f'{DATASETS_PATH}{data_sub_path}{FRAMES_SUB_PATH}')))

    for div in divisions: 
        results_path = f'{top_result_path}/{dl.dataset_name}_blur_{div}'
        results_path = nextnonexistent(results_path)
        os.mkdir(results_path)
        dl.set_result_path = results_path

        quats = {}
        print(f'[INFO] Create list of quaternions for each image timeinterval devided by {div}, remember i=0 is the quats for the second image in the sequence')
        for i in tqdm(range(len(frames)-1)): 
            timestamp0 = int(frames[i].replace(".png", ""))
            timestamp1 = int(frames[i+1].replace(".png", ""))
            dt = (timestamp1-timestamp0)/div
            qs = create_list_of_quaternions(timestamp1, timestamp0, dt, dl)
            quats[timestamp1] = qs
            
        print('[INFO] creating frames')
        # add first frame to the result folder - will not be blurred
        shutil.copy(f'{dl.long_data_path}{frames[0]}', f'{results_path}/{frames[0]}')

        with multiprocessing.Pool() as pool:
            results = []
            for frame in frames[1:]:
                result = pool.apply_async(create_frames_parallel, args=(frame,quats,dl,))
                results.append(result)
            for result in results:
                result.wait()
        
        print(f'[INFO] Created {len(os.listdir(results_path))} files')

    """for i in tqdm(range(len(frames)-1)): 
        timestamp = int(frames[i+1].replace(".png", ""))
        create_frames(timestamp, quats[i])"""


if __name__ == '__main__': 
    main()


