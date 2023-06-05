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

DATASETS_PATH = '/home/cm2113/workspace/Datasets'
CAM_SUB_PATH =  '/dso/camchain.yaml'
TIME_SUB_PATH = '/mav0/mocap0/data.csv'
FRAMES_SUB_PATH = '/mav0/cam0/data/'
NUMBER_PROC = 8

def load_gt_data(filename:str):
    """
        Function loading the data from the ground truth (filename) 
    """
    data = pd.read_csv(filename).sort_values(by=['#timestamp [ns]']).rename(columns={' p_RS_R_x [m]':'p_x', ' p_RS_R_y [m]':'p_y', ' p_RS_R_z [m]':'p_z', ' q_RS_w []':'q_w', ' q_RS_x []':'q_x', ' q_RS_y []':'q_y', ' q_RS_z []':'q_z' })
    timestamps = data['#timestamp [ns]']
    positions = []
    quaternions = []
    for _, row in data.iterrows(): 
        positions.append([row.p_x, row.p_y, row.p_z])
        quaternions.append(Quaternion(row.q_w, row.q_x, row.q_y, row.q_z))

    return timestamps, positions, quaternions

## NOT USED 
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

class data_loader: 
    """
        data_loder: object to keep track of groundtruth data and result path
    """
    def __init__(self, data_sub_path:str, top_result_path:str): 
        self.timestamps, self.positions, self.quaternions = load_gt_data(f'{DATASETS_PATH}{data_sub_path}{TIME_SUB_PATH}') 
        self.timepath = f'{DATASETS_PATH}{data_sub_path}{TIME_SUB_PATH}'
        self.cam = camera(f'{DATASETS_PATH}{data_sub_path}{CAM_SUB_PATH}')  
        self.top_result_path = top_result_path
        if not os.path.exists(self.top_result_path):
            os.mkdir(self.top_result_path)
        self.dataset_name = os.path.basename(os.path.normpath(data_sub_path))
        self.result_path = None 
        self.long_data_path = f'{DATASETS_PATH}{data_sub_path}{FRAMES_SUB_PATH}'
    
    def __str__(self): 
        return f'Timestamps: {self.timestamps[:10]}, Camera: {self.cam}, Dataset_name: {self.dataset_name}, result_path: {self.result_path}, data path: {self.long_data_path} and time path: {self.timepath}'
    
    def set_result_path(self, result_path:str):
        self.result_path = result_path 

def get_quaternion(timestamp:int, dl:data_loader): 
    """
        Retrive the quaternion that is closes in time to timestamp: find the two closest timestamps and SLERP the corresponding quaternions 
    """
    time_diff = np.array(dl.timestamps) - timestamp
    assert len(time_diff) == len(dl.timestamps), 'time_diff: {}, timestamps {}'.format(len(time_diff), len(dl.timestamps))
    assert max(dl.timestamps) > timestamp, 'timestamp: {}, max: {}'.format(timestamp, max(dl.timestamps))
    sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0][0])

    t1 = dl.timestamps[sign_switch_idx]
    t2 = dl.timestamps[sign_switch_idx + 1]

    assert t1 <= timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
    assert t2 >= timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

    q1 = dl.quaternions[sign_switch_idx]
    q2 = dl.quaternions[sign_switch_idx + 1]
    q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1)).normalised

    return q_

def compute_frames_KRK(timestamp:int, qs:list[Quaternion], dl:data_loader): 
    """
        Function rotating multiple frames, and outputs these frames 
    """
    image = cv2.imread(f'{dl.long_data_path}{str(timestamp)}.png', cv2.IMREAD_UNCHANGED)
    frames = []

    frames.append(image) 
    for i in range(len(qs)-1):
        q_delta = qs[i+1]/qs[i]  
        q_delta = q_delta.normalised 
        
        R_delta = q_delta.rotation_matrix

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
    frames = compute_frames_KRK(timestamp1, qs, dl)
    avg_image = frames[0]
    for i in range(len(frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(frames[i], alpha, avg_image, beta, 0.0)
    cv2.imwrite(f'{dl.result_path}/{str(timestamp1)}.png', avg_image)

def nextnonexistent(file):
    root, ext = os.path.splitext(file)
    i = 0
    while os.path.exists(file):
        i += 1
        file = f'{root}_{i}{ext}'
    return file

def create_list_of_quaternions(timestamp1:int, dt:float, dl:data_loader): 
    assert max(dl.timestamps) > timestamp1, '1. timestamp: {}, max: {}'.format(timestamp1, max(dl.timestamps))
    q1 = get_quaternion(timestamp1, dl)
    assert max(dl.timestamps) > timestamp1-dt, '2. timestamp: {}, max: {}'.format(timestamp1-dt, max(dl.timestamps))
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
    parser.add_argument('data', type=str, help='the data folder in the dataset path') 
    parser.add_argument('output', type=str, help='the path in which the resulting frames should be stored') 
    parser.add_argument('--div', default="1 2 3 4", type=str, help='The division factor that the time intervals are divided into (default: "1 2 3 4")')

    args = parser.parse_args()
    data_sub_path = args.data 
    top_result_path = args.output 
    divisions = [float(j) for j in args.div.split()]
    print(divisions)

    dl = data_loader(data_sub_path=data_sub_path, top_result_path=top_result_path)
    frames = sorted(os.listdir(os.path.join(f'{DATASETS_PATH}{data_sub_path}{FRAMES_SUB_PATH}')))

    for div in divisions: 
        div_name = str(div).replace('.','-')
        results_path = f'{top_result_path}/{dl.dataset_name}_blur_{div_name}'
        results_path = nextnonexistent(results_path)
        os.mkdir(results_path)
        dl.set_result_path(result_path=results_path)
        print(f'The results will be saved in the following path: {dl.result_path}')

        quats = {}
        print(f'Creating list of quaternions for each image timeinterval divided by {div}, the first and last frame are not included in the rendering')
        for i in tqdm(range(len(frames)-2)): 
            timestamp0 = int(frames[i].replace(".png", ""))
            timestamp1 = int(frames[i+1].replace(".png", ""))
            timestamp2 = int(frames[i+2].replace(".png", ""))
            dt1 = (timestamp1-timestamp0)/div
            dt2 = (timestamp1-timestamp2)/div
            #print(f'{dt1}, {dt2}')
            if timestamp1 <= max(dl.timestamps):
                qs1 = create_list_of_quaternions(timestamp1, dt1, dl)
                qs2 = create_list_of_quaternions(timestamp1, dt2, dl)
                qs = qs1[::-1] + qs2[1:]
                quats[timestamp1] = qs
            else:
                print(f'\n[WARNING] since time for image {frames[i+1]} larger than the largest mocap time, it will be excluded!!')
                frames.remove(frames[i+1])

        # add first and last frame to the result folder - will not be blurred
        print(f'Adding first and last frame to {dl.result_path}..')
        shutil.copy(f'{dl.long_data_path}{frames[0]}', f'{results_path}/{frames[0]}')
        shutil.copy(f'{dl.long_data_path}{frames[-1]}', f'{results_path}/{frames[-1]}')    

        print('Rendering motion blur ...')
        with multiprocessing.Pool(NUMBER_PROC) as pool:
            results = []
            for frame in tqdm(frames[1:], desc="Processing frames", total=len(frames[1:])):
                result = pool.apply_async(create_frames_parallel, args=(frame, quats, dl,))
                results.append(result)
            for result in tqdm(results, desc="Waiting for results", total=len(results)):
                result.wait()
        
        print(f'Created {len(os.listdir(results_path))} files')


if __name__ == '__main__': 
    main()


