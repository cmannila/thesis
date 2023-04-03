import csv
import numpy as np 
import pandas as pd 
from pyquaternion import Quaternion

DATASETS_PATH = '/home/cm2113/workspace/Datasets'
CAM_SUB_PATH =  '/dso/camchain.yaml'
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

timestamps, positions, quaternions = load_gt_data(f'{DATASETS_PATH}/tumvi/room/dataset-room1_512_16{TIME_SUB_PATH}') 
print(f'len timestamps: {len(timestamps)}, len positions {len(positions)} and len quaternions {len(quaternions)}')

T_imu_cam = np.array([-0.9995250378696743, 0.0075019185074052044, -0.02989013031643309, 0.045574835649698026, 
          0.029615343885863205, -0.03439736061393144, -0.998969345370175, -0.071161801837997044,
         -0.008522328211654736, -0.9993800792498829, 0.03415885127385616, -0.044681254117144367,
          0.0, 0.0, 0.0, 1.0])

T_imu_cam = T_imu_cam.reshape((4,4))

with open('gt_test.csv', 'w', encoding='utf-8') as f:
    #print('# timestamp tx ty tz qx qy qz qw', file=f)
    writer = csv.writer(f)
    writer.writerow('#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []'.split(','))
    #print('#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []', file=f)

    for time, pose, quat in zip(timestamps, positions, quaternions): 

        T_imu = np.eye(4)
        T_imu[:3,:3] = quat.rotation_matrix 
        T_imu[:3, 3] = pose 

        #T_cam = np.linalg.inv(T_imu_cam) @ T_imu
        T_cam = T_imu @ T_imu_cam 

        q_cam = Quaternion(matrix=T_cam[:3,:3])
        p_cam = T_cam[:3, 3]
        #print(f'{time/1e9},{p_cam[0]},{p_cam[1]},{p_cam[2]},{q_cam.w},{q_cam.x},{q_cam.y},{q_cam.z}', file=f)
        row = [time, p_cam[0],p_cam[1],p_cam[2], q_cam.w, q_cam.x, q_cam.y, q_cam.z] # = p_cam.append(q_cam.w) # + [q_cam.w] + [q_cam.x] + [q_cam.y] + [q_cam.z]
        writer.writerow(row)
f.close()

