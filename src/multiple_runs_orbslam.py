import os 
import argparse
import subprocess
import shutil 
"""./Examples/Monocular/mono_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM-VI.yaml ../Datasets/tumvi/room/dataset-room1_512_16_2/mav0/cam0/data ./Examples/Monocular/TUM_TimeStamps/dataset-room1_512.txt dataset-tum-test
./Examples/Monocular/mono_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM-VI.yaml ../Datasets/tumvi/room/dataset-room1_512_16/mav0/cam0/data ./Examples/Monocular/TUM_TimeStamps/dataset-room1_512_16.txt dataset-room1_512_16_0
"""

PATH_DATASETS = '/home/cm2113/workspace/Datasets/tumvi/room'
ORB_SLAM_PATH = '/home/cm2113/workspace/ORB_SLAM3'
DATASETS_1 = ['dataset-room1_512_16', 'dataset-room1_512_16_8', 'dataset-room1_512_16_7', 'dataset-room1_512_16_6', 'dataset-room1_512_16_5', 'dataset-room1_512_16_4', 'dataset-room1_512_16_3', 'dataset-room1_512_16_2', 'dataset-room1_512_16_2-5', 'dataset-room1_512_16_20'] # SPECIFIC 'dataset-room1_512_16', 'dataset-room1_512_16_8', 'dataset-room1_512_16_7', 'dataset-room1_512_16_6', 'dataset-room1_512_16_5', 'dataset-room1_512_16_4', 'dataset-room1_512_16_3', 'dataset-room1_512_16_2'
DATASETS_2 = ['dataset-room2_512_16', 'dataset-room2_512_16_8', 'dataset-room2_512_16_7', 'dataset-room2_512_16_6', 'dataset-room2_512_16_5', 'dataset-room2_512_16_4', 'dataset-room2_512_16_3', 'dataset-room2_512_16_2', 'dataset-room2_512_16_2-5', 'dataset-room2_512_16_20']
DATASETS_3 = ['dataset-room3_512_16', 'dataset-room3_512_16_8', 'dataset-room3_512_16_7', 'dataset-room3_512_16_6', 'dataset-room3_512_16_5', 'dataset-room3_512_16_4', 'dataset-room3_512_16_3', 'dataset-room3_512_16_2', 'dataset-room3_512_16_2-5', 'dataset-room3_512_16_20']



def main():
    parser = argparse.ArgumentParser(description='Run ORB-SLAM3 multiple times and deal with saving and extracting results')
    parser.add_argument('--iter', default=10, type=int, help='the number of iterations for each sequence, default 10')
    parser.add_argument('--imu', default="1", type=str, help='ORB-SLAM wit or without imu, default 1')
    parser.add_argument('--seq', type=int, default=None, help='sequence to run, default None, i.e., run all')
    parser.add_argument('--sub', type=str, default="")
    parser.add_argument('--ds', default=1, type=int)
    
    args = parser.parse_args()
    iter = args.iter
    imu = args.imu 
    seq = args.seq
    imu = [int(i) for i in imu.split()]
    sub = args.sub
    
    if args.ds == 1:
        datas = [DATASETS_1]
        ds = ['1']
    elif args.ds == 2:
        datas = [DATASETS_2]
        ds = ['2']
    elif args.ds == 3:
        datas = [DATASETS_3]
        ds = ['3']
    elif args.ds == -1:
        datas = [DATASETS_1, DATASETS_2, DATASETS_3]
        ds = ['1', '2', '3']

    for idx, data in enumerate(datas):
        if seq is not None:
            datasets = [data[seq]]
        else:
            datasets = data
        for im in imu:
            if im: 
                sub_path = 'Monocular-Inertial'
                sub_run_path = 'mono_inertial_tum_vi'
                sub_save ='withimu'
            else: 
                sub_path = 'Monocular'
                sub_run_path = 'mono_tum_vi'
                sub_save ='withoutimu'

            
            run_path = f'{ORB_SLAM_PATH}/Examples/{sub_path}/{sub_run_path}'
            vocabulary_path = f'{ORB_SLAM_PATH}/Vocabulary/ORBvoc.txt'
            yaml_path = f'{ORB_SLAM_PATH}/Examples/{sub_path}/TUM-VI.yaml'
            timestamps = f'{ORB_SLAM_PATH}/Examples/{sub_path}/TUM_TimeStamps/dataset-room{ds[idx]}_512.txt'
            

            for data in datasets:
                if sub != "": 
                    t = f'/home/cm2113/workspace/results/{data}/orb_slam/data{sub}'
                    os.mkdir(t) if not os.path.exists(t) else None
                    print(f'gone in here {t}')
                save_path = f'/home/cm2113/workspace/results/{data}/orb_slam/data{sub}/{sub_save}'
                os.mkdir(save_path) if not os.path.exists(save_path) else None 
                # to enable run cuts 
                if os.path.exists(save_path):
                    j = len(os.listdir(save_path))
                    print(f'{j} number of files already created, start from... {j+1}')
                else:
                    j=0
                for i in range(iter-j): 
                    i += j 
                    save_path = f'/home/cm2113/workspace/results/{data}/orb_slam/data{sub}/{sub_save}/run_{i}'
                    os.mkdir(save_path) if not os.path.exists(save_path) else None 
                    if im: 
                        cmd = f'{run_path} {vocabulary_path} {yaml_path} {PATH_DATASETS}/{data}/mav0/cam0/data {timestamps} {PATH_DATASETS}/{data}/mav0/imu0/data.csv {data}_0'
                    else: 
                        cmd = f'{run_path} {vocabulary_path} {yaml_path} {PATH_DATASETS}/{data}/mav0/cam0/data {timestamps} {data}_0'
                    print(cmd)
                    subprocess.run(cmd.split())

                    saving_name = f'f_{data}_0.txt'
                    if os.path.exists(f'{ORB_SLAM_PATH}/{saving_name}'):
                        print("FILE DOES NOT EXISTS")
                        shutil.move(f'{ORB_SLAM_PATH}/{saving_name}', f'{save_path}/{saving_name}')
                        shutil.move(f'{ORB_SLAM_PATH}/octaves.txt', f'{save_path}/octaves.txt')
                    else:
                        with open(f'{ORB_SLAM_PATH}/{saving_name}', 'w') as fp:
                            pass
                groundtruth = 'o' if im else 't'
                cmd = f'python /home/cm2113/workspace/dm-vio-python-tools/compute_results.py {data} orb_slam {im} --sub={sub} --gt={groundtruth}'
                subprocess.run(cmd.split())

            


            

if __name__=='__main__': 
    main() 


