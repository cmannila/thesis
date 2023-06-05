import os
import shutil
from tqdm import tqdm 
import yaml
from pathlib import Path

""" 
    Retrive old results to get the scaled data from DM VIO 
"""
sub = 'result_runs_pc_nexp'
ds = [2,3]
path_to_dmvio_results = '/home/cm2113/workspace/dm-vio/results/'

for d in ds: 
    tum_vi = f'tumvi_blur_{str(d)}'
    dataset=f'dataset-room{str(d)}_512_16'
    blurs = ['', '_2-5', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_20']

    for idx, blur in enumerate(blurs):
        cmd = f'run_dmvio.py --output=save --dataset={tum_vi} --dmvio_settings=tumvi.yaml --iter=1 --only_seq={idx} --result=True --sub_part=/{sub} --run='

        my_runs = {}

        for path in tqdm(list(Path(path_to_dmvio_results).glob(f"*{tum_vi}*/setup/setup.yaml"))):
            with path.open("r") as f:
                data = yaml.load(f, Loader=yaml.loader.FullLoader)

                if data['eval_tool_command'].startswith(cmd):
                    i = int(data["eval_tool_command"].replace(cmd,"")[:2])
                    my_runs[i] = os.path.dirname(os.path.dirname(path))
    
            
        for k,v in sorted(list(my_runs.items()), key=lambda x: x[0]):
            path = v 
            old_path = f'/home/cm2113/workspace/results/{dataset}{blurs[idx]}/dm_vio/data/{sub}/withimu/run_{k}'
            if os.path.exists(f'{path}/tumvi_{dataset}{blurs[idx]}_0/resultScaled.txt'):
                shutil.copy(f'{path}/tumvi_{dataset}{blurs[idx]}_0/resultScaled.txt', f'{old_path}/resultScaled.txt')
            else: 
                with open('lost_data.txt', 'a') as f:
                    f.write(f'{cmd}{k}\n')
                    print(f'{cmd}{k}')

