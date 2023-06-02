import os 
import yaml
import numpy as np 
import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt 
import seaborn as sns
"""plt.style.use('seaborn-v0_8')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False, 
    'font.weight': 'bold'
})"""
plt.rcParams['figure.figsize'] = [18, 16]


time_factor = {
    'dataset-room1_512_16' : 0, 
    'dataset-room1_512_16_20' : 20, 
    'dataset-room1_512_16_8' : 8,
    'dataset-room1_512_16_7' : 7,
    'dataset-room1_512_16_6' : 6, 
    'dataset-room1_512_16_5' : 5, 
    'dataset-room1_512_16_4' : 4, 
    'dataset-room1_512_16_3' : 3, 
    'dataset-room1_512_16_2-5' : 2.5, 
    'dataset-room1_512_16_2' : 2,
    'dataset-room2_512_16' : 0, 
    'dataset-room2_512_16_20' : 20, 
    'dataset-room2_512_16_8' : 8,
    'dataset-room2_512_16_7' : 7,
    'dataset-room2_512_16_6' : 6, 
    'dataset-room2_512_16_5' : 5, 
    'dataset-room2_512_16_4' : 4, 
    'dataset-room2_512_16_3' : 3, 
    'dataset-room2_512_16_2-5' : 2.5, 
    'dataset-room2_512_16_2' : 2,
    'dataset-room3_512_16' : 0, 
    'dataset-room3_512_16_20' : 20, 
    'dataset-room3_512_16_8' : 8,
    'dataset-room3_512_16_7' : 7,
    'dataset-room3_512_16_6' : 6, 
    'dataset-room3_512_16_5' : 5, 
    'dataset-room3_512_16_4' : 4, 
    'dataset-room3_512_16_3' : 3, 
    'dataset-room3_512_16_2-5' : 2.5, 
    'dataset-room3_512_16_2' : 2,
}

path_to_results = '/home/cm2113/workspace/results'
x = [0.0, 1.0/20.0, 1.0/8.0, 1./7., 1./6., 1./5., 1./4., 1./3., 1./2.5, 1./2.]

DATASETS_1 = ['dataset-room1_512_16', 'dataset-room1_512_16_20', 'dataset-room1_512_16_8', 'dataset-room1_512_16_7', 'dataset-room1_512_16_6', 'dataset-room1_512_16_5', 'dataset-room1_512_16_4', 'dataset-room1_512_16_3', 'dataset-room1_512_16_2-5', 'dataset-room1_512_16_2']  
DATASETS_2 = ['dataset-room2_512_16', 'dataset-room2_512_16_20', 'dataset-room2_512_16_8', 'dataset-room2_512_16_7', 'dataset-room2_512_16_6', 'dataset-room2_512_16_5', 'dataset-room2_512_16_4', 'dataset-room2_512_16_3', 'dataset-room2_512_16_2-5', 'dataset-room2_512_16_2']
DATASETS_3 = ['dataset-room3_512_16', 'dataset-room3_512_16_20', 'dataset-room3_512_16_8', 'dataset-room3_512_16_7', 'dataset-room3_512_16_6', 'dataset-room3_512_16_5', 'dataset-room3_512_16_4', 'dataset-room3_512_16_3', 'dataset-room3_512_16_2-5', 'dataset-room3_512_16_2']  

def remove_outliers(d:list[float], name:str): 
    d = np.array(d)
    q1 = np.percentile(d,25)
    q3 = np.percentile(d,75) 
    IQR = q3- q1 
    N = len(d)
    d = d[(d >= q1-1.5*IQR)]
    d = d[(d <= q3 + 1.5*IQR)]
    print(f'{N-len(d)} outliers removed from {name}')
    return d


def extract_results(data_folder,chop:int=0,align:str='sim3'):
    runs = sorted(os.listdir(data_folder))
    sorted_runs = sorted(runs, key=lambda x: int(x.split('_')[1]))
    sorted_runs = sorted_runs[chop:]
    scale_error, trans_error = [], []
    for run in sorted_runs: 
        path = f'{data_folder}/{run}/saved_results/traj_est/absolute_err_statistics_{align}_-1.yaml'
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.loader.FullLoader)
            if data["trans"]["num_samples"] < 500: 
                print(f' number of samples: {data["trans"]["num_samples"]} path: {path}')
            else: 
                scale_error.append(data["scale"]['rmse'])
                trans_error.append(data['trans']['rmse'])
                
            #print(f"run: {run}: trans rmse {data['trans']['rmse']}")
    return scale_error, trans_error

def plot_data_fill(mean, std, x_vec, imu):
    upper_limit = np.array(mean) + np.array(std) 
    lower_limit = np.array(mean) - np.array(std)
    plt.title(f'Translational error, with 10 itterations for each sequence')
    plt.plot(x_vec, mean, label = f'mean - {imu}')
    plt.fill_between(x_vec, upper_limit, lower_limit, alpha=0.3, label='sd')
    plt.xlabel('Time factor')
    plt.ylabel('Translational RMSE')

def plot_all_sequences(data:dict, dataset:list=DATASETS_1, imu:list=['withimu', 'withoutimu']):
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(25, 10))
    plt.suptitle('Each sequence RMSE w.r.t run')
    count = 0 
    max_val = 1 
    for i in range(2):
        for j in range(5):
            axis = ax[i,j]
            axis.set_title(f'{dataset[count]}')
            for i_m_u in imu:
                color = 'steelblue' if i_m_u=='withimu' else 'seagreen'
                axis.plot(data[f'{dataset[count]}{i_m_u}'],  label=i_m_u, lw=1, color=color)
                max_val = max(data[f'{dataset[count]}{i_m_u}']) if max(data[f'{dataset[count]}{i_m_u}'])>max_val else max_val
            axis.legend()
            axis.set_xlabel('run') 
            axis.set_ylabel('Translational RMSE')
            axis.set_ylim(0,max_val)
            count += 1 

def plot_all_percentiles(data:dict, dataset:list=DATASETS_1, imu:list=['withimu', 'withoutimu']):
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(25, 10))
    plt.suptitle('Each sequence RMSE w.r.t percentile - 50 runs')
    count = 0 
    max_val = 1 
    for i in range(2):
        for j in range(5):
            axis = ax[i,j]
            axis.set_title(f'{dataset[count]}')
            for i_m_u in imu:
                color = 'steelblue' if i_m_u=='withimu' else 'seagreen'
                pctiles = np.percentile(data[f'{dataset[count]}{i_m_u}'], np.arange(0, 101))
                axis.plot(pctiles,  label=i_m_u, lw=1, color=color)
                max_val = max(pctiles) if max(pctiles)>max_val else max_val
            axis.legend()
            axis.set_xlabel('Percentile') 
            axis.set_ylabel('Translational RMSE')
            axis.set_ylim(0,max_val)
            count += 1 
            
# at, only returning translational error statistics 
def compute_results(imu, system, sub='', datasets=DATASETS_1, chop=0, outliers=True): 
    res_s, res_t = {}, {}
    std_s, std_t = {}, {}
    runs = {}
    #datasets = [dataset] if dataset else DATASETS_1 
    
    for i in imu: 
        tot_s, tot_t = [], []
        s_s, s_t = [], []
        for dataset in datasets:
            data_folder = f'{path_to_results}/{dataset}/{system}/data{sub}/{i}'
            align = 'se3' if i=='withimu' else 'sim3' 
            s, t = extract_results(data_folder, chop, align)
            if not outliers: 
                t = remove_outliers(t, dataset+i)
            runs[dataset+i] = t
            tot_s.append(np.median(s))
            tot_t.append(np.median(t))
            s_s.append(np.std(s))
            s_t.append(np.std(t))
        res_s[i] = tot_s 
        res_t[i] = tot_t
        std_s[i] = s_s 
        std_t[i] = s_t
    return res_t, std_t, runs 

def get_scale_error(imu, system, sub='', datasets=DATASETS_1, chop=0,outliers=True): 
    res_s = {}
    std_s = {}
    runs = {}
    #datasets = [dataset] if dataset else DATASETS_1 
    
    for i in imu: 
        tot_s = []
        s_s = [] 
        for dataset in datasets:
            data_folder = f'{path_to_results}/{dataset}/{system}/data{sub}/{i}'
            align = 'se3' if i=='withimu' else 'sim3' 
            s, t = extract_results(data_folder, chop, align)
            if not outliers: 
                s = remove_outliers(s, dataset+i)
            runs[dataset+i] = s
            tot_s.append(np.median(s))
            s_s.append(np.std(s))
        res_s[i] = tot_s 
        std_s[i] = s_s 
    return res_s, std_s, runs
    
def read_octaves(path:str):
    data = []
    with open(path, 'r') as file: 
        for line in file:
            values = line.strip().split()
            values = [int(x) if x else 0 for x in values]
            data.append(values)
    return np.array(data)


def extract_orbs(imu:str, system, sequence:str, sub='', chop=0,nocts=8):
    octaves_runs = {}
    data_folder = f'{path_to_results}/{sequence}/{system}/data{sub}/{imu}'
    runs = sorted(os.listdir(data_folder))
    sorted_runs = sorted(runs, key=lambda x: int(x.split('_')[1]))
    sorted_runs = sorted_runs[chop:]
    
    for run in sorted_runs: 
        path = f'{data_folder}/{run}/octaves.txt'
        octaves_runs[run] = read_octaves(path)
    return octaves_runs