import os 

PATH = '/home/cm2113/workspace/thesis/results/tumvi_room1_blur/tumvi_room1_512_16_blur_6/'

files = os.listdir(PATH)
print(f'Old file names: {files[:10]}')

for file in files: 
    path = os.path.join(PATH, file)
    new_path = os.path.join(PATH, file.replace("_avg", ""))
    os.rename(path, new_path)

new_files = os.listdir(PATH)
print(f'New file names: {new_files[:10]}')
