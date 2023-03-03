import numpy as np
from pyquaternion import Quaternion
import cv2
import pandas as pd 
import os 
from tqdm import tqdm

FX = 190.97847715128717
FY = 190.9733070521226
CX = 254.93170605935475
CY = 256.8974428996504

CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
CAM_D = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
DIM = (512,512)

PATH = './image_sequence/' 

def load_data(filename):
    data = pd.read_csv(filename).sort_values(by=['#timestamp [ns]']).rename(columns={' p_RS_R_x [m]':'p_x', ' p_RS_R_y [m]':'p_y', ' p_RS_R_z [m]':'p_z', ' q_RS_w []':'q_w', ' q_RS_x []':'q_x', ' q_RS_y []':'q_y', ' q_RS_z []':'q_z' })
    timestamps = data['#timestamp [ns]']
    positions = []
    quaternions = []
    for _, row in data.iterrows(): 
        positions.append([row.p_x, row.p_y, row.p_z])
        quaternions.append(Quaternion(row.q_w, row.q_x, row.q_y, row.q_z))

    return timestamps, positions, quaternions


def get_quaternion(timestamp:int): 
    time_diff = np.array(timestamps) - timestamp

    sign_switch_idx = int(np.where(np.diff(np.sign(time_diff)))[0])

    t1 = timestamps[sign_switch_idx]
    t2 = timestamps[sign_switch_idx + 1]

    assert t1 < timestamp, "t1: {}, timestamp: {}".format(t1, timestamp)
    assert t2 > timestamp, "t2: {}, timestamp: {}".format(t2, timestamp)

    q1 = quaternions[sign_switch_idx]
    q2 = quaternions[sign_switch_idx + 1]
    q_ = Quaternion.slerp(q1, q2, (timestamp - t1) / (t2 - t1))

    return q_, sign_switch_idx

def create_list_of_quaternions(timestamp1:int, timestamp2:int): 
    q1, _ = get_quaternion(timestamp1)
    q2, _ = get_quaternion(timestamp2)
    qs = get_mid_quaternion(q1,q2)
    return qs

def get_mid_quaternion(q1:Quaternion, q2:Quaternion, tol:float=1e-3): 
    if Quaternion.absolute_distance(q1,q2) < tol:
        return [q1, q2]
    q_mid = Quaternion.slerp(q1, q2)
    left_points = get_mid_quaternion(q1, q_mid)
    right_points = get_mid_quaternion(q_mid, q2)
    return left_points + right_points[1:]

def compute_frames(qs:list[Quaternion], image):
    frames = []
    frames.append(image)
     
    for i in range(len(qs)-1): 
        R1 = qs[i].rotation_matrix
        R2 = qs[i+1].rotation_matrix 

        H, _ = cv2.findHomography(np.array([[0,0,1],[0,1,1],[1,1,1],[1,0,1]]),
                            np.array([[0,0,1],[0,1,1],[1,1,1],[1,0,1]]).dot(R1).dot(np.linalg.inv(R2)))

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

def nextnonexistent(f):
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew

timestamps, position, quaternions = load_data('data.csv')

def get_frames(timestamp1:int, timestamp2:int):
    image1 = cv2.imread(f'{PATH}{str(timestamp1)}.png', cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(f'{PATH}{str(timestamp2)}.png', cv2.IMREAD_UNCHANGED)

    qs = create_list_of_quaternions(timestamp1=timestamp1, timestamp2=timestamp2)

    frames_image1 = compute_frames(qs, image1)
    frames_image2 = compute_frames(qs[::-1], image2)
    frames_image2 = frames_image2[::-1]

    frames = []
    for i, frame1 in enumerate(frames_image1): 
        frame2 = frames_image2[i]
        alpha = (len(frames_image1)-i)/len(frames_image1)/2
        beta = 1 - alpha

        frame = cv2.addWeighted(frame1, alpha, frame2, beta, 0.0)
        frames.append(frame)
    
    return frames

def main(): 
    """ sequence of images instead """
    frames = sorted(os.listdir(os.path.join(PATH)))

    result_path = './results'
    result_path = nextnonexistent(result_path)
    print(result_path)
    os.mkdir(result_path)

    frames_fom_before = None 

    for i,frame in tqdm(list(enumerate(frames[:-1]))): 
        if i==0: # just skip the first frame ? 
            timestamp1 = int(frame.replace(".png",""))  
            timestamp2 = int(frames[i+1].replace(".png", ""))
            inbetween_frames = get_frames(timestamp1=timestamp1, timestamp2=timestamp2)

            avg_image = inbetween_frames[0]
            for i in range(len(inbetween_frames)):
                if i == 0:
                    pass
                else:
                    alpha = 1.0/(i + 1)
                    beta = 1.0 - alpha
                    avg_image = cv2.addWeighted(inbetween_frames[i], alpha, avg_image, beta, 0.0)

            cv2.imwrite(f'{result_path}/{frame}', avg_image)
            frames_fom_before = inbetween_frames
        else:
            timestamp1 = int(frame.replace(".png",""))
            timestamp2 = int(frames[i+1].replace(".png", ""))

            frames1 = frames_fom_before
            frames2 = get_frames(timestamp1=timestamp1, timestamp2=timestamp2)

            #image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            inbetween_frames = frames1 + frames2

            avg_image = inbetween_frames[0]
            for i in range(len(inbetween_frames)):
                if i == 0:
                    pass
                else:
                    alpha = 1.0/(i + 1)
                    beta = 1.0 - alpha
                    avg_image = cv2.addWeighted(inbetween_frames[i], alpha, avg_image, beta, 0.0)

            cv2.imwrite(f'{result_path}/{frame}', avg_image)

    
    """save = False 
    if save: 
        if not os.path.exists('./frames1/'):
            os.makedirs('./frames1/')
        
        if not os.path.exists('./frames2/'):
            os.makedirs('./frames2/')

        if not os.path.exists('./frames/'):
            os.makedirs('./frames/')

        for i, frame in enumerate(frames_image1): 
            cv2.imwrite(f'./frames1/frame_{i}.png', frame)
            frame2 = frames_image2[i]
            cv2.imwrite(f'./frames2/frame_{i}.png', frame2)
            col_frame = frames[i]
            cv2.imwrite(f'./frames/frame_{i}.png', col_frame)


    cv2.imshow('first image', image1)

    avg_image = frames[0]
    for i in range(len(frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(frames[i], alpha, avg_image, beta, 0.0)

    cv2.imshow('average_frame', avg_image)

    cv2.imshow('last_frame', image2)
    frames.append(image2)
    average_np = False
    if average_np: 
        frames = np.array(frames).astype(np.float32)/65535
        avg_frame_float = np.mean(frames, axis=0)
        avg_frame2 = np.round(avg_frame_float*65535).astype(np.uint16)
        cv2.imshow('average frame', avg_frame2)
    
    cv2.waitKey(0)"""

if __name__ == '__main__':
    main()
    
    """image1_name = '1520530327700094100'
    image2_name = '1520530327750096100'

    timestamp1 = int(image1_name)
    timestamp2 = int(image2_name)

    #load images 
    image1 = cv2.imread(f'{image1_name}.png', cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(f'{image2_name}.png', cv2.IMREAD_UNCHANGED)

    height, width = image1.shape

    q1,_ = get_quaternion(timestamp1)
    q2,_ = get_quaternion(timestamp2)

    frames1 = compute_frames([q1,q2], image1)
    frames2 = compute_frames([q2,q1], image2)

    for i, frame1 in enumerate(frames1): 
        frame2 = frames2[i]
        cv2.imshow(f'frame 1 {i}', frame1)
        cv2.imshow(f'frame 2 {i}', frame2)"""
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#avg_image[avg_image<=1500] = 65520

# Display the result
#cv2.imshow('Result', avg_image)



"""if not os.path.exists('./frames/'):
    print('hello')
    os.makedirs('./frames/')

for i, frame in enumerate(frames):
    cv2.imwrite(f'./frames/frame_{i}.png', frame)"""