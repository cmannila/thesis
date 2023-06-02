import cv2
import os
ds = '3'
PATH = f'/home/cm2113/workspace/Datasets/tumvi/room/dataset-room{ds}_512_16/mav0/cam0/data/'
BLURRED_PATH = f'/home/cm2113/workspace/Datasets/tumvi/room/dataset-room{ds}_512_16_2/mav0/cam0/data/'

DISPLAY = False
SEQUENCE = True

KERNEL_S = 15
SAVE_PATH = f'/home/cm2113/workspace/thesis/results/tumvi_room{ds}_blur/dataset-room{ds}_512_16_gblur_{KERNEL_S}_1/'



#image = '1520530329850104890.png'
# Load the image
if SEQUENCE:  
    os.mkdir(SAVE_PATH) if not os.path.exists(SAVE_PATH) else None 
    frames = os.listdir(PATH)
    for image in frames: 
        img = cv2.imread(os.path.join(PATH, image), cv2.IMREAD_UNCHANGED)
        img_pre_blurred = cv2.imread(os.path.join(BLURRED_PATH, image))
        
        # Apply Gaussian blur with kernel size of (5,5) and sigma value of 0
        blur = cv2.GaussianBlur(img, (KERNEL_S,KERNEL_S), 0)
        cv2.imwrite(os.path.join(SAVE_PATH, image), blur)
else: 
    image = '1520530329850104890.png'
    img = cv2.imread(os.path.join(PATH, image))
    img_pre_blurred = cv2.imread(os.path.join(BLURRED_PATH, image))
    
    # Apply Gaussian blur with kernel size of (5,5) and sigma value of 0
    blur = cv2.GaussianBlur(img, (KERNEL_S,KERNEL_S), 0)
    cv2.imwrite(os.path.join(SAVE_PATH, image), blur)

if DISPLAY: 
    # Display the original and blurred image
    cv2.imshow('Original Image', img)
    cv2.imshow('Original pre blurred Image', img_pre_blurred)
    cv2.imshow('Blurred Image', blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()