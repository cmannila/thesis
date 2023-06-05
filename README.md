# Thesis 

This repository contains both the rendering of motion blur and evaluation of the results when running on these datasets [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) and [DM-VIO](https://github.com/lukasvst/dm-vio). The results were presented in my thesis.  

At first a file structure was created as: 

```
└── Results folder 
|    └── Dataset sequence 
|        └── dm_vio
|        |   └── ev: specific folder name: eg. without_pc
|        |        └── sensor_config: withimu or withoutimu 
|        |            └── run_X
|        |                ├── <resultfile>.txt
|        |                └── photometricerror.txt
|        └── orb_slam 
|        |   └── ev: specific folder name: eg. without_pc
|        |        └── sensor_config: withimu or withoutimu
|        |            └── run_X
|        |                ├── <resultfile>.txt
|        |                └── photometricerror.txt 
|        └── Groundtruth files 
|            └── withimu 
|		 └── groundtruth.txt 
|	     └── withoutimu 
|	         └── groundtruth.txt  
```

# Motion Blur Rendering 

The implementation is based on a rotational trajectory method that uses the ground truth provided in [TUM VI dataset](https://cvg.cit.tum.de/_media/spezial/bib/schubert2018vidataset.pdf). The code is given in src [rotational_motionblur.py](https://github.com/cmannila/thesis/blob/main/src/rotational_motionblur.py). 


## Gaussian blur 

Guassian blur was rendered to compare to the motion blur, and the short code-snippet is presented in [gaussian_blur.py](https://github.com/cmannila/thesis/blob/main/src/gaussina_blur.py)


# Evaluation 

The results were evaluated mainly in the scripts [evaluation.py](https://github.com/cmannila/thesis/blob/main/src/evaluation.py) and [eval_funcs.py](https://github.com/cmannila/thesis/blob/main/src/eval_funcs.py). 

To take into account the groundtruth was transformed into the camera frame for the Monocular configuration in [transform_groundtruth.py](https://github.com/cmannila/thesis/blob/main/src/transform_groundtruth.py). 
