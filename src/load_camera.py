import yaml 
import numpy as np

class camera:
    def __init__(self, cam_config_path):
        self.path = cam_config_path
        self.intr = None
        self.K = None 
        self.D = None 
        self.DIM = None
        self._load_camera_model()
    
    def __str__(self): 
        return f'Path to camera config: {self.path} and intrinsics: {self.intr}, and distorsion coefficients: {self.D}, dimensions: {self.DIM}'
    
    def _load_camera_model(self):
        with open(self.path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.intr = data['cam0']['intrinsics'] 
            d = data['cam0']['distortion_coeffs']
            self.DIM = tuple(data['cam0']['resolution'])

        self.K = np.array([[self.intr[2], 0, self.intr[0]], [0, self.intr[3], self.intr[1]], [0, 0, 1]])
        self.D = np.array(d)

    def get_intrinsics_array(self): 
        return self.intr.copy()
    
    def get_distortion_coefficients(self):
        return self.D.copy()
    
    def get_intrinsics_matrix3D(self): 
        return self.K.copy()
    
    def get_dim(self): 
        return self.DIM.copy()
    
    
if __name__ == '__main__': 
    test_path = './camchain.yaml'
    cam = camera(test_path)
    print(cam)
    print(cam.K)
