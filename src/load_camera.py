import yaml 

class camera:
    def __init__(self, cam_config_path):
        self.path = cam_config_path
        self.intr = None
        self.d = None
        self._load_camera_model
    
    def __str__(self): 
        return f'Path to camera config: {self.path} and intrinsics: {self.intr}, and distorsion coefficients: {self.d}'
    
    def _load_camera_model(self):
        with (self.path).open("r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.intr = data['cam0']['intrinsics'] 
            self.d = data['cam0']['distortion_coeffs']

    def intrinsics(self): 
        return self.intr.copy()
    
    def distortion(self):
        return self.d.copy()
    
if __name__ == '__main__': 
    test_path = '/home/cm2113/workspace/Datasets/tumvi/room/dataset-room1_512_16/dso/camchain.yaml'
    cam = camera(test_path)
    print(cam)

