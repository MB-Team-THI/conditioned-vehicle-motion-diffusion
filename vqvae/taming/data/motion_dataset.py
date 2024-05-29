import glob
import random
import scipy
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, path, p = None):
        self.dataset_path = path
        file_list = glob.glob(self.dataset_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for sample_path in glob.glob(class_path + "/*.mat"):
                self.data.append([sample_path, class_name])
        if p is not None:
            idxs = sorted(random.sample(range(0, len(self.data)), int((p)*len(self.data))), reverse=True)
            data = [self.data.pop(i) for i in idxs]
            self.data = data
        self.class_map = {"class0" : 0, "class1": 1, "class2": 2}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, class_name = self.data[idx]
        keys = ['observed_data_x','observed_data_y', 'observed_data_vx', 'observed_data_vy', 'predicted_x', 'predicted_y']
        keys = ['observed_data_x','observed_data_y', 'observed_data_vx', 'observed_data_vy']
        scenario_id = path.rsplit('/',1)[1].replace('.mat','')
        sclist = [scipy.io.loadmat(path).get(key) for key in keys]
        class_id = self.class_map[class_name]
        return {
            keys[0]: sclist[0],
            keys[1]: sclist[1],
            keys[2]: sclist[2],
            keys[3]: sclist[3],
            'scenario_id': scenario_id}

class NormDataset():
    def __init__(self, norm_vals):
        norm_x, norm_y, norm_vx, norm_vy = norm_vals[0], norm_vals[1], norm_vals[2], norm_vals[3]
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.norm_vx = norm_vx
        self.norm_vy = norm_vy
    
    def norm_data(self, dataX, dataY, dataVX, dataVY):
        return dataX/self.norm_x, dataY/self.norm_y, dataVX/self.norm_vx, dataVY/self.norm_vy

    def inv_norm_data(self, dataX, dataY, dataVX, dataVY):
        return dataX*self.norm_x, dataY*self.norm_y, dataVX*self.norm_vx, dataVY*self.norm_vy
    


    