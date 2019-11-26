from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os,cv2

class rubbishDataSet(Dataset):
    def __init__(self, file_path, kfold_data = None, transform = None, is_train = True):
        self.transform = transform
        self.file_path = file_path
        # select .jpg format
        self.jpg_file = self.select_format_file(self.file_path)
        self.is_train = is_train
        self.kfold_data = kfold_data
        
        
    def select_format_file(self, path, format_ = 'jpg'):
        return  [i for i in os.listdir(self.file_path) if i.split('.')[1] == format_]
    
    def __getitem__(self, idx):
        if self.kfold_data is None:
            img_name = self.jpg_file[idx]
        else:
            img_name = self.kfold_data[idx]
        img_path = os.path.join(self.file_path, img_name)
        
        # some image cann't open by pil
        try:
            img = Image.open(img_path)
        except:
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
        
        # some image mode is not RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
#         print(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        
        
        if self.is_train:
            label = 1 if img_name[:2] == '10' else 0
            return img, label
        else:
            return img, 'pic_'+img_name[:-4]
    
    def __len__(self):
        if self.kfold_data is None:
            return len(self.jpg_file)
        else:
            return len(self.kfold_data)