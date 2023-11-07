import torch
import os
import glob

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, name):
        gt_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_gt'
        ours_color_lines_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb_line'
        ours_color_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb'
        ours_sate_semantic=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\sate_rgb_seg'
        render_list=r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_pair_test.txt'
        self.name=name
        self.pred_paths=[]
        self.pred_semantic_paths=[] # semantic map is estimated based on the predicted reuslt using OneFormer, label=1 is building. It is already very robust and can be regarded as the semantic metric 
        self.gt_paths=[]
        self.gt_semantic_paths=[] # semantic map is estimated based on the street rgb using OneFormer, label=1 is building. It is already very robust and can be regarded as the semantic metric 

        data_in=open(render_list,'r')
        lines=data_in.readlines()

        if name=='ours_color_lines':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=glob.glob(os.path.join(ours_color_lines_dir,line+'_*.png'))
                pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_color':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=glob.glob(os.path.join(ours_color_dir,line+'_*.png'))
                pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='sate_rgb_seg':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=glob.glob(os.path.join(ours_sate_semantic,line+'_*.png'))
                pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        data_in.close()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.pred_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.pred_paths[index],self.pred_semantic_paths[index],self.gt_paths[index],self.gt_semantic_paths[index]

dataset_names=['ours_color_lines','ours_color','sate_rgb_seg']
dataset=Dataset(dataset_names[2])
dataloader = torch.utils.data.DataLoader(dataset)
for id,data in enumerate(dataloader):
    pred_path=data[0]
    pred_semantic_path=data[1]
    gt_path=data[2]
    gt_semantic_path=data[3]
    print("pred_path:{}".format(pred_path))
    print("pred_semantic_path:{}".format(pred_semantic_path))
    print("gt_path:{}".format(gt_path))
    print("gt_semantic_path:{}".format(gt_semantic_path))


