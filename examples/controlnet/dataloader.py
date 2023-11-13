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
        pano_semantic=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_semantics'
        pano_rgb=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_rgb'
        crossmlp_semantic=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_semantics'
        crossmlp_rgb=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_rgb'


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
                #pred_path=pred_paths[0]
                #self.pred_paths.append(pred_path)
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_color':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=glob.glob(os.path.join(ours_color_dir,line+'_*.png'))
                pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_satergb_seg':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=glob.glob(os.path.join(ours_sate_semantic,line+'_*.png'))
                pred_paths=[file_path for file_path in pred_paths if "_seg.png" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='pano_rgb':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(pano_rgb,line+'_sate_rgb_fake_B_final.png')]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='pano_semantic':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(pano_semantic,line+'_sate_rgb_fake_B_final.png')]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='crossmlp_rgb':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(crossmlp_rgb,line+'_sate_rgb_I.png')]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='crossmlp_semantic':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(crossmlp_semantic,line+'_sate_rgb_I.png')]
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

# dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
# dataset=Dataset(dataset_names[0])
# dataloader = torch.utils.data.DataLoader(dataset)
# for id,data in enumerate(dataloader):
#     pred_path=data[0][0]
#     pred_semantic_path=data[1]
#     gt_path=data[2][0]
#     gt_semantic_path=data[3]
#     if not os.path.exists(pred_path) or not os.path.exists(gt_path):
#         print(pred_path)



