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
        ours_color_new=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb_new'
        ours_lines=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_line'
        ours_color_lines_new=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb_line_new'
        ours_nolora=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_nolora'
        prior_hk=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\priors\hk'
        prior_dubai=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\priors\dubai'
        prior_paris=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\priors\paris'
        prior_london=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\priors\london'
        prior_gt=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\priors\gt'
        nopaired_sem=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\nopaired\sem'
        nopaired_rgb=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\nopaired\rgb'

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
        elif name=='ours_color_new':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(ours_color_new,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_lines':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(ours_lines,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_color_lines_new':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(ours_color_lines_new,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='ours_nolora':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(ours_nolora,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='nopaired_sem':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(nopaired_sem,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='nopaired_rgb':
            for line in lines:
                line=line.rstrip('\n')
                pred_paths=[os.path.join(nopaired_rgb,line+'.png')]
                #pred_paths=[file_path for file_path in pred_paths if "seg" not in file_path]
                #pred_path=pred_paths[0]
                self.pred_paths.append(pred_paths)
                semantic_list=[]
                for pred_path in pred_paths:
                    semantic_list.append(pred_path[:-4]+'_seg.png')
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(os.path.join(gt_dir,line+".jpg"))
                self.gt_semantic_paths.append(os.path.join(gt_dir,line+"_seg.png"))
        elif name=='prior_hk':
            gt_paths=glob.glob(os.path.join(prior_gt,"*"))
            for line in lines:
                line=line.rstrip('\n')
                pred_path=os.path.join(prior_hk,line+'.png')
                self.pred_paths.append(pred_path)
                semantic_list=[]
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(gt_paths)
                self.gt_semantic_paths.append([])
        elif name=='prior_dubai':
            gt_paths=glob.glob(os.path.join(prior_gt,"*"))
            for line in lines:
                line=line.rstrip('\n')
                pred_path=os.path.join(prior_dubai,line+'.png')
                self.pred_paths.append(pred_path)
                semantic_list=[]
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(gt_paths)
                self.gt_semantic_paths.append([])
        elif name=='prior_paris':
            gt_paths=glob.glob(os.path.join(prior_gt,"*"))
            for line in lines:
                line=line.rstrip('\n')
                pred_path=os.path.join(prior_paris,line+'.png')
                self.pred_paths.append(pred_path)
                semantic_list=[]
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(gt_paths)
                self.gt_semantic_paths.append([])
        elif name=='prior_london':
            gt_paths=glob.glob(os.path.join(prior_gt,"*"))
            for line in lines:
                line=line.rstrip('\n')
                pred_path=os.path.join(prior_london,line+'.png')
                self.pred_paths.append(pred_path)
                semantic_list=[]
                #semantic_path=semantic_list[0]
                self.pred_semantic_paths.append(semantic_list)
                self.gt_paths.append(gt_paths)
                self.gt_semantic_paths.append([])
        data_in.close()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.pred_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.pred_paths[index],self.pred_semantic_paths[index],self.gt_paths[index],self.gt_semantic_paths[index]

#dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic','ours_color_new','ours_lines',"ours_color_line_new","ours_nolora"]
#dataset_names=['ours_color_new','ours_lines',"ours_color_lines_new","ours_nolora"]
#dataset_names=['prior_hk','prior_dubai','prior_paris','prior_london']
dataset_names=['nopaired_sem','nopaired_rgb']
dataset=Dataset(dataset_names[0])
dataloader = torch.utils.data.DataLoader(dataset)
for id,data in enumerate(dataloader):
    pred_path=data[0][0]
    pred_semantic_path=data[1]
    gt_path=data[2][0]
    gt_semantic_path=data[3]
    if not os.path.exists(pred_path[0]) or not os.path.exists(gt_path):
        print(pred_path)



