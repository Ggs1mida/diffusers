import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils
import glob
from skimage import io, color
from skimage.filters import roberts, sobel_h, sobel_v
from skimage.metrics import structural_similarity as ssim
from dataloader import Dataset
from os import listdir
from os.path import isfile, join
import re
import cv2
import shutil

def PSNR(true_frame, pred_frame):
    eps = 0.0001
    prediction_error = 0.0
    [h,w,c] = true_frame.shape
    dev_frame = pred_frame-true_frame
    dev_frame = np.multiply(dev_frame,dev_frame)
    prediction_error = np.sum(dev_frame)
    prediction_error = 128*128*prediction_error/(h*w*c)
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error

def SSIM(img1, img2):
    [h,w,c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:,:,0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:,:,0]
    score = ssim(img1, img2,data_range=1.0)
    return score

def L1difference(img_true,img_pred):
    [h,w] = img_true.shape
    true_gx = sobel_h(img_true)/4.0
    true_gy = sobel_v(img_true)/4.0
    pred_gx = sobel_h(img_pred)/4.0
    pred_gy = sobel_v(img_pred)/4.0
    dx = np.abs(true_gx-pred_gx)
    dy = np.abs(true_gy-pred_gy)
    prediction_error = np.sum(dx+dy)
    prediction_error=128*128*prediction_error/(h*w)
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def calculate_psnr_mask(pred_frame, true_frame, mask):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    eps = 0.0001
    prediction_error = 0.0
    [h,w,c] = true_frame.shape
    dev_frame = pred_frame-true_frame
    dev_frame = np.multiply(dev_frame,dev_frame)
    dev_frame[~mask]=np.zeros(3)
    prediction_error = np.sum(dev_frame)
    cnt=np.sum(mask!=0)*c
    prediction_error = 128*128*prediction_error/cnt
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error

def calculate_ssim_mask(img1, img2,mask):
    [h,w,c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:,:,0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:,:,0]
    max1=max(np.max(img1),np.max(img2))
    min1=min(np.min(img1),np.min(img2))
    mssim,map = ssim(img1, img2,full=True,data_range=max1-min1)
    a=(map*255).astype(np.uint8)
    cv2.imwrite(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\3.png',a)
    mean=np.mean(map)
    mask_score=np.mean(map[mask])

    return mask_score

def L1difference_mask(img_true,img_pred,mask):
    [h,w] = img_true.shape
    true_gx = sobel_h(img_true)/4.0
    true_gy = sobel_v(img_true)/4.0
    pred_gx = sobel_h(img_pred)/4.0
    pred_gy = sobel_v(img_pred)/4.0
    dx = np.abs(true_gx-pred_gx)
    dy = np.abs(true_gy-pred_gy)
    error = dx+dy
    prediction_error=np.sum(error[mask])
    cnt=np.sum(mask!=0)
    prediction_error=128*128*prediction_error/cnt
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error

def segment_gt(in_dir,out_dir,render_list):
    # Loading a single model for all three tasks
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        gt_path=os.path.join(in_dir,line.rstrip('\n')+"_street_rgb.jpg")
        image = Image.open(gt_path)
        semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        semantic_outputs = model(**semantic_inputs)
        # pass through image_processor for postprocessing
        predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0].numpy()
        cv2.imwrite(os.path.join(out_dir,line.rstrip('\n')+"_seg.png"),predicted_semantic_map)
        shutil.copyfile(gt_path,os.path.join(out_dir,line.rstrip('\n')+'.jpg'))

def segment_dir(in_dir,render_list):
    # Loading a single model for all three tasks
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for id,line in enumerate(lines):
        img_paths=glob.glob(os.path.join(in_dir,line.rstrip('\n')+"_sate_rgb_fake_B_final.png"))
        print(id)
        for img_path in img_paths:
            if 'seg.png' in img_path:
                continue
            if os.path.exists(img_path[:-4]+"_seg.png"):
                continue
            image = Image.open(img_path)
            semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
            semantic_outputs = model(**semantic_inputs)
            # pass through image_processor for postprocessing
            predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0].numpy()
            cv2.imwrite(img_path[:-4]+"_seg.png",predicted_semantic_map)

def eval_perpix(fold_pred,fold_gt,render_list):
    PSNR_arr=np.zeros((446,5))
    SSIM_arr=np.zeros((446,5))
    L1_arr=np.zeros((446,5))
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for id,line in enumerate(lines):
        gt_path=os.path.join(fold_gt,line.rstrip('\n')+"_street_rgb.jpg")
        pred_paths=glob.glob(os.path.join(fold_pred,line.rstrip('\n')+"*.png"))
        img_gt = cv2.imread(gt_path).astype(np.float32)/255.0 
        img_gt_gray = cv2.imread(gt_path,0).astype(np.float32)/255.0
        for i in range(5):
            pred_path=pred_paths[i]
            img_pred = cv2.imread(pred_path).astype(np.float32)/255.0
            img_pred_gray = cv2.imread(pred_path,0).astype(np.float32)/255.0
            PSNR_score = PSNR(img_gt, img_pred)
            SSIM_score = SSIM(img_gt, img_pred)
            L1_score = L1difference(img_gt_gray, img_pred_gray)
            PSNR_arr[id,i]=PSNR_score
            SSIM_arr[id,i]=SSIM_score
            L1_arr[id,i]=L1_score
    
    print("avg PSNR {}".format(np.mean(PSNR_arr)))
    print("avg SSIM {}".format(np.mean(SSIM_arr)))
    print("avg L1 score {}".format(np.mean(L1_arr)))
    np.savetxt(os.path.join(fold_pred,"psnr.txt"),PSNR_arr)
    np.savetxt(os.path.join(fold_pred,"ssim.txt"),SSIM_arr)
    np.savetxt(os.path.join(fold_pred,"l1.txt"),L1_arr)

def eval_building_perpix(fold_pred,fold_gt,render_list):
    PSNR_arr=np.zeros((446,5))
    SSIM_arr=np.zeros((446,5))
    L1_arr=np.zeros((446,5))
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for id,line in enumerate(lines):
        gt_path=os.path.join(fold_gt,line.rstrip('\n')+".jpg")
        gt_seg_path=os.path.join(fold_gt,line.rstrip('\n')+"_seg.png")
        pred_paths=glob.glob(os.path.join(fold_pred,line.rstrip('\n')+"*.png"))
        img_gt = cv2.imread(gt_path).astype(np.float32)/255.0
        img_gt_gray = cv2.imread(gt_path,0).astype(np.float32)/255.0
        seg_gt = cv2.imread(gt_seg_path)
        building_mask=seg_gt==1
        building_mask=building_mask[:,:,0]
        for i in range(5):
            pred_path=pred_paths[i]
            img_pred = cv2.imread(pred_path).astype(np.float32)/255.0
            img_pred_gray = cv2.imread(pred_path,0).astype(np.float32)/255.0
            PSNR_score = calculate_psnr_mask(img_gt, img_pred,building_mask)
            SSIM_score = calculate_ssim_mask(img_gt, img_pred,building_mask)
            L1_score = L1difference_mask(img_gt_gray, img_pred_gray, building_mask)
            PSNR_arr[id,i]=PSNR_score
            SSIM_arr[id,i]=SSIM_score
            L1_arr[id,i]=L1_score
    
    print("avg PSNR {}".format(np.mean(PSNR_arr)))
    print("avg SSIM {}".format(np.mean(SSIM_arr)))
    print("avg L1 score {}".format(np.mean(L1_arr)))
    np.savetxt(os.path.join(fold_pred,"psnr_building.txt"),PSNR_arr)
    np.savetxt(os.path.join(fold_pred,"ssim_building.txt"),SSIM_arr)
    np.savetxt(os.path.join(fold_pred,"l1_building.txt"),L1_arr)

def extract_into_same_dir(semantics_dir,proj_rgb_dir,proj_rgb_line_dir,gt_dir,out_dir,render_list):
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        imgs=glob.glob(os.path.join(semantics_dir,name+"*.png"))
        for img in imgs:
            if "seg.png" not in img:
                img_name=os.path.basename(img)
                shutil.copyfile(img,os.path.join(out_dir,img_name[:-4]+"_semantics.png"))

        imgs=glob.glob(os.path.join(proj_rgb_dir,name+"*.png"))
        for img in imgs:
            if "seg.png" not in img:
                img_name=os.path.basename(img)
                shutil.copyfile(img,os.path.join(out_dir,img_name[:-4]+"_proj_rgb.png"))

        imgs=glob.glob(os.path.join(proj_rgb_line_dir,name+"*.png"))
        for img in imgs:
            if "seg.png" not in img:
                img_name=os.path.basename(img)
                shutil.copyfile(img,os.path.join(out_dir,img_name[:-4]+"_proj_rgb_line.png"))

        imgs=glob.glob(os.path.join(gt_dir,name+"*.jpg"))
        for img in imgs:
            if "seg.png" not in img:
                img_name=os.path.basename(img)
                shutil.copyfile(img,os.path.join(out_dir,img_name[:-4]+"_street_rgb.jpg"))
    
    data_in.close()

def extract_into_same_dir_sota(semantics_dir,sate_dir,pano_dir,crossmlp_dir,out_dir,render_list):
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        img=os.path.join(semantics_dir,name+"_proj_label.png")
        shutil.copyfile(img,os.path.join(out_dir,name+"_proj_label.png"))

        img=os.path.join(sate_dir,name+"_sate_rgb.png")
        shutil.copyfile(img,os.path.join(out_dir,name+"_sate_rgb.png"))

        img=os.path.join(pano_dir,name+"_sate_rgb_fake_B_final.png")
        shutil.copyfile(img,os.path.join(out_dir,name+"_pano.png"))

        img=os.path.join(crossmlp_dir,name+"_sate_rgb_I.png")
        shutil.copyfile(img,os.path.join(out_dir,name+"_crossmlp.png"))
    
    data_in.close()


def eval_psnr_ssim_sharp():
    out_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_result'
    dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']

    for data_name in dataset_names:
        ssim_arr=np.zeros(446)
        psnr_arr=np.zeros(446)
        l1_arr=np.zeros(446)
        dataset=Dataset(data_name)
        dataloader = torch.utils.data.DataLoader(dataset) 
        for id,data in enumerate(dataloader):
            pred_path=data[0][0]
            pred_semantic_path=data[1]
            gt_path=data[2][0]
            gt_semantic_path=data[3]

            img_gt = cv2.imread(gt_path).astype(np.float32)/255.0
            img_gt_gray = cv2.imread(gt_path,0).astype(np.float32)/255.0
            img_pred = cv2.imread(pred_path).astype(np.float32)/255.0
            img_pred_gray = cv2.imread(pred_path,0).astype(np.float32)/255.0
            PSNR_score = PSNR(img_gt, img_pred)
            SSIM_score = SSIM(img_gt, img_pred)
            L1_score = L1difference(img_gt_gray, img_pred_gray)
            psnr_arr[id]=PSNR_score
            ssim_arr[id]=SSIM_score
            l1_arr[id]=L1_score
        np.savetxt(os.path.join(out_dir,"{}_psnr.txt").format(data_name),psnr_arr)
        np.savetxt(os.path.join(out_dir,"{}_ssim.txt").format(data_name),ssim_arr)
        np.savetxt(os.path.join(out_dir,"{}_l1sharp.txt").format(data_name),l1_arr)

        print("{} : PSNR: {}, SSIM: {}, L1:{} ".format(data_name,np.mean(psnr_arr),np.mean(ssim_arr),np.mean(l1_arr)))





if __name__ == "__main__":
    fold_pred = r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb'
    fold_gt = r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_gt'
    #render_list=r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_facade_pair.txt'
    render_list=r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_pair_test.txt'

    #get images    
    #segment_gt(r'E:\data\jax\render\dataset\street_rgb',r'J:\xuningli\cross-view\ground_view_generation\data\experiment\gt',r'E:\data\jax\render\dataset\rgb_seg_pair_train.txt')

    segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_semantics',render_list)
    segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_rgb',render_list)
    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_semantics',render_list)
    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_rgb',render_list)

    # extract_into_same_dir_sota(r'J:\xuningli\cross-view\ground_view_generation\data\dataset\proj_label',
    #     r'J:\xuningli\cross-view\ground_view_generation\data\dataset\sate_rgb',
    #                            r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_semantics',
    #                            r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_semantics',
    #                            r'J:\xuningli\cross-view\ground_view_generation\data\experiment\sota_select',
    #                            r'J:\xuningli\cross-view\ground_view_generation\data\experiment\sota_select\list.txt')
    # extract_into_same_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\sate_rgb_seg',
    # r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb',
    # r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb_line',
    # r'J:\xuningli\cross-view\ground_view_generation\data\dataset\street_rgb',
    # r'J:\xuningli\cross-view\ground_view_generation\data\experiment\abalation',
    # r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_facade_pair.txt')
    #eval_building_perpix(fold_pred,fold_gt,render_list)
    #eval_psnr_ssim_sharp()






    
    