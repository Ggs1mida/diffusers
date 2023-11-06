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
    for line in lines:
        img_paths=glob.glob(os.path.join(in_dir,line.rstrip('\n')+"_*.png"))
        for img_path in img_paths:
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

if __name__ == "__main__":
    fold_pred = r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb'
    fold_gt = r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_gt'
    render_list=r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_pair_test.txt'

    #get images    
    #segment_gt(r'E:\data\jax\render\dataset\street_rgb',r'J:\xuningli\cross-view\ground_view_generation\data\experiment\gt',r'E:\data\jax\render\dataset\rgb_seg_pair_train.txt')

    segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_proj_rgb_line',render_list)

    #eval_building_perpix(fold_pred,fold_gt,render_list)






    
    