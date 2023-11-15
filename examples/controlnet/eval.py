import numpy as np
import torch
from PIL import Image
import glob
from skimage import  color
from skimage.filters import sobel_h, sobel_v
from skimage.metrics import structural_similarity as ssim
from dataloader import Dataset
import cv2
import shutil

def PSNR(true_frame, pred_frame):
    true_frame=true_frame.astype(np.float32)/255.0
    pred_frame=pred_frame.astype(np.float32)/255.0
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
    img1=img1.astype(np.float32)/255.0
    img2=img2.astype(np.float32)/255.0
    [h,w,c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:,:,0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:,:,0]
    else:
        img1 = img1[:,:,0]
        img2 = img2[:,:,0]
    score = ssim(img1, img2,data_range=1.0)
    return score

def miou(img1,img2):
    mask1=img1!=0
    mask2=img2!=0
    intersect=np.sum(np.logical_and(mask1,mask2))
    union=np.sum(np.logical_or(mask1,mask2))
    return intersect/union,intersect,union

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
    import os
    # Loading a single model for all three tasks
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for id,line in enumerate(lines):
        img_paths=glob.glob(os.path.join(in_dir,line.rstrip('\n')+".png"))
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

def extract_into_same_dir_controlnet(in_dir,out_dir,render_list,psnr_txt):
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    psnr_arr=np.loadtxt(psnr_txt)
    id_arr=np.argmax(psnr_arr,axis=1)
    for id,line in enumerate(lines):
        name=line.rstrip('\n')
        idx=id_arr[id]
        name1=name+'_{}'.format(idx)
        shutil.copyfile(os.path.join(in_dir,"{}.png".format(name1)),os.path.join(out_dir,"{}.png".format(name)))



def eval_psnr_ssim_sharp():
    import os
    out_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_result'
    #dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
    #dataset_names=['pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
    dataset_names=['ours_color_lines_new']

    for data_name in dataset_names:
        ssim_list=[]
        psnr_list=[]
        ssim_canny_list=[]
        psnr_canny_list=[]
        dataset=Dataset(data_name)
        dataloader = torch.utils.data.DataLoader(dataset) 
        for id,data in enumerate(dataloader):
            pred_paths=data[0]
            gt_path=data[2][0]

            img_gt = cv2.imread(gt_path)
            gt_edge = np.expand_dims(cv2.Canny(img_gt,255,255),axis=2)
            p1_list=[]
            s1_list=[]
            p2_list=[]
            s2_list=[]
            for pred_path in pred_paths:
                pred_path=pred_path[0]
                img_pred = cv2.imread(pred_path)
                pred_edge = np.expand_dims(cv2.Canny(img_pred,255,255),axis=2)

                p1 = PSNR(img_gt, img_pred)
                s1 = SSIM(img_gt, img_pred)
                
                p2 = PSNR(gt_edge, pred_edge)
                s2 = SSIM(gt_edge, pred_edge)
                p1_list.append(p1)
                p2_list.append(p2)
                s1_list.append(s1)
                s2_list.append(s2)
            ssim_list.append(s1_list)
            psnr_list.append(p1_list)
            ssim_canny_list.append(s2_list)
            psnr_canny_list.append(p2_list)

        ssim_arr=np.array(ssim_list)
        ssim_canny_arr=np.array(ssim_canny_list)
        psnr_arr=np.array(psnr_list)
        psnr_canny_arr=np.array(psnr_canny_list)

        np.savetxt(os.path.join(out_dir,"{}_psnr.txt").format(data_name),psnr_arr)
        np.savetxt(os.path.join(out_dir,"{}_ssim.txt").format(data_name),ssim_arr)

        # np.savetxt(os.path.join(out_dir,"{}_canny_psnr.txt").format(data_name),psnr_canny_arr)
        # np.savetxt(os.path.join(out_dir,"{}_canny_ssim.txt").format(data_name),ssim_canny_arr)

        print("{} : PSNR: {}, SSIM: {}".format(data_name,np.mean(psnr_arr),np.mean(ssim_arr)))

def eval_layout():
    import os
    out_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_result'
    dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
    #dataset_names=['pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']

    for data_name in dataset_names:
        precision_list=[]
        recall_list=[]
        f1score_list=[]
        iou_list=[]
        dataset=Dataset(data_name)
        dataloader = torch.utils.data.DataLoader(dataset) 
        for id,data in enumerate(dataloader):
            pred_paths=data[0]
            gt_path=data[2][0]

            img_gt = cv2.imread(gt_path)
            gt_edge = cv2.Canny(img_gt,255,255)
            ps=[]
            rs=[]
            fs=[]
            ious=[]
            for pred_path in pred_paths:
                pred_path=pred_path[0]
                img_pred = cv2.imread(pred_path)
                pred_edge = cv2.Canny(img_pred,255,255)

                mask_pred=pred_edge!=0
                mask_gt=gt_edge!=0
                intersect=np.sum(np.logical_and(mask_pred,mask_gt))
                union=np.sum(np.logical_or(mask_pred,mask_gt))
                iou=intersect/union
                precision=intersect/np.sum(mask_pred)
                recall=intersect/np.sum(mask_gt)
                f1=2*(precision*recall)/(precision+recall)

                ious.append(iou)
                ps.append(precision)
                rs.append(recall)
                fs.append(f1)

            precision_list.append(ps)
            recall_list.append(rs)
            f1score_list.append(fs)
            iou_list.append(ious)
        

        precision_arr=np.array(precision_list)
        recall_arr=np.array(recall_list)
        fscore_arr=np.array(f1score_list)
        ious_arr=np.array(iou_list)

        np.savetxt(os.path.join(out_dir,"{}_precision.txt").format(data_name),precision_arr)
        np.savetxt(os.path.join(out_dir,"{}_recall.txt").format(data_name),recall_arr)
        np.savetxt(os.path.join(out_dir,"{}_fscore.txt").format(data_name),fscore_arr)
        np.savetxt(os.path.join(out_dir,"{}_ious.txt").format(data_name),ious_arr)

        print("{} : Precision: {}, Recall: {}, F1score: {}, IOU: {}".format(data_name,np.mean(precision_arr),np.mean(recall_arr),np.mean(fscore_arr),np.mean(ious_arr)))

def eval_semantic():
    import os
    out_dir=r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_result'
    dataset_names=['ours_color_lines','ours_color','ours_satergb_seg','pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
    #dataset_names=['pano_rgb','pano_semantic','crossmlp_rgb','crossmlp_semantic']
    building_id=1
    road_id=6
    ground_id=13
    side_id=11
    tree_id=4
    sky_id=2

    for data_name in dataset_names:
        score_list=[]
        dataset=Dataset(data_name)
        dataloader = torch.utils.data.DataLoader(dataset) 
        b_list=[]
        s_list=[]
        g_list=[]
        r_list=[]
        side_list=[]
        t_list=[]
        all_list=[]
        for id,data in enumerate(dataloader):
            pred_paths=data[0]
            seg_paths=data[1]
            gt_seg_path=data[3][0]
            gt_path=data[2][0]

            img_gt = cv2.imread(gt_seg_path)
            bs=[]
            gs=[]
            ss=[]
            rs=[]
            sides=[]
            ts=[]
            alls=[]

            for seg_path in seg_paths:
                img_pred = cv2.imread(seg_path[0])
                score=np.mean(img_pred==img_gt)
                alls.append(score)

                pred_mask=img_pred==building_id
                gt_mask=img_gt==building_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                bs.append(np.sum(inter)/np.sum(union))

                pred_mask=img_pred==ground_id
                gt_mask=img_gt==ground_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                gs.append(np.sum(inter)/np.sum(union))

                pred_mask=img_pred==road_id
                gt_mask=img_gt==road_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                rs.append(np.sum(inter)/np.sum(union))

                pred_mask=img_pred==side_id
                gt_mask=img_gt==side_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                sides.append(np.sum(inter)/np.sum(union))

                pred_mask=img_pred==sky_id
                gt_mask=img_gt==sky_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                ss.append(np.sum(inter)/np.sum(union))

                pred_mask=img_pred==tree_id
                gt_mask=img_gt==tree_id
                inter=np.logical_and(pred_mask,gt_mask)
                union=np.logical_or(pred_mask,gt_mask)
                ts.append(np.sum(inter)/np.sum(union))



            b_list.append(bs)
            s_list.append(ss)
            g_list.append(gs)
            side_list.append(sides)
            t_list.append(ts)
            r_list.append(rs)
            all_list.append(alls)

        all_arr=np.array(all_list)
        b_arr=np.array(b_list)
        s_arr=np.array(s_list)
        g_arr=np.array(g_list)
        side_arr=np.array(side_list)
        t_arr=np.array(t_list)
        r_arr=np.array(r_list)


        np.savetxt(os.path.join(out_dir,"{}_semantic_all.txt").format(data_name),all_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_build.txt").format(data_name),b_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_sky.txt").format(data_name),s_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_ground.txt").format(data_name),g_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_road.txt").format(data_name),r_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_tree.txt").format(data_name),t_arr)
        np.savetxt(os.path.join(out_dir,"{}_semantic_sidewalk.txt").format(data_name),side_arr)


        print("{} : semantic all: {}, build: {}, sky:{}, ground:{}".format(data_name,np.nanmean(all_arr),np.nanmean(b_arr),np.nanmean(s_arr),np.nanmean(g_arr)))

def eval_iou():
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

    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_semantics',render_list)
    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\pano_rgb',render_list)
    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_semantics',render_list)
    # segment_dir(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\crossmlp_rgb',render_list)
    #segment_dir(r'E:\tmp\ours_color',render_list)

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
    extract_into_same_dir_controlnet(r'E:\tmp\ours_color_lines2',r'E:\tmp\ours_color_lines',render_list,r'J:\xuningli\cross-view\ground_view_generation\data\experiment\eval_result\ours_color_lines_new_psnr.txt')
    #eval_building_perpix(fold_pred,fold_gt,render_list)
    #eval_psnr_ssim_sharp()
    #eval_semantic()
    #eval_layout()
    # img1=r'E:\tmp\ours_canny.png'
    # img2=r'E:\tmp\panogan_canny.png'
    # img_gt=r'E:\tmp\gt_canny.png'

    # ours = cv2.imread(img1)
    # panogan = cv2.imread(img2)
    # gt = cv2.imread(img_gt)
    # s1=SSIM(ours,gt)
    # s2=SSIM(panogan,gt)
    # p1=PSNR(ours,gt)
    # p2=PSNR(panogan,gt)
    # iou1,i1,u1=miou(ours,gt)
    # iou2,i2,u2=miou(panogan,gt)

    # a=1







    
    