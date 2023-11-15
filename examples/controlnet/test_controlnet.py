import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm


from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers import DiffusionPipeline
import cv2
#from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import StableDiffusionControlNetUnconditionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#from diffusers.models.controlnet import ControlNetModel_uncondition

ada_palette = np.asarray([
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

image_reservoir = []
latents_reservoir = []

def ade20k_color(img):
    img[img==34]=1
    img[img==104]=1
    img[img==21]=6
    out_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    out_img[img==0]=np.array([120,120,120])
    out_img[img==1]=np.array([180,120,120])
    out_img[img==2]=np.array([6,230,230])
    out_img[img==3]=np.array([80,50,50])
    out_img[img==4]=np.array([4,200,3])
    out_img[img==5]=np.array([120,120,80])
    out_img[img==6]=np.array([140,140,140])
    out_img[img==7]=np.array([204,5,255])
    out_img[img==8]=np.array([230,230,230])
    out_img[img==9]=np.array([4,250,7])
    out_img[img==10]=np.array([255,5,255])
    out_img[img==11]=np.array([235,255,7])
    out_img[img==12]=np.array([150,5,61])
    out_img[img==13]=np.array([120,120,70])
    out_img[img==17]=np.array([204,255,4])
    out_img[img==20]=np.array([0,102,200])
    out_img[img==21]=np.array([61,230,250])
    out_img[img==34]=np.array([255,41,10])
    out_img[img==87]=np.array([0,71,255])
    out_img[img==102]=np.array([163,255,0])
    return out_img

def seg(img_path):
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    url = img_path
    semantic_file=img_path[:-4]+'_semantic.png'
    semantic_color_file=img_path[:-4]+'_semantic_color.png'
    image = Image.open(url)

    # Loading a single model for all three tasks
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")

    # Semantic Segmentation
    semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    semantic_outputs = model(**semantic_inputs)
    # pass through image_processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
    cv2.imwrite(semantic_file,predicted_semantic_map.numpy())
    out_img=ade20k_color(predicted_semantic_map.numpy())
    #cv2.imwrite(semantic_color_file,out_img)
    return out_img

def depth(img_path):
    from annotator.util import resize_image, HWC3
    from annotator.midas import MidasDetector
    detect_reso=256
    img_reso=256
    apply_midas = MidasDetector()
    with torch.no_grad():
        input_image = HWC3(img_path)
        detected_map, _ = apply_midas(resize_image(input_image,detect_reso))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, img_reso)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return detected_map

def random_color(img):
    out_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    out_img[img==0]=np.array([120,120,120])
    out_img[img==1]=np.array([180,120,120])
    out_img[img==2]=np.array([6,230,230])
    out_img[img==3]=np.array([80,50,50])
    out_img[img==4]=np.array([4,200,3])
    out_img[img==5]=np.array([120,120,80])
    out_img[img==6]=np.array([140,140,140])
    out_img[img==7]=np.array([204,5,255])
    out_img[img==8]=np.array([230,230,230])
    out_img[img==9]=np.array([4,250,7])
    out_img[img==10]=np.array([255,5,255])
    out_img=np.array([out_img[:,:,2],out_img[:,:,1],out_img[:,:,0]])
    out_img=np.moveaxis(out_img,[0,1,2],[2,0,1])
    return out_img

def render_controlnet_canny_color_batch(in_dir,render_list,out_dir,controlnet_path,lora_path,prompt,negative_prompt):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    if lora_path:
        pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_img_path=os.path.join(in_dir,"proj_rgb",name+"_proj_rgb.png")
        condition_image = load_image(condition_img_path)
        generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(5)]
        images = pipe(
            prompt=[prompt]*5,
            image=condition_image,
            negative_prompt=[negative_prompt]*5,
            num_inference_steps=20,
            generator=generator
        ).images
        for id,image in enumerate(images):
            image.save(os.path.join(out_dir,"{}_{}.png".format(name,id)))

def render_controlnet_canny_color_batchsingle(in_dir,render_list,out_dir,controlnet_path,lora_path,prompt,negative_prompt):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    if lora_path:
        pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_img_path=os.path.join(in_dir,"proj_rgb",name+"_proj_rgb.png")
        condition_image = load_image(condition_img_path)
        generator = [torch.Generator(device="cpu").manual_seed(0) ]
        image = pipe(
            prompt=prompt,
            image=condition_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        image.save(os.path.join(out_dir,"{}.png".format(name)))

def render_controlnet_lines_batch(in_dir,render_list,out_dir,controlnet_path,lora_path,prompt,negative_prompt):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_img_path=os.path.join(in_dir,"experiment","train","facade_lines",name+"_lines.png")
        condition_image = load_image(condition_img_path)
        generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(5)]
        images = pipe(
            prompt=[prompt]*5,
            image=condition_image,
            negative_prompt=[negative_prompt]*5,
            num_inference_steps=20,
            generator=generator
        ).images
        for id,image in enumerate(images):
            image.save(os.path.join(out_dir,"{}_{}.png".format(name,id)))

def render_controlnet_sate_rgbseg_batch(in_dir,render_list,out_dir,controlnet_seg_path,controlnet_sate_path,lora_path,prompt,negative_prompt):
    controlnet_sate = ControlNetModel.from_pretrained(controlnet_sate_path, torch_dtype=torch.float16)
    controlnet_seg = ControlNetModel.from_pretrained(controlnet_seg_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_sate,controlnet_seg], torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_seg_path=os.path.join(in_dir,"proj_label",name+"_proj_label.png")
        condition_sate_path=os.path.join(in_dir,"sate_rgb_2column",name+"_sate_rgb.png")
        condition_seg = load_image(condition_seg_path)
        condition_sate = load_image(condition_sate_path)
        conditions=[condition_sate,condition_seg]
        generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(5)]
        images = pipe(
            [prompt]*5,
            conditions,
            negative_prompt=[negative_prompt]*5,
            num_inference_steps=20,
            generator=generator,
            controlnet_conditioning_scale=[1.0,1.0]
        ).images
        for id,image in enumerate(images):
            image.save(os.path.join(out_dir,"{}_{}.png".format(name,id)))

def render_controlnet_sate_rgbline_batch(in_dir,render_list,out_dir,controlnet_rgb_path,controlnet_line_path,lora_path,prompt,negative_prompt):
    controlnet_line = ControlNetModel.from_pretrained(controlnet_line_path, torch_dtype=torch.float16)
    controlnet_rgb = ControlNetModel.from_pretrained(controlnet_rgb_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_line,controlnet_rgb], torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_line_path=os.path.join(in_dir,"experiment","train","facade_lines",name+"_lines.png")
        condition_rgb_path=os.path.join(in_dir,"dataset","proj_rgb",name+"_proj_rgb.png")
        condition_line = load_image(condition_line_path)
        condition_rgb = load_image(condition_rgb_path)
        conditions=[condition_line,condition_rgb]
        generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(5)]
        images = pipe(
            [prompt]*5,
            conditions,
            negative_prompt=[negative_prompt]*5,
            num_inference_steps=20,
            generator=generator,
            controlnet_conditioning_scale=[1,1]
        ).images
        for id,image in enumerate(images):
            image.save(os.path.join(out_dir,"{}_{}.png".format(name,id)))

def render_controlnet_sate_rgb_batch(in_dir,render_list,out_dir,controlnet_sate_path,lora_path,prompt,negative_prompt):
    controlnet_sate = ControlNetModel.from_pretrained(controlnet_sate_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_sate, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    import os
    data_in=open(render_list,'r')
    lines=data_in.readlines()
    for line in lines:
        name=line.rstrip('\n')
        condition_sate_path=os.path.join(in_dir,"sate_rgb_2column",name+"_sate_rgb.png")
        condition_sate = load_image(condition_sate_path)
        condition_sate.save(r'J:\xuningli\cross-view\ground_view_generation\data\experiment\12.png')
        generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(5)]
        images = pipe(
            [prompt]*5,
            condition_sate,
            negative_prompt=[negative_prompt]*5,
            num_inference_steps=20,
            generator=generator
        ).images
        for id,image in enumerate(images):
            image.save(os.path.join(out_dir,"{}_{}.png".format(name,id)))


def main():
    prompt="street-view, panorama image"
    negetive_prompt="watermark, blury, artifacts, glare"
    #lora_path=r'J:\xuningli\cross-view\ground_view_generation\outputs\jax_7868_pano_lora\checkpoint-70000'
    lora_path=r'J:\xuningli\cross-view\ground_view_generation\code\outputs\jax_3535\checkpoint-55000'
    #lora_path=""
    #lora_path=r'J:\xuningli\cross-view\ground_view_generation\outputs\four_city\checkpoint-90000'
    dataset_dir_par=r'J:\xuningli\cross-view\ground_view_generation\data'
    dataset_dir=r'J:\xuningli\cross-view\ground_view_generation\data\dataset'
    test_list=r'J:\xuningli\cross-view\ground_view_generation\data\dataset\rgb_seg_pair_test.txt'
    #controlnet_color=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_proj_rgb\checkpoint-60000\controlnet'
    controlnet_color=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_proj_rgb_3535_continue\checkpoint-80000\controlnet'
    controlnet_color_nolora=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_proj_rgb_3535_nolora\checkpoint-40000\controlnet'
    controlnet_sate=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_sate_rgb\checkpoint-60000\controlnet'
    controlnet_seg=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_proj_label\checkpoint-65000\controlnet'
    #controlnet_line=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_facade_lines\checkpoint-60000\controlnet'
    controlnet_line=r'J:\xuningli\cross-view\ground_view_generation\code\diffusers\examples\controlnet\out_proj_rgb_3535_facade\checkpoint-65000\controlnet'
    
    
    #render_controlnet_canny_color_batch(dataset_dir,test_list,r'E:\tmp\ours_color_nolora',controlnet_color,lora_path,prompt,negetive_prompt)

    render_controlnet_canny_color_batchsingle(dataset_dir,test_list,r'E:\tmp\ours_nolora',controlnet_color_nolora,"",prompt,negetive_prompt)

    #render_controlnet_lines_batch(dataset_dir_par,test_list,r'E:\tmp\ours_lines',controlnet_line,lora_path,prompt,negetive_prompt)

    #render_controlnet_sate_rgbseg_batch(dataset_dir,test_list,r'J:\xuningli\cross-view\ground_view_generation\data\experiment\sate_rgb_seg',controlnet_seg,controlnet_sate,lora_path,prompt,negetive_prompt)

    #render_controlnet_sate_rgbline_batch(dataset_dir_par,test_list,r'E:\tmp\ours_color_lines',controlnet_color,controlnet_line,lora_path,prompt,negetive_prompt)

    #render_controlnet_sate_rgb_batch(dataset_dir,test_list,r'J:\xuningli\cross-view\ground_view_generation\data\experiment\ours_sate_rgb1',controlnet_sate,lora_path,prompt,negetive_prompt)






if __name__ == "__main__":
    main()
