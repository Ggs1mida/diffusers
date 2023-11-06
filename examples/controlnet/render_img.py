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

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def render_controlnet_canny():
    #writer=SummaryWriter()
    image = load_image(
        r"J:\xuningli\cross-view\stablediffusion\code\test\466.png"
    )
    #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    # prompt = ", best quality, extremely detailed"
    # prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    #prompt = [t + prompt for t in ["blackman", "whiteman", "asian", "spanish"]]
    #generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            canny_image,
            num_inference_steps=20,
            generator=generator,
            guidance_scale=3
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(r'J:\xuningli\cross-view\stablediffusion\code\test\predict_canny.png')
    canny_image.save(r'J:\xuningli\cross-view\stablediffusion\code\test\canny.png')
    # img1=torch.tensor(np.asarray(output.images[0])).permute(2,0,1).unsqueeze(0)
    # img2=torch.tensor(np.asarray(output.images[1])).permute(2,0,1).unsqueeze(0)
    # img3=torch.tensor(np.asarray(output.images[2])).permute(2,0,1).unsqueeze(0)
    # img4=torch.tensor(np.asarray(output.images[3])).permute(2,0,1).unsqueeze(0)
    # imgs=torch.cat([img1,img2,img3,img4],dim=0)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_canny_color(in_img,out_img,lora_path,prompt,negative_prompt):
    #writer=SummaryWriter()
    image = load_image(in_img
    )
    #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(image, low_threshold, high_threshold)
    #canny_image = canny_image[:, :, None]
    #canny_image  = np.concatenate([canny_image , canny_image , canny_image ], axis=2)

    #kernel=np.ones((20,20),np.float32)/400
    #blury_image=cv2.filter2D(image,-1,kernel)
    #blury_image[canny_image==255,:]=255
    processed_image=image
    processed_image = Image.fromarray(processed_image)
    processed_image.save(out_img[:-4]+"_cannycolor.jpg")
    controlnet = ControlNetModel.from_pretrained("ghoskno/Color-Canny-Controlnet-model", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    num_img=8

    # prompt = "street-view, panorama image, high resolution"
    # num_img=8
    # images=[]
    # for i in range(num_img):
    #     generator = [torch.Generator(device="cpu").manual_seed(i)]
    #     image = pipe(
    #         prompt,
    #         processed_image,
    #         num_inference_steps=20,
    #         generator=generator,
    #         guidance_scale=3
    #     ).images[0]
    #     images.append(image)
    # image_grid = make_grid(images, rows=4, cols=2)
    # image_grid.save(out_img)

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=processed_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img[:-4]+"_cannycolor_nolora.jpg")

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=processed_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img[:-4]+"_cannycolor_lora.jpg")

def render_controlnet_seg():
    #writer=SummaryWriter()

    # image = cv2.imread(r'J:\xuningli\cross-view\stablediffusion\code\test\semantic.jpg',cv2.IMREAD_GRAYSCALE)
    # #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    # image = np.array(image)
    # #image=ade20k_color(image)
    # image=random_color(image)
    image=seg(r'J:\xuningli\cross-view\stablediffusion\code\test\466.png')
    seg_image = Image.fromarray(image)
    #seg_image=image

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            seg_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(r'J:\xuningli\cross-view\stablediffusion\code\test\ade20k_seg_predict.png')
    seg_image.save(r'J:\xuningli\cross-view\stablediffusion\code\test\ade20k_seg_color.png')
    #cv2.imwrite(r'J:\xuningli\cross-view\stablediffusion\code\test\seg.png',seg_image)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_depth():
    #writer=SummaryWriter()
    img_path=r'J:\xuningli\cross-view\stablediffusion\code\test\466.png'
    image = cv2.imread(img_path)
    # #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    # image = np.array(image)
    # #image=ade20k_color(image)
    # image=random_color(image)
    depth_image=depth(image)
    depth_image = Image.fromarray(depth_image)
    depth_image.save(img_path[:-4]+'_depth.png')
    #seg_image=image

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            depth_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(img_path[:-4]+'_depth_predict.png')

def render_controlnet_lora_canny_origin():
    #writer=SummaryWriter()
    image = load_image(
        r"J:\xuningli\cross-view\stablediffusion\code\test\466.png"
    )
    #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(r"C:\code\diffusers\examples\controlnet\london_dreambothlora\london_finetuning\checkpoint-50000")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            canny_image,
            num_inference_steps=20,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(r'J:\xuningli\cross-view\stablediffusion\code\test\predict_canny_nofinetune.png')

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            canny_image,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(r'J:\xuningli\cross-view\stablediffusion\code\test\predict_canny_finetune.png')
    # img1=torch.tensor(np.asarray(output.images[0])).permute(2,0,1).unsqueeze(0)
    # img2=torch.tensor(np.asarray(output.images[1])).permute(2,0,1).unsqueeze(0)
    # img3=torch.tensor(np.asarray(output.images[2])).permute(2,0,1).unsqueeze(0)
    # img4=torch.tensor(np.asarray(output.images[3])).permute(2,0,1).unsqueeze(0)
    # imgs=torch.cat([img1,img2,img3,img4],dim=0)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_lora_depth_origin():
    img_path=r'J:\xuningli\cross-view\stablediffusion\code\test\baker_sat\baker_sat.png'
    image=cv2.imread(img_path)
    image=depth(image)
    depth_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(r"C:\code\diffusers\examples\controlnet\london_dreambothlora\london_finetuning\checkpoint-50000")
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            depth_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(img_path[:-4]+'_predict_depth_nofinetune.png')
    depth_image.save(img_path[:-4]+'_depth.png')
    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt,
            depth_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(img_path[:-4]+'_predict_depth_finetune.png')
    #cv2.imwrite(r'J:\xuningli\cross-view\stablediffusion\code\test\seg.png',seg_image)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_uncondition():
    writer=SummaryWriter()
    image = load_image(
        r'E:\data\sd\controlnet_london2155_blury\condition\299.png'
    )
    #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    # image = np.array(image)

    # low_threshold = 100
    # high_threshold = 200

    # image = cv2.Canny(image, low_threshold, high_threshold)
    # image = image[:, :, None]
    # image = np.concatenate([image, image, image], axis=2)
    # canny_image = Image.fromarray(image)

    controlnet = ControlNetModel_uncondition.from_pretrained(r'C:\code\diffusers\examples\controlnet\model_out\checkpoint-60000\controlnet')
    unet_pipeline=DiffusionPipeline.from_pretrained(r'J:\xuningli\cross-view\diffusers\out\ldm_london_2155')
    pipe=StableDiffusionControlNetUnconditionPipeline(unet_pipeline.vqvae,unet_pipeline.unet,controlnet=controlnet,scheduler=unet_pipeline.scheduler)
    # pipe = StableDiffusionControlNetUnconditionPipeline.from_pretrained(
    #     r'J:\xuningli\cross-view\diffusers\out\ldm_london_2155', controlnet=controlnet, torch_dtype=torch.float16
    # )
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    #prompt = [t + prompt for t in ["blackman", "whiteman", "asian", "spanish"]]
    generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(4)]

    # @torch.no_grad()
    # def plot_show_callback(i, t, latents):
    #     writer.add_image("latent",latents[0,:3,:,:],global_step=i)

    output = pipe(
        image,
        num_inference_steps=20,
        generator=generator,
        guess_mode=True,
        guidance_scale=3,
        controlnet_conditioning_scale=1.0
    )

    img1=torch.tensor(np.asarray(output.images[0])).permute(2,0,1).unsqueeze(0)
    img2=torch.tensor(np.asarray(output.images[1])).permute(2,0,1).unsqueeze(0)
    img3=torch.tensor(np.asarray(output.images[2])).permute(2,0,1).unsqueeze(0)
    img4=torch.tensor(np.asarray(output.images[3])).permute(2,0,1).unsqueeze(0)
    imgs=torch.cat([img1,img2,img3,img4],dim=0)
    writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_lora(prompt,negative_prompt,lora_path,out_img_path):
    num_img=8
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=256,
            num_inference_steps=20,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_nolora.png')

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=256,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_lora.png')

def render_controlnet_lora_seg_origin(img_path,prompt,negative_prompt,lora_path,out_img_path):

    image=seg(img_path)
    seg_image = Image.fromarray(image)
    seg_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(r"C:\code\diffusers\examples\controlnet\london_dreambothlora\london_finetuning\checkpoint-50000")
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = "street-view, panorama image"
    num_img=8
    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt,
            seg_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(img_path[:-4]+'_predict_seg_nofinetune.png')
    seg_image.save(img_path[:-4]+'_seg_color.png')
    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt,
            seg_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(img_path[:-4]+'_predict_seg_finetune.png')
    #cv2.imwrite(r'J:\xuningli\cross-view\stablediffusion\code\test\seg.png',seg_image)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_lora_canny(img_path,prompt,negative_prompt,lora_path,out_img_path):
    num_img=8
    #writer=SummaryWriter()
    image = load_image(img_path)
    #image=load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save(out_img_path[:-4]+"_canny.jpg")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=canny_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+"_canny_nolora.jpg")

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt,
            canny_image,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+"_canny_lora.jpg")
    # img1=torch.tensor(np.asarray(output.images[0])).permute(2,0,1).unsqueeze(0)
    # img2=torch.tensor(np.asarray(output.images[1])).permute(2,0,1).unsqueeze(0)
    # img3=torch.tensor(np.asarray(output.images[2])).permute(2,0,1).unsqueeze(0)
    # img4=torch.tensor(np.asarray(output.images[3])).permute(2,0,1).unsqueeze(0)
    # imgs=torch.cat([img1,img2,img3,img4],dim=0)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_lora_seg(img_path,prompt,negative_prompt,lora_path,out_img_path):
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    #from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    num_img=8
    #image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    #image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    image_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    image_segmentor = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")

    image=load_image(img_path)
    pixel_values = image_processor(image, task_inputs=["semantic"],return_tensors="pt")
    with torch.no_grad():
        outputs = image_segmentor(**pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    control_image.save(out_img_path[:-4]+"_segmentation.png")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.unet.load_attn_procs(lora_path)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    images=[]
    for i in range(num_img):
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            cross_attention_kwargs={"scale": 0}
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_segmentation_nolora.png')
    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_segmentation_lora.png')

def render_controlnet_lora_depth(img_path,depth_or_not,prompt,negative_prompt,lora_path,out_img_path):
    from transformers import pipeline
    num_img=8
    if depth_or_not:
        depth=cv2.imread(img_path,0)
        sky_mask=depth==0
        depth=255-depth
        depth[sky_mask]=0
        depth=depth[:,:,None]
        image = np.concatenate([depth, depth, depth], axis=2)
        control_image = Image.fromarray(image)
    else:
        image=load_image(img_path)
        depth_estimator = pipeline('depth-estimation')
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
    control_image.save(out_img_path[:-4]+'_depth.png')
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_depth_nolora.png')

    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_depth_lora.png')
    #cv2.imwrite(r'J:\xuningli\cross-view\stablediffusion\code\test\seg.png',seg_image)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def render_controlnet_lora_lineart(img_path,prompt,negative_prompt,lora_path,out_img_path):
    from transformers import pipeline
    from controlnet_aux import LineartDetector
    num_img=8

    image=load_image(img_path)
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    control_image.save(out_img_path[:-4]+'_lineart.png')
    controlnet = ControlNetModel.from_pretrained("ControlNet-1-1-preview/control_v11p_sd15_lineart", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_path)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]
        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            cross_attention_kwargs={"scale": 0},
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_lineart_nolora.png')

    images=[]
    for i in range(num_img):
        #prompt = [prompt,prompt]
        generator = [torch.Generator(device="cpu").manual_seed(i)]

        image = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(image)
    image_grid = make_grid(images, rows=4, cols=2)
    image_grid.save(out_img_path[:-4]+'_lineart_lora.png')
    #cv2.imwrite(r'J:\xuningli\cross-view\stablediffusion\code\test\seg.png',seg_image)
    #writer.add_images('output',imgs,dataformats='NCHW')
    #image_grid(output.images, 2, 2)

def main():
    prompt="street-view, panorama image, high resolution"
    negetive_prompt="watermark, blury, artifacts, glare "
    lora_path=r'J:\xuningli\cross-view\ground_view_generation\outputs\jax_7868_pano_lora\checkpoint-70000'
    #lora_path=r'J:\xuningli\cross-view\ground_view_generation\outputs\four_city\checkpoint-90000'
    
    # render based on only lora model
    # render_lora("street-view, panorama image, high resolution, hong kong",
    #             negetive_prompt,
    #             lora_path,
    #             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\fourcity\hongkong.png')


    #render based on controlnet-segmentation
    # render_controlnet_lora_canny(r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_628.jpg',
    #                             prompt,
    #                             negetive_prompt,
    #                             lora_path,
    #                             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_628_out.jpg'
    #                            )

    #render based on controlnet-segmentation
    # render_controlnet_lora_seg(r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_628.jpg',
    #                             prompt,
    #                             negetive_prompt,
    #                             lora_path,
    #                             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_628_out.jpg'
    #                            )

    #render based on controlnet-depth
    # render_controlnet_lora_depth(r'E:\data\jax\render\251\google_street\JAX_251 98.depthmap.jpg',
    #                             True,
    #                             "street-view, panorama image, high resolution",
    #                             negetive_prompt,
    #                             lora_path,
    #                             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_251_98_depth.jpg'
    #                            )

    #render color & canny
    # render_controlnet_canny_color(r'E:\data\jax\render\251\sat\JAX_251 98.png',
    #                             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\JAX_251_98_out.jpg',
    #                             lora_path,
    #                             prompt,
    #                             negetive_prompt)

    #render lineart
    # render_controlnet_lora_lineart(r'J:\xuningli\cross-view\ground_view_generation\data\tmp\sat_8170.png',
    #                             prompt,
    #                             negetive_prompt,
    #                             lora_path,
    #                             r'J:\xuningli\cross-view\ground_view_generation\data\tmp\sat_8170_out_lineart.jpg')
    #render_controlnet_canny()
    #render_controlnet_canny_color()
    #render_controlnet_seg()
    #render_controlnet_depth()
    #render_controlnet_lora_canny()
    #render_controlnet_lora_seg()
    #render_controlnet_lora_depth()
    #render_controlnet_uncondition()







if __name__ == "__main__":
    main()
