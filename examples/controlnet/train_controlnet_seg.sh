python train_controlnet.py \
 --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path "lllyasviel/control_v11p_sd15_seg"
 --output_dir="./out_proj_label" \
 --train_data_dir=E:\\data\\jax\\render\\dataset \
 --conditioning_image_column="condition" \
 --caption_column="text" \
 --lora_path="J:\\xuningli\\cross-view\\ground_view_generation\\outputs\\jax_7868_pano_lora\\checkpoint-70000" \
 --resolution=512 \
 --validation_image "/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset/proj_label/JAX_068 82_proj_label.png" "/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset/JAX_214 5_proj_label.png" "/research/GDA/xuningli/cross-view/ground_view_generation/dataset/JAX_171 71_proj_label.png" \
 --validation_prompt "street-view, panorama image, high resolution" \
 --train_batch_size=4 \
 --num_train_epochs=100 \
 --tracker_project_name="controlnet-seg" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=200 \
 --report_to wandb \