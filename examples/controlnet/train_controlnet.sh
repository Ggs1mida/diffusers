python train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="./out_proj_rgb_3535_fromscratch" \
 --train_data_dir='/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset' \
 --conditioning_image_column="condition" \
 --caption_column="text" \
 --lora_path='/research/GDA/xuningli/cross-view/ground_view_generation/code/outputs/jax_3535/checkpoint-55000' \
 --resolution=512 \
 --validation_image "/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset/proj_rgb/JAX_068 82_proj_rgb.png" "/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset/proj_rgb/JAX_214 5_proj_rgb.png" "/research/GDA/xuningli/cross-view/ground_view_generation/data/dataset/proj_rgb/JAX_171 71_proj_rgb.png" \
 --validation_prompt="street-view, panorama image" \
 --train_batch_size=2 \
 --num_train_epochs=100 \
 --tracker_project_name="controlnet-color" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=1000 \
 --report_to wandb