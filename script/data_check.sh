# !/bin/bash

port_number=50009
bench_mark="Tuft"
obj_name='teeth_20240308'
trigger_word='teeth'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="13_pretrained_vae_anomal_normal_data_with_pe_self_aug"

python ../data_check.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/data_check_20240308_0.4_1" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --min_perlin_scale 1 \
 --max_perlin_scale 4 \
 --max_beta_scale 1 \
 --min_beta_scale 0.4