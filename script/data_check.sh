# !/bin/bash

port_number=50011
category="medical"
obj_name="brain"
benchmark="NFBS"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="4_unsupervised"

python ../data_check.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/data_check_20240309_0.5_0.9" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --anomal_source_path "../../../MyData/anomal_source_l_mode" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --min_perlin_scale 1 \
 --max_perlin_scale 4 \
 --max_beta_scale 0.9 \
 --min_beta_scale 0.5