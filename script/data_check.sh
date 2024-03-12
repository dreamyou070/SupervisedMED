# !/bin/bash

port_number=50012
category="medical"
obj_name="brain"
benchmark="NFBS"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="4_unsupervised_simplex_noise_trg_beta_0"
#--do_self_aug

python ../data_check.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/data_check_anomal_simplex_noise_trg_beta_0.0_0.3" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${category}/${obj_name}/${benchmark}" \
 --anomal_source_path "../../../MyData/noise_source" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --max_beta_scale 0.3 \
 --min_beta_scale 0.0 --unsupervised