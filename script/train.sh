# !/bin/bash
#
port_number=50011
bench_mark="Tuft"
obj_name='teeth_20240308'
trigger_word='teeth'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="14_pretrained_vae_anomal_with_pe_rgb_img_pseudo_anomal"

anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 60 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --min_perlin_scale 1 \
 --max_perlin_scale 4 \
 --max_beta_scale 0.7 \
 --min_beta_scale 0.2