# !/bin/bash
#
port_number=50001
bench_mark="brain"
obj_name='brain'
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="1_pretrained_vae_pe_xray_anomal"
# \
#--do_self_aug
accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 100 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --anomal_source_path "../../../MyData/anomal_source" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --min_perlin_scale 1 \
 --max_perlin_scale 4 \
 --max_beta_scale 0.9 \
 --min_beta_scale 0.5