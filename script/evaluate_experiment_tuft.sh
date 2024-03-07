#! /bin/bash

bench_mark="Tuft"
obj_name='teeth_crop_onlyanormal'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="1_pretrained_vae_anomal_data_with_pe"

dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
base_dir="../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/reconstruction_with_test_data"

output_dir="metrics"

python ../evaluation/evaluation_code_MVTec/evaluate_tuft.py \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --evaluated_objects "${obj_name}" \
     --pro_integration_limit 0.3