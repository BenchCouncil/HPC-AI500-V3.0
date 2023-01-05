# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script launches ResNet50 training in FP16 on 8 GPUs using 2048 batch size (256 per GPU)
# Usage ./RN50_FP16_8GPU.sh <path to this repository> <path to dataset> <path to results directory>
SOURCE_DIR="$( cd "$(dirname $BASH_SOURCE[0])/../" && pwd )"
DATA_DIR=/data/ImageNet-Tensorflow/train_tfrecord
OUTPUT_DIR=/data/imagenet-output/fp16-profiling/
mkdir -p ${OUTPUT_DIR}/models
rm -rf ${OUTPUT_DIR}/*

cat ${SOURCE_DIR}/scripts/fp16-metrics | while read metric iter_step
do
  date | tee -a ${OUTPUT_DIR}/${metric}-time.log
  start=$(date +%s)
  mpiexec --allow-run-as-root --bind-to socket -np 8 \
    /usr/local/cuda/bin/nvprof --log-file ${OUTPUT_DIR}/${metric}%q{OMPI_COMM_WORLD_RANK} --system-profiling on --csv -f --continuous-sampling-interval 1 --metrics ${metric} \
    python ${SOURCE_DIR}/main.py --mode=train --iter_unit=batch --num_iter=${iter_step} --batch_size=256 --warmup_steps=100 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 --use_tf_amp  --data_dir=${DATA_DIR} --results_dir=${OUTPUT_DIR}/models
  
  end=$(date +%s)
  take=$(( end - start ))
  
  date | tee -a ${OUTPUT_DIR}/${metric}-time.log
  echo "Time taken to execute commands is ${take} seconds." | tee -a ${OUTPUT_DIR}/${metric}-time.log

done 

