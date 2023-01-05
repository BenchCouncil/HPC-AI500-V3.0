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
SOURCE_DIR="$(cd "$(dirname $BASH_SOURCE[0])/../" && pwd)"
DATA_DIR=/data/ImageNet-Tensorflow/train_tfrecord
DATA_IDX_DIR=/data/ImageNet-Tensorflow/train_imagenet_idx
NUM_EPOCHS=90
SEED=10
OUTPUT_DIR=/data/imagenet-output/fp32-${NUM_EPOCHS}-SEED${SEED}-8-8-nolrandwarmup
rm -rf ${OUTPUT_DIR}/models
mkdir -p ${OUTPUT_DIR}
cp ./fp32_change_hyper.sh ${OUTPUT_DIR}
cp ./hostfile-4 ${OUTPUT_DIR}/

date | tee -a ${OUTPUT_DIR}/time.log
start=$(date +%s)
mpiexec --allow-run-as-root --bind-to none -map-by slot -np 64 --hostfile "./hostfile-8" -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=info \
    python ${SOURCE_DIR}/main.py --mode=train --iter_unit=epoch --num_iter=${NUM_EPOCHS} --batch_size=128 --warmup_steps=0 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.008 --lr_warmup_epochs=0 --momentum=0.875 --weight_decay=3.0517578125e-05 --data_dir=${DATA_DIR} --results_dir=${OUTPUT_DIR}/models --seed=${SEED} 

end=$(date +%s)
take=$(( end - start ))

date | tee -a ${OUTPUT_DIR}/time.log
echo "Time taken to execute commands is ${take} seconds." | tee ${OUTPUT_DIR}/time.log

./fp32-eval.sh ${OUTPUT_DIR}
