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

# This script evaluates ResNet50 model in FP16 using 256 batch size on 1 GPU
# Usage: ./RN50_FP16_EVAL.sh <path to this repository> <path to dataset> <path to model directory>


SOURCE_DIR="$(cd "$(dirname $BASH_SOURCE[0])/../" && pwd)"

DATA_DIR=/data/ImageNet-Tensorflow/validation_tfrecord

# CHECKPOINT_DIR=/data/imagenet-output/fp32-90-SEED10-8-8-dali
CHECKPOINT_DIR=$1
echo ${CHECKPOINT_DIR}

sleep 10

for ckpt_path in ${CHECKPOINT_DIR}/models/model.ckpt-*.index
do
    if [ -d ${ckpt_path} ]; then
	continue
    fi
    ckpt_file=$(basename ${ckpt_path})
    ckpt_dir="${ckpt_file%.*}"

    rm -rf ${CHECKPOINT_DIR}/tmp
    mkdir -p ${CHECKPOINT_DIR}/tmp
    cp -r ${ckpt_path%.*}* ${CHECKPOINT_DIR}/graph.pbtxt ${CHECKPOINT_DIR}/tmp/
    echo "model_checkpoint_path: \"${ckpt_dir}\"" > ${CHECKPOINT_DIR}/tmp/checkpoint
    echo "Processing "$ckpt_path

    date | tee -a ${CHECKPOINT_DIR}/tmp/eval-time.log
    start=$(date +%s)
    #nvprof --metrics ipc --log-file ${OUTPUT}/ipc --system-profiling on --csv -f --continuous-sampling-interval 1 \
    python ${SOURCE_DIR}/main.py --mode=evaluate --data_dir=${DATA_DIR} --batch_size=128 --num_iter=1 --iter_unit=epoch --results_dir=${CHECKPOINT_DIR}/tmp
    # python ${SOURCE_DIR}/main.py --mode=evaluate --data_dir=${DATA_DIR} --batch_size=256 --num_iter=1 --iter_unit=epoch --use_tf_amp --results_dir=${ckpt_file}
    
    end=$(date +%s)
    take=$(( end - start ))
    
    date | tee -a ${CHECKPOINT_DIR}/tmp/eval-time.log
    echo "Time taken to execute commands is ${take} seconds." | tee -a ${CHECKPOINT_DIR}/tmp/eval-time.log

    # ckpt_dir=$(echo ${ckpt_file} | cut -d . -f1)
    # ckpt_dir="${ckpt_file%.*}"
    echo "Making directory "eval-${ckpt_dir}
    mkdir -p ${CHECKPOINT_DIR}/eval-${ckpt_dir}
    rm -rf ${CHECKPOINT_DIR}/tmp/${ckpt_dir}* ${CHECKPOINT_DIR}/eval-${ckpt_dir}
    echo "Rename "${CHECKPOINT_DIR}/tmp/" to "${CHECKPOINT_DIR}/eval-${ckpt_dir}
    mv ${CHECKPOINT_DIR}/tmp ${CHECKPOINT_DIR}/eval-${ckpt_dir}

done


