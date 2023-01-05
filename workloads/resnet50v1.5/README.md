# Image Classification
The refference implementation of Image Classification is based on the [NVIDIA Deeplearning Examples](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Classification/ConvNets/resnet50v1.5/README.md). The source code is maintained by NVIDIA. 

## Adopted Model
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is in the bottleneck blocks which requires
downsampling, for example, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.



### Feature support

The following features are supported by this model.

| Feature               | ResNet-50 v1.5 Tensorflow             |
|-----------------------|--------------------------
|Multi-GPU training with [Horovod](https://github.com/horovod/horovod)  |  Yes |
|[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html)                |  Yes |
|Automatic mixed precision (AMP) | Yes |


## Setup

The following section lists the requirements that you need to meet in order to use the ResNet50 v1.5 model.

### Requirements
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
- GPU-based architecture:
  - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)


For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry),
* [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running).

For those unable to use the [TensorFlow NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed precision or TF32 with Tensor Cores or FP32, perform the following steps using the default parameters of the ResNet-50 v1.5 model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/ConvNets
```

2. Download and preprocess the dataset.
The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/archive/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.

3. Build the ResNet-50 v1.5 TensorFlow NGC container.
```bash
docker build . -t nvidia_rn50
```

4. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with
```bash
nvidia-docker run --rm -it -v <path to imagenet>:/data/tfrecords --ipc=host nvidia_rn50
```

5. (Optional) Create index files to use DALI.
To allow proper sharding in a multi-GPU environment, DALI has to create index files for the dataset. To create index files, run inside the container:
```bash
bash ./utils/dali_index.sh /data/tfrecords <index file store location>
```
Index files can be created once and then reused. It is highly recommended to save them into a persistent location.

6. Start training.
To run training for a standard configuration (as described in [Default
configuration](#default-configuration), DGX1V, DGX2V, single GPU, FP16, FP32, 50, 90, and 250 epochs), run
one of the scripts int the `resnet50v1.5/training` directory. Ensure ImageNet is mounted in the
`/data/tfrecords` directory.

For example, to train on DGX-1 for 90 epochs using AMP, run: 

`bash ./resnet50v1.5/training/DGX1_RN50_AMP_90E.sh /path/to/result /data`

Additionally, features like DALI data preprocessing or TensorFlow XLA can be enabled with
following arguments when running those scripts:

`bash ./resnet50v1.5/training/DGX1_RN50_AMP_90E.sh /path/to/result /data --use_xla --use_dali`

7. Start validation/evaluation.
To evaluate the validation dataset located in `/data/tfrecords`, run `main.py` with
`--mode=evaluate`. For example:

`python main.py --mode=evaluate --data_dir=/data/tfrecords --batch_size <batch size> --model_dir
<model location> --results_dir <output location> [--use_xla] [--use_tf_amp]`

The optional `--use_xla` and `--use_tf_amp` flags control XLA and AMP during evaluation. 

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
 - `main.py`:               the script that controls the logic of training and validation of the ResNet-like models
 - `Dockerfile`:            Instructions for Docker to build a container with the basic set of dependencies to run ResNet like models for image classification
 - `requirements.txt`:      a set of extra Python requirements for running ResNet-like models

The `model/` directory contains the following modules used to define ResNet family models:
 - `resnet.py`: the definition of ResNet, ResNext, and SE-ResNext model
 - `blocks/conv2d_block.py`: the definition of 2D convolution block
 - `blocks/resnet_bottleneck_block.py`: the definition of ResNet-like bottleneck block
 - `layers/*.py`: definitions of specific layers used in the ResNet-like model
 
The `utils/` directory contains the following utility modules:
 - `cmdline_helper.py`: helper module for command line processing
 - `data_utils.py`: module defining input data pipelines
 - `dali_utils.py`: helper module for DALI 
 - `hvd_utils.py`: helper module for Horovod
 - `image_processing.py`: image processing and data augmentation functions
 - `learning_rate.py`: definition of used learning rate schedule
 - `optimizers.py`: definition of used custom optimizers
 - `hooks/*.py`: definitions of specific hooks allowing logging of training and inference process
 
The `runtime/` directory contains the following module that define the mechanics of the training process:
 - `runner.py`: module encapsulating the training, inference and evaluation  


### Parameters

#### The `main.py` script
The script for training and evaluating the ResNet-50 v1.5 model has a variety of parameters that control these processes.

```
usage: main.py [-h]
               [--arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}]
               [--mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}]
               [--data_dir DATA_DIR] [--data_idx_dir DATA_IDX_DIR]
               [--export_dir EXPORT_DIR] [--to_predict TO_PREDICT]
               [--batch_size BATCH_SIZE] [--num_iter NUM_ITER]
               [--iter_unit {epoch,batch}] [--warmup_steps WARMUP_STEPS]
               [--model_dir MODEL_DIR] [--results_dir RESULTS_DIR]
               [--log_filename LOG_FILENAME] [--display_every DISPLAY_EVERY]
               [--lr_init LR_INIT] [--lr_warmup_epochs LR_WARMUP_EPOCHS]
               [--weight_decay WEIGHT_DECAY] [--weight_init {fan_in,fan_out}]
               [--momentum MOMENTUM] [--loss_scale LOSS_SCALE]
               [--label_smoothing LABEL_SMOOTHING] [--mixup MIXUP]
               [--use_static_loss_scaling | --nouse_static_loss_scaling]
               [--use_xla | --nouse_xla] [--use_dali | --nouse_dali]
               [--use_tf_amp | --nouse_tf_amp]
               [--use_cosine_lr | --nouse_cosine_lr] [--seed SEED]
               [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]

JoC-RN50v1.5-TF

optional arguments:
  -h, --help            Show this help message and exit
  --arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}
                        Architecture of model to run (default is resnet50)
  --mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}
                        The execution mode of the script.
  --data_dir DATA_DIR   Path to dataset in TFRecord format. Files should be
                        named 'train-*' and 'validation-*'.
  --data_idx_dir DATA_IDX_DIR
                        Path to index files for DALI. Files should be named
                        'train-*' and 'validation-*'.
  --export_dir EXPORT_DIR
                        Directory in which to write exported SavedModel.
  --to_predict TO_PREDICT
                        Path to file or directory of files to run prediction
                        on.
  --batch_size BATCH_SIZE
                        Size of each minibatch per GPU.
  --num_iter NUM_ITER   Number of iterations to run.
  --iter_unit {epoch,batch}
                        Unit of iterations.
  --warmup_steps WARMUP_STEPS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --model_dir MODEL_DIR
                        Directory in which to write the model. If undefined,
                        results directory will be used.
  --results_dir RESULTS_DIR
                        Directory in which to write training logs, summaries
                        and checkpoints.
  --log_filename LOG_FILENAME
                        Name of the JSON file to which write the training log
  --display_every DISPLAY_EVERY
                        How often (in batches) to print out running
                        information.
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_warmup_epochs LR_WARMUP_EPOCHS
                        Number of warmup epochs for the learning rate schedule.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --momentum MOMENTUM   SGD momentum value for the momentum optimizer.
  --loss_scale LOSS_SCALE
                        Loss scale for FP16 training and fast math FP32.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --mixup MIXUP         The alpha parameter for mixup (if 0 then mixup is not
                        applied).
  --use_static_loss_scaling
                        Use static loss scaling in FP16 or FP32 AMP.
  --nouse_static_loss_scaling
  --use_xla             Enable XLA (Accelerated Linear Algebra) computation
                        for improved performance.
  --nouse_xla
  --use_dali            Enable DALI data input.
  --nouse_dali
  --use_tf_amp          Enable AMP to speedup FP32
                        computation using Tensor Cores.
  --nouse_tf_amp
  --use_cosine_lr       Use cosine learning rate schedule.
  --nouse_cosine_lr
  --seed SEED           Random seed.
  --gpu_memory_fraction GPU_MEMORY_FRACTION
                        Limit memory fraction used by the training script for DALI
  --gpu_id GPU_ID       Specify the ID of the target GPU on a multi-device platform.
                        Effective only for single-GPU mode.
  --quantize            Used to add quantization nodes in the graph (Default: Asymmetric quantization)
  --symmetric           If --quantize mode is used, this option enables symmetric quantization
  --use_qdq             Use quantize_and_dequantize (QDQ) op instead of FakeQuantWithMinMaxVars op for quantization. QDQ does only scaling.
  --finetune_checkpoint Path to pre-trained checkpoint which can be used for fine-tuning
  --quant_delay         Number of steps to be run before quantization starts to happen
```

### Quantization Aware Training
Quantization Aware training (QAT) simulates quantization during training by quantizing weights and activation layers. This will help reduce the loss in accuracy when we convert the network
trained in FP32 to INT8 for faster inference. QAT introduces additional nodes in the graph which will be used to learn the dynamic ranges of weights and activation layers. Tensorflow provides
a <a href="https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize">quantization tool</a> which automatically adds these nodes in-place. Typical workflow
for training QAT networks is to train a model until convergence and then finetune with the quantization layers. It is recommended that QAT is performed on a single GPU.

* For 1 GPU
    * Command: `sh resnet50v1.5/training/GPU1_RN50_QAT.sh <path to pre-trained ckpt dir> <path to dataset directory> <result_directory>`
        
It is recommended to finetune a model with quantization nodes rather than train a QAT model from scratch. The latter can also be performed by setting `quant_delay` parameter.
`quant_delay` is the number of steps after which quantization nodes are added for QAT. If we are fine-tuning, `quant_delay` is set to 0. 
        
For QAT network, we use <a href="https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/quantization/quantize_and_dequantize">tf.quantization.quantize_and_dequantize operation</a>.
These operations are automatically added at weights and activation layers in the RN50 by using `tf.contrib.quantize.experimental_create_training_graph` utility. Support for using `tf.quantization.quantize_and_dequantize` 
operations for `tf.contrib.quantize.experimental_create_training_graph` has been added in <a href="https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow">TensorFlow 20.01-py3 NGC container</a> and later versions, which is required for this task.

#### Post process checkpoint
  `postprocess_ckpt.py` is a utility to convert the final classification FC layer into a 1x1 convolution layer using the same weights. This is required to ensure TensorRT can parse QAT models successfully.
  This script should be used after performing QAT to reshape the FC layer weights in the final checkpoint.
  Arguments:
     * `--input` : Path to the trained checkpoint of RN50.
     * `--output` : Name of the new checkpoint file which has the FC layer weights reshaped into 1x1 conv layer weights.
     * `--dense_layer` : Name of the FC layer

### Exporting Frozen graphs
To export frozen graphs (which can be used for inference with <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>), use:

`python export_frozen_graph.py --checkpoint <path_to_checkpoint> --quantize --use_final_conv --use_qdq --symmetric --input_format NCHW --compute_format NCHW --output_file=<output_file_name>`

Arguments:

* `--checkpoint` : Optional argument to export the model with checkpoint weights.
* `--quantize` : Optional flag to export quantized graphs.
* `--use_qdq` : Use quantize_and_dequantize (QDQ) op instead of FakeQuantWithMinMaxVars op for quantization. QDQ does only scaling. 
* `--input_format` : Data format of input tensor (Default: NCHW). Use NCHW format to optimize the graph with TensorRT.
* `--compute_format` : Data format of the operations in the network (Default: NCHW). Use NCHW format to optimize the graph with TensorRT.

### Inference process
To run inference on a single example with a checkpoint and a model script, use: 

`python main.py --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`

The optional `--use_xla` and `--use_tf_amp` flags control XLA and AMP during inference.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32 / TF32

        `python ./main.py --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP

        `python ./main.py --mode=training_benchmark  --use_tf_amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32 / TF32

        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP
    
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --use_tf_amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags. Features like XLA or DALI can be controlled
with `--use_xla` and `--use_dali` flags. If no `--data_dir=<path to imagenet>` flag is specified then the benchmarks will use a synthetic dataset. 
Suggested batch sizes for training are 256 for mixed precision training and 128 for single precision training per single V100 16 GB.

