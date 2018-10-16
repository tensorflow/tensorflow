# TensorFlow-TensorRT Examples

This script will run inference using a few popular image classification models
on the ImageNet validation set.

You can turn on TensorFlow-TensorRT integration with the flag `--use_trt`. This
will apply TensorRT inference optimization to speed up execution for portions of
the model's graph where supported, and will fall back to native TensorFlow for
layers and operations which are not supported. See
https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/
for more information.

When using TF-TRT, you can also control the precision with `--precision`.
float32 is the default (`--precision fp32`) with float16 (`--precision fp16`) or
int8 (`--precision int8`) allowing further performance improvements.
int8 mode requires a calibration step which is done
automatically.

## Models

This test supports the following models for image classification:
* MobileNet v1
* MobileNet v2
* NASNet - Large
* NASNet - Mobile
* ResNet50 v1
* ResNet50 v2
* VGG16
* VGG19
* Inception v3
* Inception v4

## Setup
```
# Clone [tensorflow/models](https://github.com/tensorflow/models)
git clone https://github.com/tensorflow/models.git

# Add the models directory to PYTHONPATH to install tensorflow/models.
cd models
export PYTHONPATH="$PYTHONPATH:$PWD"

# Run the TF Slim setup.
cd research/slim
python setup.py install

# You may also need to install the requests package
pip install requests
```
Note: the PYTHONPATH environment variable will be not be saved between different
shells. You can either repeat that step each time you work in a new shell, or
add `export PYTHONPATH="$PYTHONPATH:/path/to/tensorflow_models"` to your .bashrc
file (replacing /path/to/tensorflow_models with the path to your
tensorflow/models repository).

### Data

The script supports only TFRecord format for data. The script
assumes that validation TFRecords are named according to the pattern:
`validation-*-of-00128`.

You can download and process Imagenet using [this script provided by TF
Slim](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_imagenet.sh).
Please note that this script downloads both the training and validation sets,
and this example only requires the validation set.

## Usage

`python inference.py --data_dir /imagenet_validation_data --model vgg_16 [--use_trt]`

Run with `--help` to see all available options.
