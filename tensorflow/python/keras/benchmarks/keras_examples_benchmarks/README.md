# Benchmarks for keras model exmaples

- [Benchmarks for keras model exmaples](#benchmarks-for-keras-model-exmaples)
  - [Keras Benchmarks](#keras-benchmarks)
    - [Available models](#available-models)
      - [Computer Vision examples](#computer-vision-examples)
      - [Text & Sequence examples](#text--sequence-examples)
      - [Other examples](#other-examples)
    - [Available benchmark results](#available-benchmark-results)
      - [Cifar10 CNN benchmark](#cifar10-cnn-benchmark)
      - [MNIST Conv benchmark](#mnist-conv-benchmark)
      - [MNIST Hierarchical RNN (HRNN) benchmark](#mnist-hierarchical-rnn-hrnn-benchmark)
      - [Bidirectional LSTM benchmark](#bidirectional-lstm-benchmark)
      - [Text classification with transformer benchmark](#text-classification-with-transformer-benchmark)
      - [MLP benchmark](#mlp-benchmark)
      - [Antirectifier benchmark](#antirectifier-benchmark)
      - [IRNN benchmark](#irnn-benchmark)
  - [Installing Bazel](#installing-bazel)
  - [How to run benchmarks](#how-to-run-benchmarks)
  - [How to add new benchmark tests that use `fit`](#how-to-add-new-benchmark-tests-that-use-fit)
  - [Troubleshooting](#troubleshooting)
  
## Keras Benchmarks

These are benchmark tests running on keras models: models from [keras/examples](https://github.com/keras-team/keras/tree/master/examples). Benchmarks in the current folder (`tensorflow/python/keras/benchmarks/keras_examples_benchmarks`) use Keras [built-in dataset](https://keras.io/api/datasets/) or synthetic data. In addition, these benchmarks support different distribution strategies and measure the performance with distributed training.

### Available models

These examples are implemented by functional API and Sequential API.

#### Computer Vision examples

- [cifar10_cnn_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/cifar10_cnn_benchmark_test.py): Simple CNN on CIFAR10 image dataset.
- [mnist_conv_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_conv_benchmark_test.py): Simple Convnet that achieves ~99% test accuracy on MNIST.
- [mnist_hierarchical_rnn_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_hierarchical_rnn_benchmark_test.py): Hierarchical RNN (HRNN) to classify MNIST digits.

#### Text & Sequence examples

[Bidirectional_lstm_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/bidirectional_lstm_benchmark_test.py): 2-layer bidirectional LSTM on IMDB movie review dataset.
[text_classification_transformer_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/text_classification_transformer_benchmark_test.py): Text classification with custom transformer block.
[reuters_mlp_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/reuters_mlp_benchmark_test.py): Simple MLP on Reuters newswire topic classification dataset.

#### Other examples

[antirectifier_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/antirectifier_benchmark_test.py): Simple custom layer example.
[mnist_irnn_benchmark_test.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks/mnist_irnn_benchmark_test.py): Reproduction of the IRNN experiment with pixel-by-pixel sequential MNIST in ["A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"](https://arxiv.org/abs/1504.00941) by Le et al.

### Available benchmark results

We run benchmarks on Google Cloud Platform (GCP) and here is current environment for running benchmarks tests: </br>
GPU: 2 x Tesla V100 (only for GPU test) </br>
OS: Ubuntu 18.04 </br>
CPU: 8 x vCPUs, 30 GB memory </br>
CUDA: 10.1 </br>
Bazel: 3.1.0 </br>

If you want to run benchmark tests on GPU, please make sure you already installed CUDA and other dependencies and you can follow the instructions from the [official tutorial](https://www.tensorflow.org/install/gpu) for GPU support.

#### Cifar10 CNN benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    256     | 1393.4896 |      3.21      |  15397.69   |         `off`         |
| GPU:2 |    256     |   76.49   |      2.59      |  18758.01   |      `mirrored`       |

#### MNIST Conv benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    256     |  196.52   |     12.19      |   4915.26   |         `off`         |
| GPU:2 |    256     |  24.5794  |      1.21      |  47899.32   |      `mirrored`       |

#### MNIST Hierarchical RNN (HRNN) benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    256     |  654.05   |     218.68     |   274.24    |         `off`         |
| GPU:2 |    256     |   20.77   |      3.73      |  15088.06   |      `mirrored`       |

#### Bidirectional LSTM benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    512     |  225.57   |     72.55      |   344.70    |         `off`         |
| GPU:2 |    512     |   23.54   |      3.23      |   7532.53   |      `mirrored`       |

#### Text classification with transformer benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    512     |  109.22   |     35.93      |   698.10    |         `off`         |
| GPU:2 |    512     |   9.28    |      0.83      |  26567.54   |      `mirrored`       |

#### MLP benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    128     |   3.76    |      0.54      |  17678.54   |         `off`         |
| GPU:2 |    128     |   5.91    |      0.30      |  25435.14   |      `mirrored`       |

#### Antirectifier benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    512     |   6.77    |      1.79      |  30916.39   |         `off`         |
| GPU:2 |    512     |   6.81    |      0.66      |  66563.17   |      `mirrored`       |

#### IRNN benchmark

|       | Batch_size | Wall_time | Avg_epoch_time | Exp_per_sec | Distribution_Strategy |
| :---: | :--------: | :-------: | :------------: | :---------: | :-------------------: |
|  CPU  |    1024    |  213.00   |     69.01      |   868.08    |         `off`         |
| GPU:2 |    1024    |   92.71   |     29.12      |   2042.94   |      `mirrored`       |

**Note**: For the smaller models, running models with GPU may be slower than running models with CPU as training small models is not computation dominant and there might be some overhead on model replication and data sharding with distributed training on GPUs.

## Installing Bazel

This step can be skipped if Bazel is already installed. </br>

We need to use [Bazel](https://bazel.build/) to build targets based on BUILD files. It will take a while for the first time because it will compile all dependencies from your BUILD file. For the next time, Bazel will use the cache and it’ll be much faster. Since we use Ubuntu OS, we can install bazel by using apt repository.

1. Add bazel as package source

   ```shell
   sudo apt install curl gnupg
   ```

   ```shell
   curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
   ```

   ```shell
   echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
   ```

   Before we install the bazel, We should take a look for a bazel version that can build the specific tensorflow version, you can check it from [here](https://www.tensorflow.org/install/source#tested_build_configurations). In addition, you can follow the instructions from [Bazel website](https://docs.bazel.build/versions/3.4.0/install.html).

2. Install Bazel

   ```shell
   sudo apt update && sudo apt install bazel-`version`
   ```

## How to run benchmarks

To run benchmarks in [keras/benchmarks](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/benchmarks), please take the following steps:

1. Pull the latest tensorflow repo from github.
2. Install the Bazel tool which works with tensorflow, please take a look for the Tool installation section.
3. To run benchmarks with Bazel, use the `--benchmarks=.` flags to specify the benchmarks to run.
    - To run all benchmarks on CPU

     ```shell
     bazel run -c opt benchmark_test -- --benchmarks=.
     ```

    - To run all benchmarks on GPU

     ```shell
     bazel run run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \ --benchmarks=.
     ```

    - To run a subset of benchmarks using `--benchmarks` flag, `--benchmarks`: the list of benchmarks to run. The specified value is interpreted as a regular expression and any benchmarks whose name contains a partial match to the regular expression is executed. e.g. `--benchmarks=".*lstm*."`, will run all lstm layer related benchmarks.

## How to add new benchmark tests that use `fit`

To add a new benchmark, please follow the steps:

1. Create your own benchmark test file, `xxxx_benchmark_test.py`.
2. Import `benchmark_util` to measure and track performance.
3. Create class which inherits from `tf.test.Benchmark`
4. Define and load dataset in `__init__` method.
5. Design and create a model in `_build_model` method.
6. Define the `benchmark_xxx` method and it will pass essential parameters, which includes `batch_size`, `run_iters`, `train_data` and etc. You can check examples from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/benchmarks/keras_examples_benchmarks).
7. In addition, you need to add a benchmark target in the [BUILD](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/benchmarks/BUILD) file and write the target name and dependencies. You can take current BUILD as a reference.

## Troubleshooting

1. tensorflow.python.framework.errors_impl.InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid
  
   - Make sure CUDA was installed on your machine.
   - Pull the latest tensorflow repo and run the `./configure` in the root folder of tensorflow, it will help you to create the configuration file which shows your local environment. Please check [this post](https://www.tensorflow.org/install/source#configure_the_build) to know the details.