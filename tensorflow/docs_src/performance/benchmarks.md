# TensorFlow Performance Benchmarks

## Overview

A selection of image classification models were tested across multiple platforms
to create a point of reference for the TensorFlow community. The methodology,
links to the scripts, and commands to reproduce the results are in the
[appendix](#appendix).

## Results for image classification models

InceptionV3 ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567)),
ResNet-50 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)),
ResNet-152 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), VGG16
([arXiv:1409.1556](https://arxiv.org/abs/1409.1556)), and
[AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
were tested using the [ImageNet](http://www.image-net.org/) data set. Tests were
run on Google Compute Engine, Amazon Elastic Compute Cloud (Amazon EC2), and an
NVIDIA® DGX-1™. Most of the tests were run with both synthetic and real data.
Testing with synthetic data was done by using a `tf.Variable` set to the same
shape as the data expected by each model for ImageNet. We believe it is
important to include real data measurements when benchmarking a platform. This
load tests both the underlying hardware and the framework at preparing data for
actual training. We start with synthetic data to remove disk I/O as a variable
and to set a baseline. Real data is then used to verify that the TensorFlow
input pipeline and the underlying disk I/O are saturating the compute units.

### Training with NVIDIA® DGX-1™ (NVIDIA® Tesla® P100)

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/perf_summary_p100_single_server.png">
</div>

Details and additional results are in the [Details for NVIDIA® DGX-1™ (NVIDIA®
Tesla® P100)](#details-for-nvidia®-dgx-1™-nvidia®-tesla®-p100) section.

### Training with NVIDIA® Tesla® K80

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/perf_summary_k80_single_server.png">
</div>

Details and additional results are in the [Details for Google Compute Engine
(NVIDIA® Tesla® K80)](#details-for-google-compute-engine-nvidia®-tesla®-k80) and
[Details for Amazon EC2 (NVIDIA® Tesla®
K80)](#details-for-amazon-ec2-nvidia®-tesla®-k80) sections.

### Distributed training with NVIDIA® Tesla® K80

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/perf_summary_k80_aws_distributed.png">
</div>

Details and additional results are in the [Details for Amazon EC2 Distributed
(NVIDIA® Tesla® K80)](#details-for-amazon-ec2-distributed-nvidia®-tesla®-k80)
section.

### Compare synthetic with real data training

**NVIDIA® Tesla® P100**

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_summary_p100_data_compare_inceptionv3.png">
  <img style="width:35%" src="../images/perf_summary_p100_data_compare_resnet50.png">
</div>

**NVIDIA® Tesla® K80**

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_summary_k80_data_compare_inceptionv3.png">
  <img style="width:35%" src="../images/perf_summary_k80_data_compare_resnet50.png">
</div>

## Details for NVIDIA® DGX-1™ (NVIDIA® Tesla® P100)

### Environment

*   **Instance type**: NVIDIA® DGX-1™
*   **GPU:** 8x NVIDIA® Tesla® P100
*   **OS:** Ubuntu 16.04 LTS with tests run via Docker
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** local SSD
*   **DataSet:** ImageNet

Batch size and optimizer used for each model.

                   | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 64         | 512     | 64
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

Configuration used for each model.

Model       | variable_update        | local_parameter_device
----------- | ---------------------- | ----------------------
InceptionV3 | parameter_server       | cpu
ResNet50    | parameter_server       | cpu
ResNet152   | parameter_server       | cpu
AlexNet     | replicated (with NCCL) | n/a
VGG16       | replicated (with NCCL) | n/a

### Results

Batch size and optimizer used for each model are listed in the table below.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../images/perf_summary_p100_single_server.png">
</div>

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_dgx1_synth_p100_single_server_scaling.png">
  <img style="width:35%" src="../images/perf_dgx1_real_p100_single_server_scaling.png">
</div>

**Training synthetic data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 142         | 238       | 95.6       | 2987    | 132
2    | 284         | 479       | 187        | 5658    | 259
4    | 569         | 948       | 374        | 10509   | 511
8    | 1131        | 1886      | 744        | 17822   | 959

**Training real data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 142         | 239       | 95.5       | 2890    | 132
2    | 278         | 468       | 187        | 4448    | 245
4    | 551         | 938       | 373        | 7105    | 466
8    | 1079        | 1802      | 721        | N/A     | 794

Training AlexNet with real data on 8 GPUs was excluded from the graph and table
above due to it maxing out the input pipeline.

## Details for Google Compute Engine (NVIDIA® Tesla® K80)

### Environment

*   **Instance type**: n1-standard-32-k80x8
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1.7 TB Shared SSD persistent disk (800 MB/s)
*   **DataSet:** ImageNet
*   **Test Date:** April 2017

Batch size and optimizer used for each model are listed in the table below. In
addition to the batch sizes listed in the table, InceptionV3 and ResNet-50 were
tested with a batch size of 32. Those results are in the *other results*
section.

                   | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 32         | 512     | 32
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

The configuration used for each model was `variable_update` equal to
`parameter_server` and `local_parameter_device` equal to `cpu`.

### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_gce_synth_k80_single_server_scaling.png">
  <img style="width:35%" src="../images/perf_gce_real_k80_single_server_scaling.png">
</div>

**Training synthetic data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.5        | 56.8      | 20.8       | 656     | 30.3
2    | 57.8        | 107       | 39.1       | 1210    | 56.2
4    | 116         | 212       | 77.2       | 2330    | 106
8    | 227         | 419       | 151        | 4640    | 222

**Training real data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.5        | 56.7      | 20.7       | 639     | 30.2
2    | 57.8        | 107       | 39         | 1136    | 55.5
4    | 115         | 211       | 77.3       | 2067    | 106
8    | 225         | 418       | 150        | 4056    | 213

### Other Results

**Training synthetic data**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.3                        | 53.9
2    | 55.0                        | 101
4    | 109                         | 200
8    | 216                         | 398

**Training real data**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.3                        | 53.6
2    | 55                          | 102
4    | 109                         | 200
8    | 215                         | 387

## Details for Amazon EC2 (NVIDIA® Tesla® K80)

### Environment

*   **Instance type**: p2.8xlarge
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1TB Amazon EFS (burst 100 MiB/sec for 12 hours, continuous 50
    MiB/sec)
*   **DataSet:** ImageNet
*   **Test Date:** April 2017

Batch size and optimizer used for each model are listed in the table below. In
addition to the batch sizes listed in the table, InceptionV3 and ResNet-50 were
tested with a batch size of 32. Those results are in the *other results*
section.

                   | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 32         | 512     | 32
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

Configuration used for each model.

Model       | variable_update           | local_parameter_device
----------- | ------------------------- | ----------------------
InceptionV3 | parameter_server          | cpu
ResNet-50   | replicated (without NCCL) | gpu
ResNet-152  | replicated (without NCCL) | gpu
AlexNet     | parameter_server          | gpu
VGG16       | parameter_server          | gpu

### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_aws_synth_k80_single_server_scaling.png">
  <img style="width:35%" src="../images/perf_aws_real_k80_single_server_scaling.png">
</div>

**Training synthetic data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.8        | 56.3      | 20.9       | 684     | 32.4
2    | 58.7        | 108       | 39.3       | 1244    | 61.5
4    | 117         | 217       | 79.1       | 2479    | 123
8    | 230         | 419       | 156        | 4853    | 234

**Training real data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | Alexnet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.5        | 56.0      | 20.6       | 674     | 32.0
2    | 58.7        | 107       | 39.0       | 1227    | 61.0
4    | 118         | 205       | 77.9       | 2201    | 120
8    | 228         | 405       | 152        | N/A     | 191

Training AlexNet with real data on 8 GPUs was excluded from the graph and table
above due to our EFS setup not providing enough throughput.

### Other Results

**Training synthetic data**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.9                        | 53.5
2    | 57.5                        | 101
4    | 114                         | 202
8    | 216                         | 380

**Training real data**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 30.0                        | 53.6
2    | 57.5                        | 101
4    | 113                         | 202
8    | 212                         | 379

## Details for Amazon EC2 Distributed (NVIDIA® Tesla® K80)

### Environment

*   **Instance type**: p2.8xlarge
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1.0 TB EFS (burst 100 MB/sec for 12 hours, continuous 50 MB/sec)
*   **DataSet:** ImageNet
*   **Test Date:** April 2017

The batch size and optimizer used for the tests are listed in the table. In
addition to the batch sizes listed in the table, InceptionV3 and ResNet-50 were
tested with a batch size of 32. Those results are in the *other results*
section.

                   | InceptionV3 | ResNet-50 | ResNet-152
------------------ | ----------- | --------- | ----------
Batch size per GPU | 64          | 64        | 32
Optimizer          | sgd         | sgd       | sgd

Configuration used for each model.

Model       | variable_update        | local_parameter_device
----------- | ---------------------- | ----------------------
InceptionV3 | distributed_replicated | n/a
ResNet-50   | distributed_replicated | n/a
ResNet-152  | distributed_replicated | n/a

To simplify server setup, EC2 instances (p2.8xlarge) running worker servers also
ran parameter servers. Equal numbers of parameter servers and work servers were
used with the following exceptions:

*   InceptionV3: 8 instances / 6 parameter servers
*   ResNet-50: (batch size 32) 8 instances / 4 parameter servers
*   ResNet-152: 8 instances / 4 parameter servers

### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:95%" src="../images/perf_summary_k80_aws_distributed.png">
</div>

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:70%" src="../images/perf_aws_synth_k80_distributed_scaling.png">
</div>

**Training synthetic data**

GPUs | InceptionV3 | ResNet-50 | ResNet-152
---- | ----------- | --------- | ----------
1    | 29.7        | 55.0      | 19.8
8    | 229         | 410       | 150
16   | 459         | 825       | 300
32   | 902         | 1468      | 575
64   | 1783        | 3051      | 1004

### Other Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:50%" src="../images/perf_aws_synth_k80_multi_server_batch32.png">
</div>

**Training synthetic data**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.2                        | 53.0
8    | 219                         | 363
16   | 427                         | 719
32   | 820                         | 1265
64   | 1608                        | 2623

## Appendix

### Executing benchmark tests

The code for the benchmarks was created to be both used for benchmarking
TensorFlow as well as used as a tool to test hardware platforms. The benchmark
code includes modes such as `trivial` that run a virtually empty model that is
useful for testing the maximum possibly samples/sec for the input pipeline among
other things. Not only does this test TensorFlow but also the throughput of the
underlying systems. There are two ways to execute the benchmarks in
[tf_cnn_benchmarks.py](TODO: LINK TO GITHUB):

1.  Execute [tf_cnn_benchmarks.py](TODO: LINK TO GITHUB) directly
2.  Utilize the [small wrapper](TODO: LINK TO GITHUB) that helps pick the
    correct config

The wrapper is suggested as a starting point. Then investigate the variety of
options available in `tf_cnn_benchmarks.py`. While the wrapper extensive
examples, below are a couple highlights.

Run ResNet-50 on a single instance with 8 GPUs. The `system` argument is used to
determine the optimal configuration. The supported values are gce, aws, and
dgx1. If `system` is not passeed, the best config for the most widely available
hardware is used.

```bash
python main.py --model=resnet50 --num_gpus=8
python main.py --system=aws --model=resnet50 --num_gpus=8
```

Run ResNet-50 on 2 hosts, e.g. host_0 (10.0.0.1) and host_1 (10.0.0.2), with 8
GPUs each on aws.

```bash
# Run the following commands on host_0 (10.0.0.1):
  $  python main.py --system=aws --model=resnet50 --job_name=worker
     --hosts=10.0.0.1,10.0.0.2 --task_index=0

  $  python main.py --system=aws --model=resnet50 --job_name=ps
     --hosts=10.0.0.1,10.0.0.2 --task_index=0

# Run the following commands on host_1 (10.0.0.2):
  $  python main.py --system=aws --model=resnet50 --job_name=worker
     --hosts=10.0.0.1,10.0.0.2 --task_index=1

  $  python main.py --system=aws --model=resnet50 --job_name=ps
     --hosts=10.0.0.1,10.0.0.2 --task_index=1
```

### Methodology

Unless otherwise stated, each test is run 5 times and then the times are
averaged together. GPUs are run in their default state on the given platform.
For NVIDIA® Tesla® K80 this means leaving on [GPU
Boost](https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/)
unless it has been turned off by the provider. For a given test, 10 warmup steps
are done and then the next 100 steps are averaged.
