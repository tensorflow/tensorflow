# Roadmap
**Last updated: Feb 15, 2018**

TensorFlow is a rapidly moving, community supported project. This document is intended 
to provide guidance about priorities and focus areas of the core set of TensorFlow 
developers and about functionality that can be expected in the upcoming releases of 
TensorFlow. Many of these areas are driven by  community use cases, and we welcome 
further 
[contributions](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) 
to TensorFlow.

The features below do not have concrete release dates. However, the majority can be 
expected in the next one to two releases. 

### APIs
#### High Level APIs:
* Easy multi-GPU utilization with Estimators
* Easy-to-use high-level pre-made estimators for Gradient Boosted Trees, Time Series, and other models

#### Eager Execution:
* Efficient utilization of multiple GPUs
* Distributed training (multi-machine)
* Performance improvements
* Simpler export to a GraphDef/SavedModel 

#### Keras API:
* Better integration with tf.data (ability to call `model.fit` with data tensors)
* Full support for Eager Execution (both Eager support for the regular Keras API, and ability 
to create Keras models Eager- style via Model subclassing)
* Better distribution/multi-GPU support and TPU support (including a smoother model-to-estimator workflow)

#### Official Models:
* A set of 
[reference models](https://github.com/tensorflow/models/tree/master/official) 
across image recognition, speech, object detection, and 
  translation that demonstrate best practices and serve as a starting point for 
  high-performance model development.

#### Contrib:
* Deprecation notices added to parts of tf.contrib where preferred implementations exist outside of tf.contrib.
* As much as possible, large projects inside tf.contrib moved to separate repositories.
* The tf.contrib module will eventually be discontinued in its current form, experimental development will in future happen in other repositories.


#### Probabilistic Reasoning and Statistical Analysis:
* Rich set of tools for probabilistic and statistical analysis in tf.distributions 
  and tf.probability. These include new samplers, layers, optimizers, losses, and structured models
* Statistical tools for hypothesis testing, convergence diagnostics, and sample statistics
* Edward 2.0: High-level API for probabilistic programming

### Platforms
#### TensorFlow Lite:
* Increased coverage of supported ops in TensorFlow Lite
* Easier conversion of a trained TensorFlow graph for use on TensorFlow Lite
* Support for GPU acceleration in TensorFlow Lite (iOS and Android)
* Support for hardware accelerators via Android NeuralNets API 
* Improved CPU performance by quantization and other network optimizations (eg. pruning, distillation)
* Increased support for devices beyond Android and iOS (eg. RPi, Cortex-M)

### Performance
#### Distributed TensorFlow:
* Multi-GPU support optimized for a variety of GPU topologies
* Improved mechanisms for distributing computations on several machines

#### Optimizations:
* Mixed precision training support with initial example model and guide
* Native TensorRT support
* Int8 support for SkyLake via MKL
* Dynamic loading of SIMD-optimized kernels

### Documentation and Usability:
* Updated documentation, tutorials and Getting Started guides
* Process to enable external contributions to tutorials, documentation, and blogs showcasing best practice use-cases of TensorFlow and high-impact applications

### Community and Partner Engagement
#### Special Interest Groups: 
* Mobilizing the community to work together in focused domains
* [tf-distribute](https://groups.google.com/a/tensorflow.org/forum/#!forum/tf-distribute): build and packaging of TensorFlow
* More to be identified and launched

#### Community:
* Incorporate public feedback on significant design decisions via a Request-for-Comment (RFC) process
* Formalize process for external contributions to land in TensorFlow and associated projects 
* Grow global TensorFlow communities and user groups
* Collaborate with partners to co-develop and publish research papers
