# Roadmap
**Last updated: Apr 27, 2018**

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
* Easy multi-GPU and TPU utilization with Estimators
* Easy-to-use high-level pre-made estimators for Gradient Boosted Trees, Time Series, and other models

#### Eager Execution:
* Efficient utilization of multiple GPUs
* Distributed training support (multi-machine)
* Performance improvements
* Simpler export to a GraphDef/SavedModel 

#### Keras API:
* Better integration with tf.data (ability to call `model.fit` with data tensors)
* Full support for Eager Execution (both Eager support for the regular Keras API, and ability 
to create Keras models Eager- style via Model subclassing)
* Better distribution/multi-GPU support and TPU support (including a smoother model-to-estimator workflow)

#### Official Models:
* A set of 
[models](https://github.com/tensorflow/models/tree/master/official) 
across image recognition, speech, object detection, and 
  translation that demonstrate best practices and serve as a starting point for 
  high-performance model development.

#### Contrib:
* Deprecate parts of tf.contrib where preferred implementations exist outside of tf.contrib.
* As much as possible, move large projects inside tf.contrib to separate repositories.
* The tf.contrib module will eventually be discontinued in its current form, experimental development will in future happen in other repositories.


#### Probabilistic Reasoning and Statistical Analysis:
* Rich set of tools for probabilistic and statistical analysis in tf.distributions 
  and tf.probability. These include new samplers, layers, optimizers, losses, and structured models
* Statistical tools for hypothesis testing, convergence diagnostics, and sample statistics
* Edward 2.0: High-level API for probabilistic programming

### Platforms
#### TensorFlow Lite:
* Increase coverage of supported ops in TensorFlow Lite
* Easier conversion of a trained TensorFlow graph for use on TensorFlow Lite
* Support for GPU acceleration in TensorFlow Lite (iOS and Android)
* Support for hardware accelerators via Android NeuralNets API 
* Improve CPU performance by quantization and other network optimizations (eg. pruning, distillation)
* Increase support for devices beyond Android and iOS (eg. RPi, Cortex-M)

#### TensorFlow.js:
* Release package for Node.js bindings to the TensorFlow C API through the TensorFlow.js backend interface
* Expand support for importing TensorFlow SavedModels and Keras models into browser with unified APIs supporting retraining in browser
* Improve Layers API and allow model exporting/saving
* Release tfjs-data API for efficient data input pipelines

#### TensorFlow with Swift:
* Establish open source project including documentation, open design, and code availability.
* Continue implementing and refining implementation and design through 2018.
* Aim for implementation to be solid enough for general use later in 2018.

### Performance
#### Distributed TensorFlow:
* Optimize Multi-GPU support for a variety of GPU topologies
* Improve mechanisms for distributing computations on several machines

#### GPU Optimizations:
* Simplify mixed precision API with initial example model and guide.
* Finalize TensorRT API and move to core.
* CUDA 9.2 and NCCL 2.x default in TensorFlow builds.
* Optimizations for DGX-2.
* Remove support for CUDA less than 8.x and cuDNN less than 6.x.


#### CPU Optimizations
* Int8 support for SkyLake via MKL
* Dynamic loading of SIMD-optimized kernels
* MKL for Linux and Windows

### End-to-end ML systems:
#### TensorFlow Hub:
* Expand support for module-types in TF Hub with TF Eager integration, Keras layers integration, and TensorFlow.js integration
* Accept variable-sized image input
* Improve multi-GPU estimator support
* Document and improve TPU integration

#### TensorFlow Extended:
* Open source more of the TensorFlow Extended platform to facilitate adoption of TensorFlow in production settings.
* Release TFX libraries for Data Validation

### Documentation and Resources:
* Update documentation, tutorials and Getting Started guides on all features and APIs
* Update [Youtube Tensorflow channel](https://youtube.com/tensorflow) weekly with new content:
Coding TensorFlow - where we teach folks coding with tensorflow
TensorFlow Meets - where we highlight community contributions
Ask TensorFlow - where we answer community questions
Guest and Showcase videos
* Update [Official TensorFlow blog](https://blog.tensorflow.org) with regular articles from Google team and the Community


### Community and Partner Engagement
#### Special Interest Groups: 
* Mobilize the community to work together in focused domains
* [tf-distribute](https://groups.google.com/a/tensorflow.org/forum/#!forum/tf-distribute): build and packaging of TensorFlow
* SIG TensorBoard, SIG Rust, and more to be identified and launched

#### Community:
* Incorporate public feedback on significant design decisions via a Request-for-Comment (RFC) process
* Formalize process for external contributions to land in TensorFlow and associated projects 
* Grow global TensorFlow communities and user groups
* Collaborate with partners to co-develop and publish research papers
* Process to enable external contributions to tutorials, documentation, and blogs showcasing best practice use-cases of TensorFlow and high-impact applications
