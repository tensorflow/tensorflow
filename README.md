<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://securityscorecards.dev/viewer/?uri=github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/44)](https://ossrank.com/p/44)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![TF Official Continuous](https://tensorflow.github.io/build/TF%20Official%20Continuous.svg)](https://tensorflow.github.io/build#TF%20Official%20Continuous)
[![TF Official Nightly](https://tensorflow.github.io/build/TF%20Official%20Nightly.svg)](https://tensorflow.github.io/build#TF%20Official%20Nightly)

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

### Table of Contents
1. [Introduction](https://github.com/saagar-pat/tensorflow/edit/master/README.md#introduction)
2. [Installation](https://github.com/saagar-pat/tensorflow/edit/master/README.md#install)
<br />2.1. [Using pip](https://github.com/saagar-pat/tensorflow/edit/master/README.md#install-using-pip)
<br />2.2. [Enabling GPU Support](https://github.com/saagar-pat/tensorflow/edit/master/README.md#enabling-gpu-support)
<br />2.3. [Docker Installation](https://github.com/saagar-pat/tensorflow/edit/master/README.md#docker-installation)
<br />2.4. [Building from Source](https://github.com/saagar-pat/tensorflow/edit/master/README.md#building-from-source)
<br />2.5. [Getting Started](https://github.com/saagar-pat/tensorflow/edit/master/README.md#getting-started)
4. [Contributing](https://github.com/saagar-pat/tensorflow/edit/master/README.md#patching-contributing)
5. [Patching Guidelines](https://github.com/saagar-pat/tensorflow/edit/master/README.md#patching-guidelines)
6. [Continuous Build Status](https://github.com/saagar-pat/tensorflow/edit/master/README.md#continuous-build-status)
7. [Official Builds](https://github.com/saagar-pat/tensorflow/edit/master/README.md#official-builds)
8. [Resources](https://github.com/saagar-pat/tensorflow/edit/master/README.md#resources)
9. [Courses](https://github.com/saagar-pat/tensorflow/edit/master/README.md#courses)
10. [License](https://github.com/saagar-pat/tensorflow/edit/master/README.md#license)


## Introduction
[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML-powered applications.

TensorFlow was originally developed by researchers and engineers working within
the Machine Intelligence team at Google Brain to conduct research in machine
learning and neural networks. However, the framework is versatile enough to be
used in other areas as well.

TensorFlow is a powerful tool and is behind some well-known and widely used projects such as:
*   **Google Translate**: Created by Google and used by numerous real-time translations.
*   **DeepMind’s AlphaGo**: Google's AI that beat world champions at Go a decade before experts thought possible.
*   **Snapchat**: Used for facial recognition and interactive filters.

Whether you are a researcher, data scientist, developer, or entrepenuer, TensorFlow gives you the tools necessary to explore the potential of AI and machine learning.

TensorFlow provides stable [Python](https://www.tensorflow.org/api_docs/python)
and [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as a
non-guaranteed backward compatible API for
[other languages](https://www.tensorflow.org/api_docs).

Keep up-to-date with release announcements and security updates by subscribing
to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
See all the [mailing lists](https://www.tensorflow.org/community/forums).

## Install

TensorFlow provides multiple installation options depending on your device's setup and requirements. The default installation utilizes the CPU, but GPU support can be enables for larger more processing intensive tasks.

See the [TensorFlow install guide](https://www.tensorflow.org/install) for the
[pip package](https://www.tensorflow.org/install/pip), to
[enable GPU support](https://www.tensorflow.org/install/gpu), use a
[Docker container](https://www.tensorflow.org/install/docker), and
[build from source](https://www.tensorflow.org/install/source).

### Install Using Pip
To install the current release, which includes support for
[CUDA-enabled GPU cards](https://www.tensorflow.org/install/gpu) *(Ubuntu and
Windows)*:

```
$ pip install tensorflow
```

Other devices (DirectX and MacOS-metal) are supported using
[Device plugins](https://www.tensorflow.org/install/gpu_plugins#available_devices).

A smaller CPU-only package is also available:

```
$ pip install tensorflow-cpu
```

To update TensorFlow to the latest version, add `--upgrade` flag to the above
commands.

### Enabling GPU Support
TensorFlow supports [CUDA-enabled GPUs](https://www.tensorflow.org/install/gpu) for acceleration. Ensure that you have the appropriate CUDA and cuDNN libraries installed.

```
$ pip install tensorflow
```

### Docker Installation
Docker is an easy way to run Tensorflow within an isolated enviroment without having to worry about installation conflicts or managing any dependent programs on your local machine. Before installing TensorFlow in a Docker container, be sure to have [Docker](https://docs.docker.com/engine/install/) installed on your system.

To run TensorFlow in a Docker container run:
```
$ docker pull tensorflow/tensorflow
```

For GPU-enabled Docker containers run:
```
$ docker pull tensorflow/tensorflow:latest-gpu
```

### Building From Source
If you require more control over the build process or need certain features that are not available in the prebuilt libraries you should build [TensorFlow install from Source](https://www.tensorflow.org/install/source).

*Nightly binaries are available for testing using the
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) packages on PyPi.*

TensorFlow is also able to run in a variety of instances on many different platforms. 
Please visit any of these links for respective download/installation documentation:
*   [LiteRT (Formerly known as Tensorflow Lite)](https://ai.google.dev/edge/litert) for mobile devices. 
*   [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is made for another instance in which high-performance model and machine learning algorithms on servers.
*   [TensorFlow.js](https://www.tensorflow.org/js) which is a machine learning library, capable of developing and testing machine learning models, nestled within JavaScript.

#### *Try your first TensorFlow program*

```shell
$ python
```

```python
>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
b'Hello, TensorFlow!'
```

For more examples, see the
[TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the
[contribution guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, please see
[TensorFlow Forum](https://discuss.tensorflow.org/) for general questions and
discussion, and please direct specific questions to
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow). Within GitHub issues, Labels are implemented for clarity, for first time contributors please refer to the [contribution guidelines](CONTRIBUTING.md) document, and we would recommend filtering by the [good-first-issue](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue) label. **

The TensorFlow project strives to abide by generally accepted best practices in
open-source software development.

## Patching guidelines

Follow these steps to patch a specific version of TensorFlow, for example, to
apply fixes to bugs or security vulnerabilities:

*   Clone the TensorFlow repo and switch to the corresponding branch for your
    desired TensorFlow version, for example, branch `r2.8` for version 2.8.
*   Apply (that is, cherry-pick) the desired changes and resolve any code
    conflicts.
*   Run TensorFlow tests and ensure they pass.
*   [Build](https://www.tensorflow.org/install/source) the TensorFlow pip
    package from source.

## Continuous build status

You can find more community-supported platforms and configurations in the
[TensorFlow SIG Build community builds table](https://github.com/tensorflow/build#community-supported-tensorflow-builds).

### Official Builds

Build Type                    | Status                                                                                                                                                                           | Artifacts
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Linux CPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html)           | [PyPI](https://pypi.org/project/tf-nightly/)
**Linux GPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html) | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Linux XLA**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html)         | TBA
**macOS**                     | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html)     | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows CPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html)       | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows GPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html)       | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Android**                   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html)               | [Download](https://bintray.com/google/tensorflow/tensorflow/_latestVersion)
**Raspberry Pi 0 and 1**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html)           | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl)
**Raspberry Pi 2 and 3**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html)           | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl)
**Libtensorflow MacOS CPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/macos/latest/macos_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux CPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/cpu/ubuntu_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux GPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/gpu/ubuntu_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows CPU** | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/cpu/windows_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows GPU** | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/gpu/windows_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)

## Resources

*   [TensorFlow.org](https://www.tensorflow.org)
*   [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)
*   [TensorFlow Examples](https://github.com/tensorflow/examples)
*   [TensorFlow Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)
*   [TensorFlow Blog](https://blog.tensorflow.org)
*   [Learn ML with TensorFlow](https://www.tensorflow.org/resources/learn-ml)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow model optimization roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap)
*   [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
*   [TensorBoard Visualization Toolkit](https://github.com/tensorflow/tensorboard)
*   [TensorFlow Code Search](https://cs.opensource.google/tensorflow/tensorflow)

Learn more about the
[TensorFlow community](https://www.tensorflow.org/community) and how to
[contribute](https://www.tensorflow.org/community/contribute).

## Courses

* [Coursera](https://www.coursera.org/search?query=TensorFlow)
* [Udacity](https://www.udacity.com/courses/all?search=TensorFlow)
* [Edx](https://www.edx.org/search?q=TensorFlow)

## License

[Apache License 2.0](LICENSE)
