<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML-powered applications.

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization to
conduct machine learning and deep neural networks research. The system is
general enough to be applicable in a wide variety of other domains, as well.

TensorFlow provides stable [Python](https://www.tensorflow.org/api_docs/python)
and [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backward compatible API for
[other languages](https://www.tensorflow.org/api_docs).

Keep up-to-date with release announcements and security updates by subscribing
to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
See all the [mailing lists](https://www.tensorflow.org/community/forums).

## Install

See the [TensorFlow install guide](https://www.tensorflow.org/install) for the
[pip package](https://www.tensorflow.org/install/pip), to
[enable GPU support](https://www.tensorflow.org/install/gpu), use a
[Docker container](https://www.tensorflow.org/install/docker), and
[build from source](https://www.tensorflow.org/install/source).

To install the current release, which includes support for
[CUDA-enabled GPU cards](https://www.tensorflow.org/install/gpu) *(Ubuntu and
Windows)*:

```
$ pip install tensorflow
```

A smaller CPU-only package is also available:

```
$ pip install tensorflow-cpu
```

To update TensorFlow to the latest version, add `--upgrade` flag to the above
commands.

*Nightly binaries are available for testing using the
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) packages on PyPi.*

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
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
for general questions and discussion, and please direct specific questions to
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in
open-source software development:

[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

## Continuous build status

### Official Builds

Build Type                    | Status                                                                                                                                                                                             | Artifacts
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Linux CPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html)                             | [PyPI](https://pypi.org/project/tf-nightly/)
**Linux GPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html)                   | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Linux XLA**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html)                           | TBA
**macOS**                     | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html)                       | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows CPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html)                         | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows GPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html)                         | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Android**                   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html)                                 | [![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg)](https://bintray.com/google/tensorflow/tensorflow/_latestVersion)
**Raspberry Pi 0 and 1**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html)                             | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl)
**Raspberry Pi 2 and 3**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html)                             | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl)
**Libtensorflow MacOS CPU**   | Status Temporarily Unavailable | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/macos/latest/macos_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux CPU**   | Status Temporarily Unavailable | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/cpu/ubuntu_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux GPU**   | Status Temporarily Unavailable | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/gpu/ubuntu_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows CPU** | Status Temporarily Unavailable | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/cpu/windows_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows GPU** | Status Temporarily Unavailable | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/gpu/windows_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)

### Community Supported Builds

Build Type                                                                          | Status                                                                                                                                                                                                                                                                                                                                                                                              | Artifacts
----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Linux AMD ROCm GPU** Nightly                                                      | [![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly/badge/icon)](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly)                                                                                                                                                                                                                                                        | [Nightly](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly/lastSuccessfulBuild/)
**Linux AMD ROCm GPU** Stable Release                                               | [![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/)                                                                                                                                                                                                                                                       | Release [1.15](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/lastSuccessfulBuild/) / [2.x](http://ml-ci.amd.com:21096/job/tensorflow-rocm-v2-release/lastSuccessfulBuild/)
**Linux s390x** Nightly                                                             | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/)                                                                                                                                                                                                                                                                   | [Nightly](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/)
**Linux s390x CPU** Stable Release                                                  | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/badge/icon)](https://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/)                                                                                                                                                                                                                                            | [Release](https://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/)
**Linux ppc64le CPU** Nightly                                                       | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/)                                                                                                                                                                                                                                             | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Nightly_Artifact/)
**Linux ppc64le CPU** Stable Release                                                | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/)                                                                                                                                                                                                                             | Release [1.15](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/) / [2.x](https://powerci.osuosl.org/job/TensorFlow2_PPC64LE_CPU_Release_Build/)
**Linux ppc64le GPU** Nightly                                                       | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/)                                                                                                                                                                                                                                             | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Nightly_Artifact/)
**Linux ppc64le GPU** Stable Release                                                | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/)                                                                                                                                                                                                                             | Release [1.15](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/) / [2.x](https://powerci.osuosl.org/job/TensorFlow2_PPC64LE_GPU_Release_Build/)
**Linux aarch64 CPU** Nightly (Linaro)                                              | [![Build Status](https://ci.linaro.org/jenkins/buildStatus/icon?job=ldcg-python-tensorflow-nightly)](https://ci.linaro.org/jenkins/job/ldcg-python-tensorflow-nightly/)                                                                                                                                                                                                                             | [Nightly](http://snapshots.linaro.org/ldcg/python/tensorflow-nightly/latest/)
**Linux aarch64 CPU** Stable Release (Linaro)                                       | [![Build Status](https://ci.linaro.org/jenkins/buildStatus/icon?job=ldcg-python-tensorflow)](https://ci.linaro.org/jenkins/job/ldcg-python-tensorflow/)                                                                                                                                                                                                                                             | Release [1.x & 2.x](http://snapshots.linaro.org/ldcg/python/tensorflow/latest/)
**Linux aarch64 CPU** Nightly (OpenLab)<br> Python 3.6                              | [![Build Status](http://openlabtesting.org:15000/badge?project=tensorflow%2Ftensorflow)](https://status.openlabtesting.org/builds/builds?project=tensorflow%2Ftensorflow&job_name=tensorflow-arm64-build-daily-master)                                                                                                                                                                              | [Nightly](https://status.openlabtesting.org/builds/builds?project=tensorflow%2Ftensorflow&job_name=tensorflow-arm64-build-daily-master)
**Linux aarch64 CPU** Stable Release (OpenLab)                                      | [![Build Status](http://openlabtesting.org:15000/badge?project=tensorflow%2Ftensorflow&job_name=tensorflow-v1.15.3-cpu-arm64-release-build-show&job_name=tensorflow-v2.1.0-cpu-arm64-release-build-show)](http://status.openlabtesting.org/builds?project=tensorflow%2Ftensorflow&job_name=tensorflow-v2.1.0-cpu-arm64-release-build-show&job_name=tensorflow-v1.15.3-cpu-arm64-release-build-show) | Release [1.15](http://status.openlabtesting.org/builds?project=tensorflow%2Ftensorflow&job_name=tensorflow-v1.15.3-cpu-arm64-release-build-show) / [2.x](http://status.openlabtesting.org/builds?project=tensorflow%2Ftensorflow&job_name=tensorflow-v2.1.0-cpu-arm64-release-build-show)
**Linux CPU with Intel oneAPI Deep Neural Network Library (oneDNN)** Nightly        | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/)                                                                                                                                                                                                                           | [Nightly](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/)
**Linux CPU with Intel oneAPI Deep Neural Network Library (oneDNN)** Stable Release | ![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/badge/icon)                                                                                                                                                                                                                                                                                                    | Release [1.15](https://pypi.org/project/intel-tensorflow/1.15.0/) / [2.x](https://pypi.org/project/intel-tensorflow/)
**Red Hat® Enterprise Linux® 7.6 CPU & GPU** <br> Python 2.7, 3.6                   | [![Build Status](https://jenkins-tensorflow.apps.ci.centos.org/buildStatus/icon?job=tensorflow-rhel7-3.6&build=2)](https://jenkins-tensorflow.apps.ci.centos.org/job/tensorflow-rhel7-3.6/2/)                                                                                                                                                                                                       | [1.13.1 PyPI](https://tensorflow.pypi.thoth-station.ninja/index/)

### Community Supported Containers

Container Type                                                    | Status | Artifacts
----------------------------------------------------------------- | ------ | ---------
**TensorFlow aarch64 Neoverse-N1 CPU** Stable (Linaro)<br> Debian | Static | Release [2.3](https://hub.docker.com/r/linaro/tensorflow-arm-neoverse-n1)

## Resources

*   [TensorFlow.org](https://www.tensorflow.org)
*   [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)
*   [TensorFlow Examples](https://github.com/tensorflow/examples)
*   [DeepLearning.AI TensorFlow Developer Professional Certificate](https://www.coursera.org/specializations/tensorflow-in-practice)
*   [TensorFlow: Data and Deployment from Coursera](https://www.coursera.org/specializations/tensorflow-data-and-deployment)
*   [Getting Started with TensorFlow 2 from Coursera](https://www.coursera.org/learn/getting-started-with-tensor-flow2)
*   [TensorFlow: Advanced Techniques from Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques)
*   [Intro to TensorFlow for A.I, M.L, and D.L from Coursera](https://www.coursera.org/learn/introduction-tensorflow)
*   [Intro to TensorFlow for Deep Learning from Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
*   [Introduction to TensorFlow Lite from Udacity](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)
*   [Machine Learning with TensorFlow on GCP](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)
*   [TensorFlow Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)
*   [TensorFlow Blog](https://blog.tensorflow.org)
*   [Learn ML with TensorFlow](https://www.tensorflow.org/resources/learn-ml)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow Roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap)
*   [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
*   [TensorBoard Visualization Toolkit](https://github.com/tensorflow/tensorboard)

Learn more about the
[TensorFlow community](https://www.tensorflow.org/community) and how to
[contribute](https://www.tensorflow.org/community/contribute).

## License

[Apache License 2.0](LICENSE)
