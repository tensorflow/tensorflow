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

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) is an end-to-end, open-source platform for
machine learning that enables researchers and developers to build, train, and
deploy ML-powered applications at scale. It offers a comprehensive and flexible
ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that support both
cutting-edge research and production-ready systems.

Originally developed by researchers and engineers on the **Machine Intelligence**
team at **Google Brain**, TensorFlow was designed to advance research in machine
learning and neural networks. Over time, it has evolved into a highly versatile
framework that is widely used across academia, industry, and many other domains
beyond research.

TensorFlow provides stable and well-supported APIs for
[Python](https://www.tensorflow.org/api_docs/python) and
[C++](https://www.tensorflow.org/api_docs/cc), making it suitable for both rapid
prototyping and high-performance production use. In addition, it offers APIs for
[other languages](https://www.tensorflow.org/api_docs), which are available with
non-guaranteed backward compatibility.

To stay informed about new releases, important announcements, and security
updates, subscribe to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
You can also explore the full list of TensorFlow
[mailing lists and community forums](https://www.tensorflow.org/community/forums)
to engage with the broader ecosystem.


## Installing TensorFlow

TensorFlow provides multiple installation options depending on your platform,
hardware, and use case. For complete and up-to-date instructions, refer to the
official [TensorFlow installation guide](https://www.tensorflow.org/install), which
covers:

- Installing via the [pip package](https://www.tensorflow.org/install/pip)
- [Enabling GPU support](https://www.tensorflow.org/install/gpu) for accelerated training
- Running TensorFlow using a [Docker container](https://www.tensorflow.org/install/docker)
- [Building TensorFlow from source](https://www.tensorflow.org/install/source) for advanced use cases

---

### Standard Installation (CPU and GPU Support)

To install the **current stable release of TensorFlow**, which includes support
for [CUDA-enabled GPU cards](https://www.tensorflow.org/install/gpu) on **Ubuntu**
and **Windows**, run the following command:

```
pip install tensorflow
```


### Support for Other Devices

TensorFlow also supports additional hardware backends such as DirectX and
Apple Metal (macOS) through
[Device Plugins](https://www.tensorflow.org/install/gpu_plugins#available_devices).
These plugins allow TensorFlow to leverage platform-specific acceleration where
available.


### CPU-only Installation

If you do not require GPU acceleration, you can install a smaller, CPU-only
package:

```
$ pip install tensorflow-cpu
```


### Updating TensorFlow

To upgrade TensorFlow to the latest version, add the `--upgrade` flag to the
installation command:

```
pip install --upgrade tensorflow
```

(or replace tensorflow with tensorflow-cpu if using the CPU-only package)


### Nightly Builds (For Testing and Development)

For users who want to test upcoming features or contribute to TensorFlow
development, nightly builds are available on PyPI:
*Nightly binaries are available for testing using the
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) packages on PyPI.*


### *Try your first TensorFlow program*

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
[TensorFlow Tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the
[Contribution Guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub Issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, please see
[TensorFlow Forum](https://discuss.tensorflow.org/) for general questions and
discussion, and please direct specific questions to
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in
open-source software development.

## Patching guidelines

Follow these steps to patch a specific version of TensorFlow, for example, to
apply fixes to bugs or security vulnerabilities:

*   Clone the TensorFlow repository and switch to the appropriate branch for
    your desired version—for example, `r2.8` for version 2.8.
*   Apply the desired changes (i.e., cherry-pick them) and resolve any code
    conflicts.
*   Run TensorFlow tests and ensure they pass.
*   [Build](https://www.tensorflow.org/install/source) the TensorFlow pip
    package from source.

## Continuous build status

You can find more community-supported platforms and configurations in the
[TensorFlow SIG Build Community Builds Table](https://github.com/tensorflow/build#community-supported-tensorflow-builds).

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
[TensorFlow Community](https://www.tensorflow.org/community) and how to
[Contribute](https://www.tensorflow.org/community/contribute).

## Courses

* [Coursera](https://www.coursera.org/search?query=TensorFlow)
* [Udacity](https://www.udacity.com/courses/all?search=TensorFlow)
* [Edx](https://www.edx.org/search?q=TensorFlow)

## License

[Apache License 2.0](LICENSE)
