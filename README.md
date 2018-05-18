<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------


| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  The graph nodes represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture enables you to deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard), a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

Keep up to date with release announcements and security updates by
subscribing to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).

## Installation
*See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install our release binaries or how to build from source.*

People who are a little more adventurous can also try our nightly binaries:

**Nightly pip packages**
* We are pleased to announce that TensorFlow now offers nightly pip packages
under the [tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) project on pypi.
Simply run `pip install tf-nightly` or `pip install tf-nightly-gpu` in a clean
environment to install the nightly TensorFlow build. We support CPU and GPU
packages on Linux, Mac, and Windows.


#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs. So please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) for general questions
and discussion, and please direct specific questions to [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in open-source software development:

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)


## Continuous build status

### Official Builds

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **Linux CPU**   | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.png) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Linux GPU**   | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-cc.png) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Linux XLA**   | TBA | TBA |
| **MacOS**       | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.png) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows CPU** | [![Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows GPU** | [![Status](http://ci.tensorflow.org/job/tf-master-win-gpu-cmake/badge/icon)](http://ci.tensorflow.org/job/tf-master-win-gpu-cmake/) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Android**     | [![Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) | [![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg)](https://bintray.com/google/tensorflow/tensorflow/_latestVersion) [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/) [build history](https://ci.tensorflow.org/view/Nightly/job/nightly-android/) |


### Community Supported Builds

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **IBM s390x**       | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/) | TBA |
| **IBM ppc64le CPU** | [![Build Status](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/badge/icon)](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/) | TBA |


## For more information

* [TensorFlow Website](https://www.tensorflow.org)
* [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
* [TensorFlow YouTube Channel](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow Course at Stanford](https://web.stanford.edu/class/cs20si)

Learn more about the TensorFlow community at the [community page of tensorflow.org](https://www.tensorflow.org/community) for a few ways to participate.

## License

[Apache License 2.0](LICENSE)
