<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------

| **`Linux CPU`**                          | **`Linux GPU`**                          | **`Mac OS CPU`**                         | **`Windows CPU`**                        | **`Android`**                            |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](https://ci.tensorflow.org/job/tensorflow-master-cpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-linux-gpu)](https://ci.tensorflow.org/job/tensorflow-master-linux-gpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](https://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) |

**TensorFlow** 是一个使用数据流图进行数值计算开源软件库。
图的节点表示数学运算，节点之间的边表示流动的多维数据数组（张量）。
这种灵活的架构允许你在无需重写代码的情况下，将计算在桌面端、服务端或移动端部署到一个或多个 CPU 和 GPU 中。
TensorFlow 还包含 TensorBoard，它是一个数据可视化工具包。

TensorFlow 最初由 Google 机器智能研究机构内的 
Google Brain 团队的研究人员和工程师开发，用于进行机器学习和深度神经网络研究。
此系统一般足以适用于各种其他领域。

**如果你想参与贡献 TensorFlow，请先查看我们的 [贡献指南](CONTRIBUTING.md)。此项目遵循 TensorFlow
[项目规范](CODE_OF_CONDUCT.md)。我们期望你能遵循此规范。**

**我们还使用 [GitHub issues](https://github.com/tensorflow/tensorflow/issues) 来跟进 requests 和 bugs。对于一般性问题和讨论请查看 
[TensorFlow 讨论](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)，或直接在 [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow) 提问。**

## 安装
*在 [安装 TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) 页面中查看关于稳定二进制版的安装或从源码安装的安装步骤。*

喜欢挑战的人也可以尝试我们的开发版：

**开发版 pip 包**
* 我们非常高兴发布 TensorFlow 的开发版，现在 pypi 提供开发版的 pip 包 [tf-nightly](https://pypi.python.org/pypi/tf-nightly) 和
  [tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) 项目。在干净的环境中简单运行 `pip install tf-nightly` 或 `pip install tf-nightly-gpu` 即可安装 TensorFlow 开发版。 我们为 Linux、Mac 和 Windows 提供  CPU 和 GPU 支持。


**独立的 whl 文件**
* Linux CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp27-none-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp34-cp34m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp35-cp35m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/))
* Linux GPU: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/42/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp27-none-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp34-cp34m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp35-cp35m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/))
* Mac CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py2-none-any.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/)) / [Python 3](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py3-none-any.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/))
* Windows CPU-only: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp35-cp35m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp36-cp36m-win_amd64.whl) ([构建历史](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/))
* Windows GPU: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp35-cp35m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp36-cp36m-win_amd64.whl) ([构建历史](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/))
* Android: [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/)
  ([构建历史](https://ci.tensorflow.org/view/Nightly/job/nightly-android/))

#### *开启你的第一个 TensorFlow 程序*

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

## 更多信息

* [TensorFlow 网站](https://www.tensorflow.org)
* [TensorFlow 白皮书](https://www.tensorflow.org/about/bib)
* [TensorFlow 模型](https://github.com/tensorflow/models)
* [TensorFlow MOOC 教程](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow Stanford 教程](https://web.stanford.edu/class/cs20si)

你可以在 [tensorflow.org 社区页](https://www.tensorflow.org/community) 了解更多关于参与 TensorFlow 社区的方法。

## 许可

[Apache 许可 2.0](LICENSE)
