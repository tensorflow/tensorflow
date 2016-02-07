<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

|  **`Linux CPU`**   |  **`Linux GPU PIP`** | **`Mac OS CPU`** |  **`Android`** |
|-------------------|----------------------|------------------|----------------|
| [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master)](http://ci.tensorflow.org/job/tensorflow-master) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-gpu_pip)](http://ci.tensorflow.org/job/tensorflow-master-gpu_pip) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](http://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](http://ci.tensorflow.org/job/tensorflow-master-android) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

**If you'd like to contribute to tensorflow, be sure to review the [contribution
guidelines](CONTRIBUTING.md).**

**We use [github issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](tensorflow/g3doc/resources/index.md#community) for general questions
and discussion.**

## Installation
*See [Download and Setup](tensorflow/g3doc/get_started/os_setup.md).*

#### *Try your first TensorFlow program*
```python
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>>
```

##For more information
* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [Tensorflow MOOC on Udacity] (https://www.udacity.com/course/deep-learning--ud730)
