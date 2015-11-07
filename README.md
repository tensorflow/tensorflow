#TensorFlow

TensorFlow is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.


**Note: Currently we do not accept pull requests on github -- see
[CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute code
changes to TensorFlow through
[tensorflow.googlesource.com](https://tensorflow.googlesource.com/tensorflow)**

**We use [github issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](tensorflow/g3doc/resources/index.md#community) for general questions
and discussion.**

# Download and Setup

To install TensorFlow using a binary package, see the instructions below.  For
more detailed installation instructions, including installing from source, see
[here](tensorflow/g3doc/get_started/os_setup.md).

## Binary Installation

### Ubuntu/Linux

Make sure you have [pip](https://pypi.python.org/pypi/pip) installed:

```sh
$ sudo apt-get install python-pip
```

Install TensorFlow:

```sh
# For CPU-only version
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version.  See detailed install instructions
# for GPU configuration information.
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

### Mac OS X

Make sure you have [pip](https://pypi.python.org/pypi/pip) installed:

If using `easy_install`:

```sh
$ sudo easy_install pip
```

Install TensorFlow (only CPU binary version is currently available).

```sh
$ sudo pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

### Try your first TensorFlow program

```sh
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print sess.run(a+b)
42
>>>

```


##For more information

* [TensorFlow website](http://tensorflow.org)
