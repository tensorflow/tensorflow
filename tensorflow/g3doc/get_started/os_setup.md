# Download and Setup

## Binary Installation

### Ubuntu/Linux

Make sure you have `pip` and `numpy` installed :

```sh
$ sudo apt-get install python-pip python-numpy
```

Install TensorFlow:

```sh
# For CPU-only version
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

### Mac OS X

Make sure you have `pip` installed:

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

If you are running the GPU version and you see
```sh
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

you most likely need to set your `LD_LIBRARY_PATH` to point to the location of
your CUDA libraries.

### Train the MNIST neural net model

```sh
$ python tensorflow/models/image/mnist/convolutional.py
Succesfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Succesfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Succesfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Succesfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op
parallelism threads: 3
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op
parallelism threads: 4
Initialized!
Epoch 0.00
Minibatch loss: 12.054, learning rate: 0.010000
Minibatch error: 90.6%
Validation error: 84.6%
...
...

```

## Source Installation {#source}

### Clone the TensorFlow repository

TODO(keveman): Supply clone command for external users.

```sh
$ git clone --recurse-submodules https://YOUR_WHITELISTED_EMAIL_WITH_AT_REPLACED_BY_DOT@tensorflow.googlesource.com/tf3
```


### Installation for Linux

#### Install Bazel


Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for Bazel. Then download and build the Bazel source with the
following commands:

```sh
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ git checkout tags/0.1.0
$ ./compile.sh
```

These commands use the commit tag `0.1.0`, which is known to work with
TensorFlow. `HEAD` may be unstable.

Add the executable `output/bazel` to your `$PATH` environment variable.

#### Install other dependencies

```sh
$ sudo apt-get install python-numpy swig python-dev
```

#### Optional: Install CUDA (GPUs on Linux)

In order to build TensorFlow with GPU support, both Cuda Toolkit 7.0 and CUDNN
6.5 V2 from NVIDIA need to be installed.

##### Download and install Cuda Toolkit 7.0

https://developer.nvidia.com/cuda-toolkit-70

Install the toolkit into e.g. `/usr/local/cuda`

##### Download and install CUDNN Toolkit 6.5

https://developer.nvidia.com/rdp/cudnn-archive

Uncompress and copy the cudnn files into the toolkit directory.  Assuming the
toolkit is installed in `/usr/local/cuda`:

``` bash
tar xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
```

##### Configure TensorFlow's canonical view of Cuda libraries
From the root of your source tree, run:

``` bash
$ ./configure
Do you wish to bulid TensorFlow with GPU support? [y/n] y
GPU support will be enabled for TensorFlow

Please specify the location where CUDA 7.0 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda
CUDA 7.0 toolkit found

Please specify the location where CUDNN 6.5 V2 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda
CUDNN 6.5 V2 library found

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

This creates a canonical set of symbolic links to the Cuda libraries on your system.
Every time you change the Cuda library paths you need to run this step again before
you invoke the bazel build command.

##### Build your target with GPU support.
From the root of your source tree, run:

```sh
$ bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
# Lots of output. This tutorial iteratively calculates the major eigenvalue of
# a 2x2 matrix, on GPU. The last few lines look like this.
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000008 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000003 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000006 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```

Note that "--config=cuda" is needed to enable the GPU support.

##### Known issues

* Although it is possible to build both Cuda and non-Cuda configs under the same
source tree, we recommend to run "bazel clean" when switching between these two
configs in the same source tree.

* You have to run configure before running bazel build. Otherwise, the build
will fail with a clear error message. In the future, we might consider making
this more conveninent by including the configure step in our build process,
given necessary bazel new feature support.

### Installation for Mac OS X

Mac needs the same set of dependencies as Linux, however their installing those
dependencies is different. Here is a set of useful links to help with installing
the dependencies on Mac OS X :

#### Bazel

Look for installation instructions for Mac OS X on
[this](http://bazel.io/docs/install.html) page.

#### SWIG

[Mac OS X installation](http://www.swig.org/Doc3.0/Preface.html#Preface_osx_installation).

Notes : You need to install
[PCRE](ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/) and *NOT* PCRE2.

#### Numpy

Follow installation instructions [here](http://docs.scipy.org/doc/numpy/user/install.html).

### Build and train your first TensorFlow neural net model

```sh
$ cd tf3

$ bazel build tensorflow/models/image/mnist:convolutional

$ bazel-bin/tensorflow/models/image/mnist/convolutional
Succesfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Succesfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Succesfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Succesfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
Initialized!
Epoch 0.00
Minibatch loss: 12.054, learning rate: 0.010000
Minibatch error: 90.6%
Validation error: 84.6%
Epoch 0.12
Minibatch loss: 3.285, learning rate: 0.010000
Minibatch error: 6.2%
Validation error: 7.0%
...
...
```
