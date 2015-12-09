# Download and Setup

You can install TensorFlow either from our provided binary packages or from the
github source.

## Requirements

The TensorFlow Python API currently supports Python 2.7 and Python 3.3+ from
source.  We are preparing Python 3 pip packages to go with the 0.6.0 release.

The GPU version (Linux only) currently requires the Cuda Toolkit 7.0 and CUDNN
6.5 V2.  Please see [Cuda installation](#install_cuda).

## Overview

We support different ways to install TensorFlow:

*  [Pip install](#pip_install): Install TensorFlow on your machine, possibly
   upgrading previously installed Python packages.  May impact existing
   Python programs on your machine.
*  [Virtualenv install](#virtualenv_install): Install TensorFlow in its own
   directory, not impacting any existing Python programs on your machine.
*  [Docker install](#docker_install): Run TensorFlow in a Docker container
   isolated from all other programs on your machine.

If you are familiar with Pip, Virtualenv, or Docker, please feel free to adapt
the instructions to your particular needs.  The names of the pip and Docker
images are listed in the corresponding installation sections.

If you encounter installation errors, see
[common problems](#common_install_problems) for some solutions.

## Pip Installation {#pip_install}

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) is a package
management system used to install and manage software packages written in
Python.

The packages that will be installed or upgraded during the pip install are listed in the
[REQUIRED_PACKAGES section of setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)

Install pip if it is not already installed:

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev

# Mac OS X
$ sudo easy_install pip
```

Install TensorFlow:

```bash
# Ubuntu/Linux 64-bit, CPU only:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

You can now [test your installation](#test_install).

## Virtualenv installation {#virtualenv_install}

[Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) is a tool
to keep the dependencies required by different Python projects in separate
places.  The Virtualenv installation of TensorFlow will not override
pre-existing version of the Python packages needed by TensorFlow.

With [Virtualenv](https://pypi.python.org/pypi/virtualenv) the installation is
as follows:

*  Install pip and Virtualenv.
*  Create a Virtualenv environment.
*  Activate the Virtualenv environment and install TensorFlow in it.
*  After the install you will activate the Virtualenv environment each time you
   want to use TensorFlow.

Install pip and Virtualenv:

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv

# Mac OS X
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```

Create a Virtualenv environment in the directory `~/tensorflow`:

```bash
$ virtualenv --system-site-packages ~/tensorflow
```

Activate the environment and use pip to install TensorFlow inside it:

```bash
$ source ~/tensorflow/bin/activate  # If using bash
$ source ~/tensorflow/bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change

# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled:
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

With the Virtualenv environment activated, you can now
[test your installation](#test_install).

When you are done using TensorFlow, deactivate the environment.

```bash
(tensorflow)$ deactivate

$  # Your prompt should change back
```

To use TensorFlow later you will have to activate the Virtualenv environment again:

```bash
$ source ~/tensorflow/bin/activate  # If using bash.
$ source ~/tensorflow/bin/activate.csh  # If using csh.
(tensorflow)$  # Your prompt should change.
# Run Python programs that use TensorFlow.
...
# When you are done using TensorFlow, deactivate the environment.
(tensorflow)$ deactivate
```

## Docker installation {#docker_install}

[Docker](http://docker.com/) is a system to build self contained versions of a
Linux operating system running on your machine.  When you install and run
TensorFlow via Docker it completely isolates the installation from pre-existing
packages on your machine.

We provide 2 Docker images:

*  `b.gcr.io/tensorflow/tensorflow`: TensorFlow CPU binary image.
*  `b.gcr.io/tensorflow/tensorflow-full`: CPU Binary image plus source code.

With Docker the installation is as follows:

*  Install Docker on your machine.
*  Create a [Docker
group](http://docs.docker.com/engine/installation/ubuntulinux/#create-a-docker-group)
to allow launching containers without `sudo`.
*  Launch a Docker container with the TensorFlow image.  The image
   gets downloaded automatically on first launch.

See [installing Docker](http://docs.docker.com/engine/installation/) for instructions
on installing Docker on your machine.

After Docker is installed, launch a Docker container with the TensorFlow binary
image as follows.

```bash
$ docker run -it b.gcr.io/tensorflow/tensorflow
```

You can now [test your installation](#test_install) within the Docker container.

You can alternatively launch the TensorFlow source image, for example if you want
to experiment directly with the source.

```bash
$ docker run -it b.gcr.io/tensorflow/tensorflow-full
```

## Test the TensorFlow installation {#test_install}

### (Optional, Linux) Enable GPU Support

If you installed the GPU version of TensorFlow, you must also install the Cuda
Toolkit 7.0 and CUDNN 6.5 V2.  Please see [Cuda installation](#install_cuda).

You also need to set the `LD_LIBRARY_PATH` and `CUDA_HOME` environment
variables.  Consider adding the commands below to your `~/.bash_profile`.  These
assume your CUDA installation is in `/usr/local/cuda`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

### Run TensorFlow from the Command Line

See [common problems](#common_install_problems) if an error happens.

Open a terminal and type the following:

```bash
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print sess.run(a + b)
42
>>>
```

### Run a TensorFlow demo model

All TensorFlow packages, including the demo models, are installed in the Python library.
The exact location of the Python library depends on your system, but is usually one of:

```bash
/usr/local/lib/python2.7/dist-packages/tensorflow
/usr/local/lib/python2.7/site-packages/tensorflow
```

You can find out the directory with the following command:

```bash
$ python -c 'import site; print "\n".join(site.getsitepackages())'
```

The simple demo model for classifying handwritten digits from the MNIST dataset
is in the sub-directory `models/image/mnist/convolutional.py`.  You can run it from the command
line as follows:

```bash
# Using 'python -m' to find the program in the python search path:
$ python -m tensorflow.models.image.mnist.convolutional
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
...etc...

# You can alternatively pass the path to the model program file to the python interpreter.
$ python /usr/local/lib/python2.7/dist-packages/tensorflow/models/image/mnist/convolutional.py
...
```

## Installing from sources {#source}

When installing from source you will build a pip wheel that you then install
using pip. You'll need pip for that, so install it as described
[above](#pip_install).

### Clone the TensorFlow repository

```bash
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
```

`--recurse-submodules` is required to fetch the protobuf library that TensorFlow
depends on.

### Installation for Linux

#### Install Bazel

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for Bazel. Then download bazel version 0.1.1 using the
[installer for your system](https://github.com/bazelbuild/bazel/releases) and
run the installer as mentioned there:

```bash
$ chmod +x PATH_TO_INSTALL.SH
$ ./PATH_TO_INSTALL.SH --user
```

Remember to replace `PATH_TO_INSTALL.SH` with the location where you
downloaded the installer.

Finally, follow the instructions in that script to place `bazel` into your
binary path.

#### Install other dependencies

```bash
$ sudo apt-get install python-numpy swig python-dev
```

#### Configure the installation {#configure}

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter and allows (optional)
configuration of the CUDA libraries (see [below](#configure_cuda)).

This step is used to locate the python and numpy header files.

```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
```

#### Optional: Install CUDA (GPUs on Linux) {#install_cuda}

In order to build or run TensorFlow with GPU support, both Cuda Toolkit 7.0 and
CUDNN 6.5 V2 from NVIDIA need to be installed.

TensorFlow GPU support requires having a GPU card with NVidia Compute Capability >= 3.5.
Supported cards include but are not limited to:

* NVidia Titan
* NVidia Titan X
* NVidia K20
* NVidia K40

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

##### Configure TensorFlow's canonical view of Cuda libraries {#configure_cuda}
When running the `configure` script from the root of your source tree, select
the option `Y` when asked to build TensorFlow with GPU support.

``` bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow

Please specify the location where CUDA 7.0 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Please specify the location where CUDNN 6.5 V2 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

This creates a canonical set of symbolic links to the Cuda libraries on your system.
Every time you change the Cuda library paths you need to run this step again before
you invoke the bazel build command.

##### Build your target with GPU support
From the root of your source tree, run:

```bash
$ bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
# Lots of output. This tutorial iteratively calculates the major eigenvalue of
# a 2x2 matrix, on GPU. The last few lines look like this.
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```

Note that "--config=cuda" is needed to enable the GPU support.

##### Enabling Cuda 3.0
TensorFlow officially supports Cuda devices with 3.5 and 5.2 compute
capabilities. In order to enable earlier Cuda devices such as Grid K520, you
need to target Cuda 3.0. This can be done through TensorFlow unofficial
settings with "configure".

```bash
$ TF_UNOFFICIAL_SETTING=1 ./configure

# Same as the official settings above

WARNING: You are configuring unofficial settings in TensorFlow. Because some
external libraries are not backward compatible, these settings are largely
untested and unsupported.

Please specify a list of comma-separated Cuda compute capabilities you want to
build with. You can find the compute capability of your device at:
https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases
your build time and binary size. [Default is: "3.5,5.2"]: 3.0

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

##### Known issues

* Although it is possible to build both Cuda and non-Cuda configs under the same
source tree, we recommend to run "bazel clean" when switching between these two
configs in the same source tree.

* You have to run configure before running bazel build. Otherwise, the build
will fail with a clear error message. In the future, we might consider making
this more conveninent by including the configure step in our build process,
given necessary bazel new feature support.

### Installation for Mac OS X

We recommend using [homebrew](http://brew.sh) to install the bazel and SWIG
dependencies, and installing python dependencies using easy_install or pip.

#### Dependencies

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for Bazel. You can then use homebrew to install bazel and SWIG:

```bash
$ brew install bazel swig
```

You can install the python dependencies using easy_install or pip. Using
easy_install, run

```bash
$ sudo easy_install -U six
$ sudo easy_install -U numpy
$ sudo easy_install wheel
```

We also recommend the [ipython](https://ipython.org) enhanced python shell, so
best install that too:

```bash
$ sudo easy_install ipython
```

#### Configure the installation {#configure_osx}

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter.

This step is used to locate the python and numpy header files.

```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N]
```

### Create the pip package and install {#create-pip}

```bash
$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# The name of the .whl file will depend on your platform.
$ pip install /tmp/tensorflow_pkg/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

## Train your first TensorFlow neural net model

Starting from the root of your source tree, run:

```python
$ cd tensorflow/models/image/mnist
$ python convolutional.py
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

## Common Problems {#common_install_problems}

### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the the GPU installation [instructions](#install_cuda).

### Pip installation issues

#### Can't find setup.py

If, during `pip install`, you encounter an error like:

```bash
...
IOError: [Errno 2] No such file or directory: '/tmp/pip-o6Tpui-build/setup.py'
```

Solution: upgrade your version of pip:

```bash
pip install --upgrade pip
```

This may require `sudo`, depending on how pip is installed.

#### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.

### Linux issues

If you encounter:

```python
...
 "__add__", "__radd__",
             ^
SyntaxError: invalid syntax
```

Solution: make sure you are using Python 2.7.

### Mac OS X: ImportError: No module named copyreg

On Mac OS X, you may encounter the following when importing tensorflow.

```python
>>> import tensorflow as tf
...
ImportError: No module named copyreg
```

Solution: TensorFlow depends on protobuf, which requires the Python package
`six-1.10.0`. Apple's default Python installation only provides `six-1.4.1`.

You can resolve the issue in one of the following ways:

* Upgrade the Python installation with the current version of `six`:

```bash
$ sudo easy_install -U six
```

* Install TensorFlow with a separate Python library:

    *  Using [Virtualenv](#virtualenv_install).
    *  Using [Docker](#docker_install).

* Install a separate copy of Python via [Homebrew](http://brew.sh/) or
[MacPorts](https://www.macports.org/) and re-install TensorFlow in that
copy of Python.

### Mac OS X: TypeError: `__init__()` got an unexpected keyword argument 'syntax'

On Mac OS X, you may encounter the following when importing tensorflow.

```
>>> import tensorflow as tf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py", line 4, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python2.7/site-packages/tensorflow/python/__init__.py", line 13, in <module>
    from tensorflow.core.framework.graph_pb2 import *
...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py", line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02 \x03(\x0b\x32 .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
TypeError: __init__() got an unexpected keyword argument 'syntax'
```

This is due to a conflict between protobuf versions (we require protobuf 3.0.0).
The best current solution is to make sure older versions of protobuf are not
installed, such as:

```bash
$ pip install --upgrade protobuf
```
