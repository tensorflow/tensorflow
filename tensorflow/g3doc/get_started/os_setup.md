# Download and Setup

You can install TensorFlow using our provided binary packages or from source.

## Binary Installation

The TensorFlow Python API currently requires Python 2.7: we are
[working](https://github.com/tensorflow/tensorflow/issues/1) on adding support
for Python 3.

The simplest way to install TensorFlow is using
[pip](https://pypi.python.org/pypi/pip) for both Linux and Mac.

If you encounter installation errors, see
[common problems](#common_install_problems) for some solutions. To simplify
installation, please consider using our virtualenv-based instructions
[here](#virtualenv_install).

### Ubuntu/Linux 64-bit

```bash
# For CPU-only version
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version (only install this version if you have the CUDA sdk installed)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

### Mac OS X

On OS X, we recommend installing [homebrew](http://brew.sh) and `brew install
python` before proceeding, or installing TensorFlow within [virtualenv](#virtualenv_install).

```bash
# Only CPU-version is available at the moment.
$ pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

## Docker-based installation

We also support running TensorFlow via [Docker](http://docker.com/), which lets
you avoid worrying about setting up dependencies.

First, [install Docker](http://docs.docker.com/engine/installation/). Once
Docker is up and running, you can start a container with one command:

```bash
$ docker run -it b.gcr.io/tensorflow/tensorflow
```

This will start a container with TensorFlow and all its dependencies already
installed.

### Additional images

The default Docker image above contains just a minimal set of libraries for
getting up and running with TensorFlow. We also have the following container,
which you can use in the `docker run` command above:

* `b.gcr.io/tensorflow/tensorflow-full`: Contains a complete TensorFlow source
  installation, including all utilities needed to build and run TensorFlow. This
  makes it easy to experiment directly with the source, without needing to
  install any of the dependencies described above.

## VirtualEnv-based installation {#virtualenv_install}

We recommend using [virtualenv](https://pypi.python.org/pypi/virtualenv) to
create an isolated container and install TensorFlow in that container -- it is
optional but makes verifying installation issues easier.

First, install all required tools:

```bash
# On Linux:
$ sudo apt-get install python-pip python-dev python-virtualenv

# On Mac:
$ sudo easy_install pip  # If pip is not already installed
$ sudo pip install --upgrade virtualenv
```

Next, set up a new virtualenv environment.  To set it up in the
directory `~/tensorflow`, run:

```bash
$ virtualenv --system-site-packages ~/tensorflow
$ cd ~/tensorflow
```

Then activate the virtualenv:

```bash
$ source bin/activate  # If using bash
$ source bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change
```

Inside the virtualenv, install TensorFlow:

```bash
# For CPU-only linux x86_64 version
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled linux x86_64 version
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For Mac CPU-only version
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

Make sure you have downloaded the source code for TensorFlow, and then you can
then run an example TensorFlow program like:

```bash
(tensorflow)$ cd tensorflow/models/image/mnist
(tensorflow)$ python convolutional.py

# When you are done using TensorFlow:
(tensorflow)$ deactivate  # Deactivate the virtualenv

$  # Your prompt should change back
```

## Try your first TensorFlow program

### (Optional) Enable GPU Support

If you installed the GPU-enabled TensorFlow pip binary, you must have the
correct versions of the CUDA SDK and CUDNN installed on your
system.  Please see [the CUDA installation instructions](#install_cuda).

You also need to set the `LD_LIBRARY_PATH` and `CUDA_HOME` environment
variables.  Consider adding the commands below to your `~/.bash_profile`.  These
assume your CUDA installation is in `/usr/local/cuda`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

### Run TensorFlow

Open a python terminal:

```bash
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

## Installing from sources {#source}

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

Remember to replace `PATH_TO_INSTALL.SH` to point to the location where you
downloaded the installer.

Finally, follow the instructions in that script to place bazel into your binary
path.

#### Install other dependencies

```bash
$ sudo apt-get install python-numpy swig python-dev
```

#### Optional: Install CUDA (GPUs on Linux) {#install_cuda}

In order to build or run TensorFlow with GPU support, both Cuda Toolkit 7.0 and
CUDNN 6.5 V2 from NVIDIA need to be installed.

TensorFlow GPU support requires having a GPU card with NVidia Compute Capability >= 3.5.  Supported cards include but are not limited to:

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

##### Configure TensorFlow's canonical view of Cuda libraries
From the root of your source tree, run:

``` bash
$ ./configure
Do you wish to build TensorFlow with GPU support? [y/n] y
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

Mac needs the same set of dependencies as Linux, however installing those
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

Solution: upgrade your version of `pip`:

```bash
pip install --upgrade pip
```

This may require `sudo`, depending on how `pip` is installed.

#### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.

### On Linux

If you encounter:

```python
...
 "__add__", "__radd__",
             ^
SyntaxError: invalid syntax
```

Solution: make sure you are using Python 2.7.

### On MacOSX


If you encounter:

```python
import six.moves.copyreg as copyreg

ImportError: No module named copyreg
```

Solution: TensorFlow depends on protobuf, which requires `six-1.10.0`. Apple's
default python environment has `six-1.4.1` and may be difficult to upgrade.
There are several ways to fix this:

1. Upgrade the system-wide copy of `six`:

    ```bash
    sudo easy_install -U six
    ```

2. Install a separate copy of python via homebrew:

    ```bash
    brew install python
    ```

3. Build or use TensorFlow
   [within `virtualenv`](#virtualenv_install).



If you encounter:

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
brew reinstall --devel protobuf
```
