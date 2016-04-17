# Download and Setup

You can install TensorFlow either from our provided binary packages or from the
github source.

## Requirements

The TensorFlow Python API supports Python 2.7 and Python 3.3+.

The GPU version (Linux only) requires the Cuda Toolkit >= 7.0 and cuDNN >=
v2.  Please see [Cuda installation](#optional-install-cuda-gpus-on-linux)
for details.

## Overview

We support different ways to install TensorFlow:

*  [Pip install](#pip-installation): Install TensorFlow on your machine, possibly
   upgrading previously installed Python packages.  May impact existing
   Python programs on your machine.
*  [Virtualenv install](#virtualenv-installation): Install TensorFlow in its own
   directory, not impacting any existing Python programs on your machine.
*  [Anaconda install](#anaconda-installation): Install TensorFlow in its own
   environment for those running the Anaconda Python distribution.  Does not
   impact existing Python programs on your machine.
*  [Docker install](#docker-installation): Run TensorFlow in a Docker container
   isolated from all other programs on your machine.

If you are familiar with Pip, Virtualenv, Anaconda, or Docker, please feel free to adapt
the instructions to your particular needs.  The names of the pip and Docker
images are listed in the corresponding installation sections.

If you encounter installation errors, see
[common problems](#common-problems) for some solutions.

## Pip Installation

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) is a package
management system used to install and manage software packages written in
Python.

The packages that will be installed or upgraded during the pip install are listed in the
[REQUIRED_PACKAGES section of setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)

Install pip (or pip3 for python3) if it is not already installed:

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev

# Mac OS X
$ sudo easy_install pip
```

Install TensorFlow:

```bash
# Ubuntu/Linux 64-bit, CPU only:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl
```

For python3:

```bash
# Ubuntu/Linux 64-bit, CPU only:
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py3-none-any.whl
```

NOTE: If you are upgrading from a previous installation of TensorFlow < 0.7.1,
you should uninstall the previous TensorFlow *and protobuf* using `pip
uninstall` first to make sure you get a clean installation of the updated
protobuf dependency.


You can now [test your installation](#test-the-tensorflow-installation).

## Virtualenv installation

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
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl
```

and again for python3:

```bash
$ source ~/tensorflow/bin/activate  # If using bash
$ source ~/tensorflow/bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change

# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
(tensorflow)$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only:
(tensorflow)$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py3-none-any.whl
```

With the Virtualenv environment activated, you can now
[test your installation](#test-the-tensorflow-installation).

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

## Anaconda environment installation

[Anaconda](https://www.continuum.io/why-anaconda) is a Python distribution that
includes a large number of standard numeric and scientific computing packages.
Anaconda uses a package manager called "conda" that has its own 
[environment system](http://conda.pydata.org/docs/using/envs.html) similar to Virtualenv.

As with Virtualenv, conda environments keep the dependencies required by
different Python projects in separate places.  The Anaconda environment
installation of TensorFlow will not override pre-existing version of the Python
packages needed by TensorFlow.

*  Install Anaconda.
*  Create a conda environment.
*  Activate the conda environment and install TensorFlow in it.
*  After the install you will activate the conda environment each time you
   want to use TensorFlow.

Install Anaconda:

Follow the instructions on the [Anaconda download site](https://www.continuum.io/downloads)

Create a conda environment called `tensorflow`:

```bash
# Python 2.7
$ conda create -n tensorflow python=2.7

# Python 3.5
$ conda create -n tensorflow python=3.5
```

Activate the environment and use pip to install TensorFlow inside it.
Use the `--ignore-installed` flag to prevent errors about `easy_install`.

```bash
$ source activate tensorflow
(tensorflow)$  # Your prompt should change

# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl
```

and again for Python 3:

```bash
$ source activate tensorflow
(tensorflow)$  # Your prompt should change

# Ubuntu/Linux 64-bit, CPU only:
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only:
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py3-none-any.whl
```

With the conda environment activated, you can now
[test your installation](#test-the-tensorflow-installation).

When you are done using TensorFlow, deactivate the environment.

```bash
(tensorflow)$ source deactivate

$  # Your prompt should change back
```

To use TensorFlow later you will have to activate the conda environment again:

```bash
$ source activate tensorflow
(tensorflow)$  # Your prompt should change.
# Run Python programs that use TensorFlow.
...
# When you are done using TensorFlow, deactivate the environment.
(tensorflow)$ source deactivate
```

## Docker installation

[Docker](http://docker.com/) is a system to build self contained versions of a
Linux operating system running on your machine.  When you install and run
TensorFlow via Docker it completely isolates the installation from pre-existing
packages on your machine.

We provide 4 Docker images:

* `gcr.io/tensorflow/tensorflow`: TensorFlow CPU binary image.
* `gcr.io/tensorflow/tensorflow:latest-devel`: CPU Binary image plus source
code.
* `gcr.io/tensorflow/tensorflow:latest-gpu`: TensorFlow GPU binary image.
* `gcr.io/tensorflow/tensorflow:latest-devel-gpu`: GPU Binary image plus source
code.

We also have tags with `latest` replaced by a released version (e.g., `0.8.0rc0-gpu`).

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
$ docker run -it gcr.io/tensorflow/tensorflow
```

If you're using a container with GPU support, some additional flags must be
passed to expose the GPU device to the container. For the default config, we
include a
[script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/docker_run_gpu.sh)
in the repo with these flags, so the command-line would look like

```bash
$ path/to/repo/tensorflow/tools/docker/docker_run_gpu.sh gcr.io/tensorflow/tensorflow:gpu
```

You can now [test your installation](#test-the-tensorflow-installation) within the Docker container.

## Test the TensorFlow installation

### (Optional, Linux) Enable GPU Support

If you installed the GPU version of TensorFlow, you must also install the Cuda
Toolkit 7.0 and cuDNN v2.  Please see [Cuda installation](#optional-install-cuda-gpus-on-linux).

You also need to set the `LD_LIBRARY_PATH` and `CUDA_HOME` environment
variables.  Consider adding the commands below to your `~/.bash_profile`.  These
assume your CUDA installation is in `/usr/local/cuda`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

### Run TensorFlow from the Command Line

See [common problems](#common-problems) if an error happens.

Open a terminal and type the following:

```bash
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
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

You can find out the directory with the following command (make sure to use the Python you installed TensorFlow to, for example, use `python3` instead of `python` if you installed for Python 3):

```bash
$ python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```

The simple demo model for classifying handwritten digits from the MNIST dataset
is in the sub-directory `models/image/mnist/convolutional.py`.  You can run it from the command
line as follows (make sure to use the Python you installed TensorFlow with):

```bash
# Using 'python -m' to find the program in the python search path:
$ python -m tensorflow.models.image.mnist.convolutional
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
...etc...

# You can alternatively pass the path to the model program file to the python
# interpreter (make sure to use the python distribution you installed
# TensorFlow to, for example, .../python3.X/... for Python 3).
$ python /usr/local/lib/python2.7/dist-packages/tensorflow/models/image/mnist/convolutional.py
...
```

## Installing from sources

When installing from source you will build a pip wheel that you then install
using pip. You'll need pip for that, so install it as described
[above](#pip-installation).

### Clone the TensorFlow repository

```bash
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
```

`--recurse-submodules` is required to fetch the protobuf library that TensorFlow
depends on. Note that these instructions will install the latest master branch
of tensorflow. If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command.

### Installation for Linux

#### Install Bazel

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for bazel. Then download the latest stable bazel version using the
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

#### Configure the installation

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter and allows (optional)
configuration of the CUDA libraries (see [below](#configure-tensorflows-canonical-view-of-cuda-libraries)).

This step is used to locate the python and numpy header files.

```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
```

#### Optional: Install CUDA (GPUs on Linux)

In order to build or run TensorFlow with GPU support, both NVIDIA's Cuda Toolkit (>= 7.0) and
cuDNN (>= v2) need to be installed.

TensorFlow GPU support requires having a GPU card with NVidia Compute Capability >= 3.0.
Supported cards include but are not limited to:

* NVidia Titan
* NVidia Titan X
* NVidia K20
* NVidia K40

##### Download and install Cuda Toolkit

https://developer.nvidia.com/cuda-downloads

Install the toolkit into e.g. `/usr/local/cuda`

##### Download and install cuDNN

https://developer.nvidia.com/cudnn

Uncompress and copy the cuDNN files into the toolkit directory.  Assuming the
toolkit is installed in `/usr/local/cuda`, run the following commands (edited
to reflect the cuDNN version you downloaded):

``` bash
tar xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

##### Configure TensorFlow's canonical view of Cuda libraries

When running the `configure` script from the root of your source tree, select
the option `Y` when asked to build TensorFlow with GPU support. If you have 
several versions of Cuda or cuDNN installed, you should definitely select
one explicitly instead of relying on the system default. You should see
prompts like the following:

``` bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow

Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave
empty to use system default]: 7.5

Please specify the location where CUDA 7.5 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Please specify the Cudnn version you want to use. [Leave empty to use system
default]: 4.0.4

Please specify the location where the cuDNN 4.0.4 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cudnn-r4-rc/

Please specify a list of comma-separated Cuda compute capabilities you want to
build with. You can find the compute capability of your device at: 
https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your
build time and binary size. [Default is: \"3.5,5.2\"]: 3.5
    
Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

This creates a canonical set of symbolic links to the Cuda libraries on your system.
Every time you change the Cuda library paths you need to run this step again before
you invoke the bazel build command. For the Cudnn libraries, use '6.5' for R2, '7.0'
for R3, and '4.0.4' for R4-RC.


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

##### Known issues

* Although it is possible to build both Cuda and non-Cuda configs under the same
source tree, we recommend to run `bazel clean` when switching between these two
configs in the same source tree.

* You have to run configure before running bazel build. Otherwise, the build
will fail with a clear error message. In the future, we might consider making
this more convenient by including the configure step in our build process.

### Installation for Mac OS X

We recommend using [homebrew](http://brew.sh) to install the bazel and SWIG
dependencies, and installing python dependencies using easy_install or pip.

Of course you can also install Swig from source without using homebrew. In that
case, be sure to install its dependency [PCRE](http://www.pcre.org) and not PCRE2.

#### Dependencies

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for bazel. You can then use homebrew to install bazel and SWIG:

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

We also recommend the [ipython](https://ipython.org) enhanced python shell,
which you can install as follows:

```bash
$ sudo easy_install ipython
```

#### Configure the installation

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter.

This step is used to locate the python and numpy header files.

```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N]
```

### Create the pip package and install

When building from source, you will still build a pip package and install that.

```bash
$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# The name of the .whl file will depend on your platform.
$ pip install /tmp/tensorflow_pkg/tensorflow-0.8.0rc0-py2-none-linux_x86_64.whl
```

## Setting up TensorFlow for Development

If you're working on TensorFlow itself, it is useful to be able to test your
changes in an interactive python shell without having to reinstall TensorFlow.

To set up TensorFlow such that all files are linked (instead of copied) from the
system directories, run the following commands inside the TensorFlow root
directory:

```bash
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

mkdir _python_build
cd _python_build
ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
ln -s ../tensorflow/tools/pip_package/* .
python setup.py develop
```

Note that this setup still requires you to rebuild the
`//tensorflow/tools/pip_package:build_pip_package` target every time you change
a C++ file; add, delete, or move any python file; or if you change bazel build
rules.

## Train your first TensorFlow neural net model

Starting from the root of your source tree, run:

```bash
$ cd tensorflow/models/image/mnist
$ python convolutional.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
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

## Common Problems

### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the GPU installation [instructions](#optional-install-cuda-gpus-on-linux).
If you built from source, and you left the Cuda or cuDNN version empty, try specifying them
explicitly.

### Protobuf library related issues

TensorFlow pip package depends on protobuf pip package version
3.0.0b2. Protobuf's pip package downloaded from [PyPI](https://pypi.python.org)
(when running `pip install protobuf`) is a Python only library, that has
Python implementations of proto serialization/deserialization which can be 10x-50x
slower than the C++ implementation. Protobuf also supports a binary extension
for the Python package that contains fast C++ based proto parsing. This
extension is not available in the standard Python only PIP package. We have
created a custom binary pip package for protobuf that contains the binary
extension. Follow these instructions to install the custom binary protobuf pip
package :

```bash
# Ubuntu/Linux 64-bit:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl

# Mac OS X:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp27-none-any.whl
```

and for Python 3 :

```bash
# Ubuntu/Linux 64-bit:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp34-none-linux_x86_64.whl

# Mac OS X:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp35-none-any.whl
```

Install the above package _after_ you have installed TensorFlow via pip, as the
standard `pip install tensorflow` would install the python only pip package. The
above pip package will over-write the existing protobuf package.
Note that the binary pip package already has support for protobuf larger than
64MB, that should fix errors such as these :

```bash
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.

```

### Pip installation issues

#### Cannot import name 'descriptor'

```python
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/core/framework/graph_pb2.py", line 6, in <module>
    from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'
```

If you the above error when upgrading to a newer version of TensorFlow, try
uninstalling both TensorFlow and protobuf (if installed) and re-installing
TensorFlow (which will also install the correct protobuf dependency).

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

    *  Using [Virtualenv](#virtualenv-installation).
    *  Using [Docker](#docker-installation).

* Install a separate copy of Python via [Homebrew](http://brew.sh/) or
[MacPorts](https://www.macports.org/) and re-install TensorFlow in that
copy of Python.

### Mac OS X: OSError: [Errno 1] Operation not permitted:

On El Capitan, "six" is a special package that can't be modified, and this
error is reported when "pip install" tried to modify this package. To fix use
"ignore_installed" flag, ie

sudo pip install --ignore-installed six https://storage.googleapis.com/....


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
