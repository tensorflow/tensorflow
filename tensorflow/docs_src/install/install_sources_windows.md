# Install TensorFlow from Sources on Windows

This guide explains how to build TensorFlow sources into a TensorFlow binary and
how to install that TensorFlow binary on Windows.

## Determine which TensorFlow to install

You must choose one of the following types of TensorFlow to build and install:

*   **TensorFlow with CPU support only**. If your system does not have a NVIDIA®
    GPU, build and install this version. Note that this version of TensorFlow is
    typically easier to build and install, so even if you have an NVIDIA GPU, we
    recommend building and installing this version first.
*   **TensorFlow with GPU support**. TensorFlow programs typically run
    significantly faster on a GPU than on a CPU. Therefore, if your system has a
    NVIDIA GPU and you need to run performance-critical applications, you should
    ultimately build and install this version. Beyond the NVIDIA GPU itself,
    your system must also fulfill the NVIDIA software requirements described in
    the following document:

    *   [Installing TensorFlow on Windows](install_windows.md#NVIDIARequirements)

## Prepare environment for Windows

Before building TensorFlow on Windows, install the following build tools on your
system:

*   [MSYS2](#InstallMSYS2)
*   [Visual C++ build tools](#InstallVCBuildTools)
*   [Bazel for Windows](#InstallBazel)
*   [TensorFlow Python dependencies](#InstallPython)
*   [optionally, NVIDIA packages to support TensorFlow for GPU](#InstallCUDA)

<a name="InstallMSYS2"></a>

### Install MSYS2

Bash bin tools are used in TensorFlow Bazel build, you can install them through [MSYS2](https://www.msys2.org/).

Assume you installed MSYS2 at `C:\msys64`, add `C:\msys64\usr\bin` to your `%PATH%` environment variable.

To install necessary bash bin tools, issue the following command under `cmd.exe`:

<pre>
C:\> <b>pacman -S git patch unzip</b>
</pre>

<a name="InstallVCBuildTools"></a>

### Install Visual C++ Build Tools 2015

To build TensorFlow, you need to install Visual C++ build tools 2015. It is a part of Visual Studio 2015.
But you can install it separately by the following way:

  * Open the [official downloand page](https://visualstudio.microsoft.com/vs/older-downloads/).
  * Go to <b>Redistributables and Build Tools</b> section.
  * Find <b>Microsoft Build Tools 2015 Update 3</b> and click download.
  * Run the installer.

It's possible to build TensorFlow with newer version of Visual C++ build tools,
but we only test against Visual Studio 2015 Update 3.

<a name="InstallBazel"></a>

### Install Bazel

If bazel is not installed on your system, install it now by following
[these instructions](https://docs.bazel.build/versions/master/install-windows.html).
It is recommended to use a Bazel version >= `0.15.0`.

Add the directory where you installed Bazel to your `%PATH%` environment variable.

<a name="InstallPython"></a>

### Install TensorFlow Python dependencies

If you don't have Python 3.5 or Python 3.6 installed, install it now:

  * [Python 3.5.x 64-bit from python.org](https://www.python.org/downloads/release/python-352/)
  * [Python 3.6.x 64-bit from python.org](https://www.python.org/downloads/release/python-362/)

To build and install TensorFlow, you must install the following python packages:

*   `six`, which provides simple utilities for wrapping over differences between
    Python 2 and Python 3.
*   `numpy`, which is a numerical processing package that TensorFlow requires.
*   `wheel`, which enables you to manage Python compressed packages in the wheel
    (.whl) format.
*   `keras_applications`, the applications module of the Keras deep learning library.
*   `keras_preprocessing`, the data preprocessing and data augmentation module
    of the Keras deep learning library.

Assume you already have `pip3` in `%PATH%`, issue the following command:

<pre>
C:\> <b>pip3 install six numpy wheel</b>
C:\> <b>pip3 install keras_applications==1.0.4 --no-deps</b>
C:\> <b>pip3 install keras_preprocessing==1.0.2 --no-deps</b>
</pre>

<a name="InstallCUDA"></a>

### Optional: install TensorFlow for GPU prerequisites

If you are building TensorFlow without GPU support, skip this section.

The following NVIDIA® _hardware_ must be installed on your system:

*   GPU card with CUDA Compute Capability 3.5 or higher. See
    [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) for a list of
    supported GPU cards.

The following NVIDIA® _software_ must be installed on your system:

*   [GPU drivers](http://nvidia.com/driver). CUDA 9.0 requires 384.x or higher.
*   [CUDA Toolkit](http://nvidia.com/cuda) (>= 8.0). We recommend version 9.0.
*   [cuDNN SDK](http://developer.nvidia.com/cudnn) (>= 6.0). We recommend
    version 7.1.x.
*   [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit, but
    you also need to append its path to `%PATH%` environment
    variable.

Assume you have CUDA Toolkit installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`
and cuDNN at `C:\tools\cuda`, issue the following commands.

<pre>
C:\> SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
C:\> SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%
C:\> SET PATH=C:\tools\cuda\bin;%PATH%
</pre>

## Clone the TensorFlow repository

Now you need to clone **the latest** TensorFlow repository,
thanks to MSYS2 we already have `git` avaiable, issue the following command:

<pre>C:\> <b>git clone https://github.com/tensorflow/tensorflow.git</b> </pre>

The preceding <code>git clone</code> command creates a subdirectory named
`tensorflow`. After cloning, you may optionally build a **specific branch**
(such as a release branch) by invoking the following commands:

<pre>
C:\> <b>cd tensorflow</b>
C:\> <b>git checkout</b> <i>Branch</i> # where <i>Branch</i> is the desired branch
</pre>

For example, to work with the `r1.11` release instead of the master release,
issue the following command:

<pre>C:\> <b>git checkout r1.11</b></pre>

Next, you must now configure the installation.

## Configure the installation

The root of the source tree contains a python script named <code>configure.py</code>.
This script asks you to identify the pathname of all relevant TensorFlow
dependencies and specify other build configuration options such as compiler
flags. You must run this script *prior* to creating the pip package and
installing TensorFlow.

If you wish to build TensorFlow with GPU, `configure.py` will ask you to specify
the version numbers of CUDA and cuDNN. If several versions of CUDA or cuDNN are
installed on your system, explicitly select the desired version instead of
relying on the default.

One of the questions that `configure.py` will ask is as follows:

<pre>
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]:
</pre>

Here is an example execution of the `configure.py` script. Note that your own input
will likely differ from our sample input:

<pre>
C:\> <b>cd tensorflow</b>  # cd to the top-level directory created
C:\tensorflow> <b>python ./configure.py</b>
Starting local Bazel server and connecting to it...
................
You have bazel 0.15.0 installed.
Please specify the location of python. [Default is C:\python36\python.exe]: 

Found possible Python library paths:
  C:\python36\lib\site-packages
Please input the desired Python library path to use.  Default is [C:\python36\lib\site-packages]

Do you wish to build TensorFlow with CUDA support? [y/N]: <b>Y</b>
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]:

Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: <b>7.0</b>

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]: <b>C:\tools\cuda</b>

Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,7.0]: <b>3.7</b>

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]: 

Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]:
Eigen strong inline overridden.

Configuration finished
</pre>

## Build the pip package

### CPU-only support

To build a pip package for TensorFlow with CPU-only support:

<pre>
C:\tensorflow> <b>bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package</b>
</pre>

### GPU support

To build a pip package for TensorFlow with GPU support:

<pre>
C:\tensorflow> <b>bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package</b>
</pre>

**NOTE :** When building with GPU support, you might want to add `--copt=-nvcc_options=disable-warnings`
to suppress nvcc warning messages.

The `bazel build` command builds a binary named `build_pip_package`
(an executable binary to launch bash and run a bash script to create the pip package).
Running this binary as follows will build a `.whl` file within the `C:/tmp/tensorflow_pkg` directory:

<pre>
C:\tensorflow> <b>bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg</b>
</pre>

## Install the pip package

Invoke `pip3 install` to install that pip package. The filename of the `.whl`
file depends on the TensorFlow version and your platform. For example, the
following command will install the pip package for TensorFlow 1.11.0rc0:

<pre>
C:\tensorflow> <b>pip3 install C:/tmp/tensorflow_pkg/tensorflow-1.11.0rc0-cp36-cp36m-win_amd64.whl</b>
</pre>

## Validate your installation

Validate your TensorFlow installation by doing the following:

Start a terminal.

Change directory (`cd`) to any directory on your system other than the
`tensorflow` subdirectory from which you invoked the `configure` command.

Invoke python:

<pre>$ <b>python</b></pre>

Enter the following short program inside the python interactive shell:

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

If the system outputs the following, then you are ready to begin writing
TensorFlow programs:

<pre>Hello, TensorFlow!</pre>

To learn more, see the [TensorFlow tutorials](../tutorials/).

## Build under MSYS shell
The above instruction assumes you are building under the Windows native command line (`cmd.exe`), but you can also
build TensorFlow from MSYS shell. There are a few things to notice:

*   Disable the path conversion heuristic in MSYS. MSYS automatically converts arguments that look
    like a Unix path to Windows path when running a program, this will confuse Bazel.
    (eg. A Bazel label `//foo/bar:bin` is considered a Unix absolute path, only because it starts with a slash)

  ```sh
$ export MSYS_NO_PATHCONV=1
$ export MSYS2_ARG_CONV_EXCL="*"
```

*   Add the directory where you install Bazel in `$PATH`. Assume you have Bazel
    installed at `C:\tools\bazel.exe`, issue the following command:

  ```sh
# `:` is used as path separator, so we have to convert the path to Unix style.
$ export PATH="/c/tools:$PATH"
```

*   Add the directory where you install Python in `$PATH`. Assume you have
    Python installed at `C:\Python36\python.exe`, issue the following command:

  ```sh
$ export PATH="/c/Python36:$PATH"
```

*   If you have Python in `$PATH`, you can run configure script just by
    `./configure`, a shell script will help you invoke python.

*   (For GPU build only) Add Cuda and cuDNN bin directories in `$PATH` in the following way:

  ```sh
$ export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin:$PATH"
$ export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/extras/CUPTI/libx64:$PATH"
$ export PATH="/c/tools/cuda/bin:$PATH"
```

The rest steps should be the same as building under `cmd.exe`.
