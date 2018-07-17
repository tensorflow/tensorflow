# Install TensorFlow from Sources

This guide explains how to build TensorFlow sources into a TensorFlow binary and
how to install that TensorFlow binary. Note that we provide well-tested,
pre-built TensorFlow binaries for Ubuntu, macOS, and Windows systems. In
addition, there are pre-built TensorFlow
[docker images](https://hub.docker.com/r/tensorflow/tensorflow/). So, don't
build a TensorFlow binary yourself unless you are very comfortable building
complex packages from source and dealing with the inevitable aftermath should
things not go exactly as documented.

If the last paragraph didn't scare you off, welcome. This guide explains how to
build TensorFlow on 64-bit desktops and laptops running either of the following
operating systems:

*   Ubuntu
*   macOS X

Note: Some users have successfully built and installed TensorFlow from sources
on non-supported systems. Please remember that we do not fix issues stemming
from these attempts.

We **do not support** building TensorFlow on Windows. That said, if you'd like
to try to build TensorFlow on Windows anyway, use either of the following:

*   [Bazel on Windows](https://bazel.build/versions/master/docs/windows.html)
*   [TensorFlow CMake build](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/cmake)

Note: Starting from 1.6 release, our prebuilt binaries will use AVX
instructions. Older CPUs may not be able to execute these binaries.

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
    one of the following documents:

    *   @ {$install_linux#NVIDIARequirements$Installing TensorFlow on Ubuntu}
    *   @ {$install_mac#NVIDIARequirements$Installing TensorFlow on macOS}

## Clone the TensorFlow repository

Start the process of building TensorFlow by cloning a TensorFlow repository.

To clone **the latest** TensorFlow repository, issue the following command:

<pre>$ <b>git clone https://github.com/tensorflow/tensorflow</b> </pre>

The preceding <code>git clone</code> command creates a subdirectory named
`tensorflow`. After cloning, you may optionally build a **specific branch**
(such as a release branch) by invoking the following commands:

<pre>
$ <b>cd tensorflow</b>
$ <b>git checkout</b> <i>Branch</i> # where <i>Branch</i> is the desired branch
</pre>

For example, to work with the `r1.0` release instead of the master release,
issue the following command:

<pre>$ <b>git checkout r1.0</b></pre>

Next, you must prepare your environment for [Linux](#PrepareLinux) or
[macOS](#PrepareMac)

<a name="PrepareLinux"></a>

## Prepare environment for Linux

Before building TensorFlow on Linux, install the following build tools on your
system:

*   bazel
*   TensorFlow Python dependencies
*   optionally, NVIDIA packages to support TensorFlow for GPU.

### Install Bazel

If bazel is not installed on your system, install it now by following
[these directions](https://bazel.build/versions/master/docs/install.html).

### Install TensorFlow Python dependencies

To install TensorFlow, you must install the following packages:

*   `numpy`, which is a numerical processing package that TensorFlow requires.
*   `dev`, which enables adding extensions to Python.
*   `pip`, which enables you to install and manage certain Python packages.
*   `wheel`, which enables you to manage Python compressed packages in the wheel
    (.whl) format.

To install these packages for Python 2.7, issue the following command:

<pre>
$ <b>sudo apt-get install python-numpy python-dev python-pip python-wheel</b>
</pre>

To install these packages for Python 3.n, issue the following command:

<pre>
$ <b>sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel</b>
</pre>

### Optional: install TensorFlow for GPU prerequisites

If you are building TensorFlow without GPU support, skip this section.

The following NVIDIA® <i>hardware</i> must be installed on your system:

*   GPU card with CUDA Compute Capability 3.5 or higher. See
    [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) for a list of
    supported GPU cards.

The following NVIDIA® <i>software</i> must be installed on your system:

*   [GPU drivers](http://nvidia.com/driver). CUDA 9.0 requires 384.x or higher.
*   [CUDA Toolkit](http://nvidia.com/cuda) (>= 8.0). We recommend version 9.0.
*   [cuDNN SDK](http://developer.nvidia.com/cudnn) (>= 6.0). We recommend
    version 7.1.x.
*   [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit, but
    you also need to append its path to the `LD_LIBRARY_PATH` environment
    variable: `export
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64`
*   *OPTIONAL*: [NCCL 2.2](https://developer.nvidia.com/nccl) to use TensorFlow
    with multiple GPUs.
*   *OPTIONAL*:
    [TensorRT](http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
    which can improve latency and throughput for inference for some models.

While it is possible to install the NVIDIA libraries via `apt-get` from the
NVIDIA repository, the libraries and headers are installed in locations that
make it difficult to configure and debug build issues. Downloading and
installing the libraries manually or using docker
([latest-devel-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags/)) is
recommended.

### Next

After preparing the environment, you must now
[configure the installation](#ConfigureInstallation).

<a name="PrepareMac"></a>

## Prepare environment for macOS

Before building TensorFlow, you must install the following on your system:

*   bazel
*   TensorFlow Python dependencies.
*   optionally, NVIDIA packages to support TensorFlow for GPU.

### Install bazel

If bazel is not installed on your system, install it now by following
[these directions](https://bazel.build/versions/master/docs/install.html#mac-os-x).

### Install python dependencies

To build TensorFlow, you must install the following packages:

*   six
*   numpy, which is a numerical processing package that TensorFlow requires.
*   wheel, which enables you to manage Python compressed packages in the wheel
    (.whl) format.

You may install the python dependencies using pip. If you don't have pip on your
machine, we recommend using homebrew to install Python and pip as
[documented here](http://docs.python-guide.org/en/latest/starting/install/osx/).
If you follow these instructions, you will not need to disable SIP.

After installing pip, invoke the following commands:

<pre> $ <b>sudo pip install six numpy wheel</b> </pre>

Note: These are just the minimum requirements to _build_ tensorflow. Installing
the pip package will download additional packages required to _run_ it. If you
plan on executing tasks directly with `bazel` , without the pip installation,
you may need to install additional python packages. For example, you should `pip
install mock enum34` before running TensorFlow's tests with bazel.

<a name="ConfigureInstallation"></a>

## Configure the installation

The root of the source tree contains a bash script named <code>configure</code>.
This script asks you to identify the pathname of all relevant TensorFlow
dependencies and specify other build configuration options such as compiler
flags. You must run this script *prior* to creating the pip package and
installing TensorFlow.

If you wish to build TensorFlow with GPU, `configure` will ask you to specify
the version numbers of CUDA and cuDNN. If several versions of CUDA or cuDNN are
installed on your system, explicitly select the desired version instead of
relying on the default.

One of the questions that `configure` will ask is as follows:

<pre>
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]
</pre>

This question refers to a later phase in which you'll use bazel to
[build the pip package](#build-the-pip-package) or the
[C/Java libraries](#BuildCorJava). We recommend accepting the default
(`-march=native`), which will optimize the generated code for your local
machine's CPU type. However, if you are building TensorFlow on one CPU type but
will run TensorFlow on a different CPU type, then consider specifying a more
specific optimization flag as described in
[the gcc documentation](https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html).

Here is an example execution of the `configure` script. Note that your own input
will likely differ from our sample input:

<pre>
$ <b>cd tensorflow</b>  # cd to the top-level directory created
$ <b>./configure</b>
Please specify the location of python. [Default is /usr/bin/python]: <b>/usr/bin/python2.7</b>
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
No XLA support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N]
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N]
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] <b>Y</b>
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: <b>9.0</b>
Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: <b>7</b>
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.

Do you wish to build TensorFlow with MPI support? [y/N]
MPI support will not be enabled for TensorFlow
Configuration finished
</pre>

[Default is: "3.5,7.0"]: <b>6.0,7.0</b>

If you told `configure` to build for GPU support, then `configure` will create a
canonical set of symbolic links to the CUDA libraries on your system. Therefore,
every time you change the CUDA library paths, you must rerun the `configure`
script before re-invoking the <code>bazel build</code> command.

Note the following:

*   Although it is possible to build both CUDA and non-CUDA configs under the
    same source tree, we recommend running `bazel clean` when switching between
    these two configurations in the same source tree.
*   If you don't run the `configure` script *before* running the `bazel build`
    command, the `bazel build` command will fail.

## Build the pip package

Note: If you're only interested in building the libraries for the TensorFlow C
or Java APIs, see [Build the C or Java libraries](#BuildCorJava), you do not
need to build the pip package in that case.

### CPU-only support

To build a pip package for TensorFlow with CPU-only support:

<pre>
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
</pre>

To build a pip package for TensorFlow with CPU-only support for the Intel®
MKL-DNN:

<pre>
$ bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
</pre>

### GPU support

To build a pip package for TensorFlow with GPU support:

<pre>
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
</pre>

**NOTE on gcc 5 or later:** the binary pip packages available on the TensorFlow
website are built with gcc 4, which uses the older ABI. To make your build
compatible with the older ABI, you need to add
`--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` to your `bazel build` command. ABI
compatibility allows custom ops built against the TensorFlow pip package to
continue to work against your built package.

<b>Tip:</b> By default, building TensorFlow from sources consumes a lot of RAM.
If RAM is an issue on your system, you may limit RAM usage by specifying
<code>--local_resources 2048,.5,1.0</code> while invoking `bazel`.

The <code>bazel build</code> command builds a script named `build_pip_package`.
Running this script as follows will build a `.whl` file within the
`/tmp/tensorflow_pkg` directory:

<pre>
$ <b>bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</b>
</pre>

## Install the pip package

Invoke `pip install` to install that pip package. The filename of the `.whl`
file depends on your platform. For example, the following command will install
the pip package

for TensorFlow 1.9.0rc0 on Linux:

<pre>
$ <b>sudo pip install /tmp/tensorflow_pkg/tensorflow-1.9.0rc0-py2-none-any.whl</b>
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

If the system outputs an error message instead of a greeting, see
[Common installation problems](#common_installation_problems).

## Common build and installation problems

The build and installation problems you encounter typically depend on the
operating system. See the "Common installation problems" section of one of the
following guides:

*   @
    {$install_linux#common_installation_problems$Installing TensorFlow on Linux}
*   @
    {$install_mac#common_installation_problems$Installing TensorFlow on Mac OS}
*   @
    {$install_windows#common_installation_problems$Installing TensorFlow on Windows}

Beyond the errors documented in those two guides, the following table notes
additional errors specific to building TensorFlow. Note that we are relying on
Stack Overflow as the repository for build and installation problems. If you
encounter an error message not listed in the preceding two guides or in the
following table, search for it on Stack Overflow. If Stack Overflow doesn't show
the error message, ask a new question on Stack Overflow and specify the
`tensorflow` tag.

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>

<tr>
  <td><a
  href="https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions">41293077</a></td>
  <td><pre>W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow
  library wasn't compiled to use SSE4.1 instructions, but these are available on
  your machine and could speed up CPU computations.</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><pre>ImportError: libcudart.so.8.0: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><pre>ImportError: libcudnn.5: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/35953210">35953210</a></td>
  <td>Invoking `python` or `ipython` generates the following error:
  <pre>ImportError: cannot import name pywrap_tensorflow</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/questions/45276830">45276830</a></td>
  <td><pre>external/local_config_cc/BUILD:50:5: in apple_cc_toolchain rule
  @local_config_cc//:cc-compiler-darwin_x86_64: Xcode version must be specified
  to use an Apple CROSSTOOL.</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/47080760">47080760</a></td>
  <td><pre>undefined reference to `cublasGemmEx@libcublas.so.9.0'</pre></td>
</tr>

</table>

## Tested source configurations

**Linux**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.11.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.9.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.11.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.10.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.8.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.10.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.7.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.6.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.8.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.5.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.8.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.5.4</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.5.4</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.3.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>

**Mac**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.11.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.10.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.10.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.8.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.8.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.5.4</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>

**Windows**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.9.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.8.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.7.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.6.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.5.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.3.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
</table>

<a name="BuildCorJava"></a>

## Build the C or Java libraries

The instructions above are tailored to building the TensorFlow Python packages.

If you're interested in building the libraries for the TensorFlow C API, do the
following:

1.  Follow the steps up to [Configure the installation](#ConfigureInstallation)
2.  Build the C libraries following instructions in the
    [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md).

If you're interested inv building the libraries for the TensorFlow Java API, do
the following:

1.  Follow the steps up to [Configure the installation](#ConfigureInstallation)
2.  Build the Java library following instructions in the
    [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md).
