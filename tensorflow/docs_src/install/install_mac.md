# Installing TensorFlow on Mac OS X

This guide explains how to install TensorFlow on Mac OS X.

## Determine which TensorFlow to install

You must choose the type of TensorFlow to install.  Your choices are as follows:

  * **TensorFlow with CPU support only**. If your system does not have a
    NVIDIA CUDA® GPU, you should install this version. Note that TensorFlow
    with CPU support is typically easier to install than TensorFlow with
    GPU support. Therefore, even if you have an NVIDIA CUDA GPU, we recommend
    installing this version first as a diagnostic step just in case you run
    into problems installing TensorFlow with GPU support.
  * **TensorFlow with GPU support**. TensorFlow programs typically run
    significantly faster on a GPU than on a CPU. Therefore, if your system has
    a NVIDIA CUDA GPU meeting the prerequisites shown below and you need
    to run performance-critical applications, you should ultimately
    install this version.


### Requirements to run TensorFlow with GPU support

If you are installing TensorFlow with GPU support using one of the mechanisms
described in this guide, then the following NVIDIA software must be
installed on your system:


  * CUDA Toolkit 8.0. For details, see
    [NVIDIA's documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x).
    Ensure that you append the relevant CUDA pathnames to the
    `LD_LIBRARY_PATH` environment variable as described in the
    NVIDIA documentation.
  * The NVIDIA drivers associated with CUDA Toolkit 8.0.
  * cuDNN v5.1. For details, see
    [NVIDIA's documentation](https://developer.nvidia.com/cudnn).
    Ensure that you create the `CUDA_HOME` environment variable as described in
    the NVIDIA documentation.
  * GPU card with CUDA Compute Capability 3.0 or higher.  See
    [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus)
    for a list of supported GPU cards.

If you have an earlier version of the preceding packages, please upgrade to
the specified versions. If upgrading is not possible, you may still run
TensorFlow with GPU support, but only if you do both of the following:

  * Install TensorFlow from sources as described
    @{$install_sources$here}.
  * Install or upgrade to at least the following NVIDIA versions:
    * CUDA toolkit 7.0 or greater
    * cuDNN v3 or greater
    * GPU card with CUDA Compute Capability 3.0 or higher.


## Determine how to install TensorFlow

You must pick the mechanism by which you install TensorFlow. The supported choices are as follows:

  * virtualenv
  * "native" pip
  * Docker
  * installing from sources, which is for experts and is documented in
    a separate guide.

**We recommend the virtualenv installation.**
[Virtualenv](https://virtualenv.pypa.io/en/stable/)
is a virtual Python environment isolated from other Python development,
incapable of interfering with or being affected by other Python programs
on the same machine.  During the virtualenv installation process,
you will install not only TensorFlow but also all the packages that
TensorFlow requires.  (This is actually pretty easy.)
To start working with TensorFlow, you simply need to "activate" the
virtual environment.  All in all, virtualenv provides a safe and
reliable mechanism for installing and running TensorFlow.

Native pip installs TensorFlow directly on your system without going through
any container or virtual environment system. Since a native pip installation
is not walled-off, the pip installation might interfere with or be influenced
by other Python-based installations on your system. Furthermore, you might need
to disable System Integrity Protection (SIP) in order to install through native
pip.  However, if you understand SIP, pip, and your Python environment, a
native pip installation is relatively easy to perform.

[Docker](http://docker.com/) completely isolates the TensorFlow installation
from pre-existing packages on your machine. The Docker container contains
TensorFlow and all its dependencies. Note that the Docker image can be quite
large (hundreds of MBs). You might choose the Docker installation if you are
incorporating TensorFlow into a larger application architecture that
already uses Docker.

Important: Docker currently does not support TensorFlow with GPU support
on Mac OS; that is, on Mac OS, Docker only supports TensorFlow with
CPU support.

In Anaconda, you may use conda to create a virtual environment.
However, within Anaconda, we recommend installing TensorFlow with the
`pip install` command, not with the `conda install` command.

**NOTE:** The conda package is community supported, not officially supported.
That is, the TensorFlow team neither tests nor maintains the conda package.
Use that package at your own risk.

## Installing with virtualenv

Take the following steps to install TensorFlow with Virtualenv:

  1. Start a terminal (a shell). You'll perform all subsequent steps
     in this shell.

  2. Install pip and virtualenv by issuing the following commands:

     <pre> $ <b>sudo easy_install pip</b>
     $ <b>sudo pip install --upgrade virtualenv</b> </pre>

  3. Create a virtualenv environment by issuing a command of one
     of the following formats:

     <pre> $ <b>virtualenv --system-site-packages</b> <i>targetDirectory</i> # for Python 2.7
     $ <b>virtualenv --system-site-packages -p python3</b> <i>targetDirectory</i> # for Python 3.n
     </pre>

     The <i>targetDirectory</i> identifies the top of the virtualenv tree.
     Our instructions assume that <i>targetDirectory</i>
     is `~/tensorflow`, but you may choose any directory.

  4. Activate the virtualenv environment by issuing one of the
     following commands:

     <pre> $ <b>source ~/tensorflow/bin/activate</b>      # If using bash, sh, ksh, or zsh
     $ <b>source ~/tensorflow/bin/activate.csh</b>  # If using csh or tcsh </pre>

     The preceding `source` command should change your prompt to the following:

     <pre> (tensorflow)$ </pre>

  5. If pip version 8.1 or later is installed on your system, issue one of
     the following commands to install TensorFlow and all the packages that
     TensorFlow requires into the active Virtualenv environment:

     <pre> $ <b>pip install --upgrade tensorflow</b>      # for Python 2.7
     $ <b>pip3 install --upgrade tensorflow</b>     # for Python 3.n
     $ <b>pip install --upgrade tensorflow-gpu</b>  # for Python 2.7 and GPU
     $ <b>pip3 install --upgrade tensorflow-gpu</b> # for Python 3.n and GPU </pre>

     If the preceding command succeed, skip Step 6. If it failed,
     perform Step 6.

  6. Optional. If Step 5 failed (typically because you invoked a pip version
     lower than 8.1), install TensorFlow in the active
     virtualenv environment by issuing a command of the following format:

     <pre> $ <b>pip install --upgrade</b> <i>TF_BINARY_URL</i>   # Python 2.7
     $ <b>pip3 install --upgrade</b> <i>TF_BINARY_URL</i>  # Python 3.n </pre>

     where <i>TF_BINARY_URL</i> identifies the URL
     of the TensorFlow Python package. The appropriate value of
     <i>TF_BINARY_URL</i> depends on the operating system,
     Python version, and GPU support. Find the appropriate value for
     <i>TF_BINARY_URL</i> for your system
     [here](#the_url_of_the_tensorflow_python_package).
     For example, if you are installing TensorFlow for Mac OS X,
     Python 2.7, and CPU-only support, the command to install
     TensorFlow in the active Virtualenv is as follows:

     <pre> $ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl</b></pre>

If you encounter installation problems, see
[Common Installation Problems](#CommonInstallationProblems).


### Next Steps

After installing TensorFlow,
[validate your installation](#ValidateYourInstallation)
to confirm that the installation worked properly.

Note that you must activate the virtualenv environment each time you
use TensorFlow in a new shell.  If the virtualenv environment is not
currently active (that is, the prompt is not `(tensorflow)`, invoke
one of the following commands:

<pre> $ <b>source ~/tensorflow/bin/activate</b>      # bash, sh, ksh, or zsh
$ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh </pre>

Your prompt will transform to the following to indicate that your
tensorflow environment is active:

<pre> (tensorflow)$ </pre>

When the virtualenv environment is active, you may run
TensorFlow programs from this shell.

When you are done using TensorFlow, you may deactivate the
environment by issuing the following command:

<pre> (tensorflow)$ <b>deactivate</b> </pre>

The prompt will revert back to your default prompt (as defined by `PS1`).


### Uninstalling TensorFlow

If you want to uninstall TensorFlow, simply remove the tree you created. For example:

<pre> $ <b>rm -r ~/tensorflow</b> </pre>


## Installing with native pip

We have uploaded the TensorFlow binaries to PyPI.
Therefore, you can install TensorFlow through pip.

The
[REQUIRED_PACKAGES section of setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)
lists the packages that pip will install or upgrade.


### Prerequisite: Python

In order to install TensorFlow, your system must contain one of the following Python versions:

  * Python 2.7
  * Python 3.3+

If your system does not already have one of the preceding Python versions,
[install](https://wiki.python.org/moin/BeginnersGuide/Download) it now.

When installing Python, you might need to disable
System Integrity Protection (SIP) to permit any entity other than
Mac App Store to install software.


### Prerequisite: pip

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) installs
and manages software packages written in Python. If you intend to install
with native pip, then one of the following flavors of pip must be
installed on your system:

  * `pip`, for Python 2.7
  * `pip3`, for Python 3.n.

`pip` or `pip3` was probably installed on your system when you
installed Python.  To determine whether pip or pip3 is actually
installed on your system, issue one of the following commands:

<pre>$ <b>pip -V</b>  # for Python 2.7
$ <b>pip3 -V</b> # for Python 3.n </pre>

We strongly recommend pip or pip3 version 8.1 or higher in order
to install TensorFlow.  If pip or pip3 8.1 or later is not
installed, issue the following commands to install or upgrade:

<pre>$ <b>sudo easy_install --upgrade pip</b>
$ <b>sudo easy_install --upgrade six</b> </pre>


### Install TensorFlow

Assuming the prerequisite software is installed on your Mac,
take the following steps:

  1. Install TensorFlow by invoking **one** of the following commands:

     <pre>$ <b>pip install tensorflow</b>      # Python 2.7; CPU support (no GPU support)
     $ <b>pip3 install tensorflow</b>     # Python 3.n; CPU support (no GPU support)
     $ <b>pip install tensorflow-gpu</b>  # Python 2.7;  GPU support
     $ <b>pip3 install tensorflow-gpu</b> # Python 3.n; GPU support </pre>

     If the preceding command runs to completion, you should now
     [validate your installation](#ValidateYourInstallation).

  2. (Optional.) If Step 1 failed, install the latest version of TensorFlow
     by issuing a command of the following format:

     <pre>$ <b>sudo pip  install --upgrade</b> <i>TF_BINARY_URL</i>   # Python 2.7
     $ <b>sudo pip3 install --upgrade</b> <i>TF_BINARY_URL</i>   # Python 3.n </pre>

     where <i>TF_BINARY_URL</i> identifies the URL of the TensorFlow Python
     package. The appropriate value of <i>TF_BINARY_URL</i> depends on the
     operating system, Python version, and GPU support. Find the appropriate
     value for <i>TF_BINARY_URL</i>
     [here](#the_url_of_the_tensorflow_python_package).  For example, if
     you are installing TensorFlow for Mac OS, Python 2.7, and CPU-only
     support, issue the following command:

     <pre> $ <b>sudo pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl</b> </pre>

     If the preceding command fails, see
     [Common installation problems](#CommonInstallationProblems).



### Next Steps

After installing TensorFlow,
[validate your installation](#ValidateYourInstallation)
to confirm that the installation worked properly.


### Uninstalling TensorFlow

To uninstall TensorFlow, issue one of following commands:

<pre>$ <b>pip uninstall tensorflow</b>
$ <b>pip3 uninstall tensorflow</b> </pre>


## Installing with Docker

Follow these steps to install TensorFlow through Docker.

  1. Install Docker on your machine as described in the
     [Docker documentation](https://docs.docker.com/engine/installation/#/on-macos-and-windows).

  2. Launch a Docker container that contains one of the TensorFlow
     binary images.

The remainder of this section explains how to launch a Docker container.

**Note**: You may only launch a Docker container with CPU support.
(Docker doesn't provide GPU support on Mac OS.)

To launch a Docker container that holds the TensorFlow binary image,
enter a command of the following format:

<pre> $ <b>docker run -it <i>-p hostPort:containerPort</i> TensorFlowImage</b> </pre>

where:

  * <i>-p hostPort:containerPort</i> is optional. If you'd like to run
    TensorFlow programs from the shell, omit this option. If you'd like
    to run TensorFlow programs from Jupyter notebook,  set both
    <i>hostPort</i> and <i>containerPort</i> to <code>8888</code>.
    If you'd like to run TensorBoard inside the container, add
    a second `-p` flag, setting both <i>hostPort</i> and <i>containerPort</i>
    to 6006.
  * <i>TensorFlowImage</i> is required. It identifies the Docker container.
    You must specify one of the following values:
    * <code>gcr.io/tensorflow/tensorflow</code>: TensorFlow binary image.
    * <code>gcr.io/tensorflow/tensorflow:latest-devel</code>: TensorFlow
      Binary image plus source code.

<code>gcr.io</code> is the Google Container Registry. Note that some
TensorFlow images are also available at
[dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/).

For example, the following command launches a TensorFlow CPU binary image
in a Docker container from which you can run TensorFlow programs in a shell:

<pre>$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b></pre>

The following command also launches a TensorFlow CPU binary image in a
Docker container. However, in this Docker container, you can run
TensorFlow programs in a Jupyter notebook:

<pre>$ <b>docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow</b></pre>

Docker will download the TensorFlow binary image the first time you launch it.


### Next Steps

You should now
[validate your installation](#ValidateYourInstallation).


## Installing with Anaconda

**The Anaconda installation is community supported, not officially supported.**

Take the following steps to install TensorFlow in an Anaconda environment:

  1. Follow the instructions on the
     [Anaconda download site](https://www.continuum.io/downloads)
     to download and install Anaconda.

  2. Create a conda environment named `tensorflow`
     by invoking the following command:

     <pre>$ <b>conda create -n tensorflow</b></pre>

  3. Activate the conda environment by issuing the following command:

     <pre>$ <b>source activate tensorflow</b>
     (tensorflow)$  # Your prompt should change</pre>

  4. Issue a command of the following format to install
     TensorFlow inside your conda environment:

     <pre>(tensorflow)<b>$ pip install --ignore-installed --upgrade $TF_PYTHON_URL</b></pre>

     where `TF_PYTHON_URL` is the
     [URL of the TensorFlow Python package](#the_url_of_the_tensorflow_python_package).
     For example, the following command installs the CPU-only version of
     TensorFlow for Python 2.7:

     <pre> (tensorflow)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl</b></pre>


<a name="ValidateYourInstallation"></a>
## Validate your installation

To validate your TensorFlow installation, do the following:

  1. Ensure that your environment is prepared to run TensorFlow programs.
  2. Run a short TensorFlow program.


### Prepare your environment

If you installed on native pip, virtualenv, or Anaconda, then
do the following:

  1. Start a terminal.
  2. If you installed with virtualenv or Anaconda, activate your container.
  3. If you installed TensorFlow source code, navigate to any
     directory *except* one containing TensorFlow source code.

If you installed through Docker, start a Docker container that runs bash.
For example:

<pre>$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b></pre>



### Run a short TensorFlow program

Invoke python from your shell as follows:

<pre>$ <b>python</b></pre>

Enter the following short program inside the python interactive shell:

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

If the system outputs the following, then you are ready to begin
writing TensorFlow programs:

<pre>Hello, TensorFlow!</pre>

If you are new to TensorFlow, see
@{$get_started$Getting Started with TensorFlow}.

If the system outputs an error message instead of a greeting, see
[Common installation problems](#common_installation_problems).

## Common installation problems

We are relying on Stack Overflow to document TensorFlow installation problems
and their remedies.  The following table contains links to Stack Overflow
answers for some common installation problems.
If you encounter an error message or other
installation problem not listed in the following table, search for it
on Stack Overflow.  If Stack Overflow doesn't show the error message,
ask a new question about it on Stack Overflow and specify
the `tensorflow` tag.

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>

<tr>
  <td><a href="https://stackoverflow.com/q/36159194">36159194</a></td>
  <td><pre>ImportError: libcudart.so.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41991101">41991101</a></td>
  <td><pre>ImportError: libcudnn.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><pre>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></pre>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><pre>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><pre>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33622019">33622019</a></td>
  <td><pre>ImportError: No module named copyreg</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/37810228">37810228</a></td>
  <td>During a <tt>pip install</tt> operation, the system returns:
  <pre>OSError: [Errno 1] Operation not permitted</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/33622842">33622842</a></td>
  <td>An <tt>import tensorflow</tt> statement triggers an error such as the
  following:<pre>Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py",
    line 4, in <module>
    from tensorflow.python import *
    ...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py",
    line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02
      \x03(\x0b\x32
      .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01
      \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
  TypeError: __init__() got an unexpected keyword argument 'syntax'</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42073336">42073336</a></td>
  <td>An <tt>import tensorflow</tt> statement triggers the following error:
<pre>
>>> import tensorflow as tf
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.dylib locally
"import tensorflow" terminated by signal SIGSEGV (Address boundary error)
</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42075397">42075397</a></td>
  <td>A <tt>pip install</tt> command triggers the following error:
<pre>...<lots of warnings and errors>
You have not agreed to the Xcode license agreements, please run
'xcodebuild -license' (for user-level acceptance) or
'sudo xcodebuild -license' (for system-wide acceptance) from within a
Terminal window to review and agree to the Xcode license agreements.
...<more stack trace output>
  File "numpy/core/setup.py", line 653, in get_mathlib_info

    raise RuntimeError("Broken toolchain: cannot link a simple C program")

RuntimeError: Broken toolchain: cannot link a simple C program</pre>
</td>

<tr>
  <td><a href="http://stackoverflow.com/questions/42376790/">42376790</a></td>
  <td>After installing for GPU, an `import tensorflow` statement
      triggers the following error:
<pre>tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA
  library libcublas.8.0.dylib locally
  Segmentation fault: 11
</pre></td>
</tr>

</table>




<a name="TF_PYTHON_URL"></a>
## The URL of the TensorFlow Python package

A few installation mechanisms require the URL of the TensorFlow Python package.
The value you specify depends on three factors:

  * operating system
  * Python version
  * CPU only vs. GPU support

This section documents the relevant values for Mac OS installations.

### Python 2.7

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl
</pre>

GPU support:

<pre>
https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.0.1-py2-none-any.whl
</pre>

Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see
[Installing TensorFlow from Sources](install_sources.md).


### Python 3.4, 3.5, or 3.6

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
</pre>

GPU support:

<pre>
https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.0.1-py3-none-any.whl
</pre>

Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see
[Installing TensorFlow from Sources](install_sources.md).



<a name="Protobuf31"></a>
## Protobuf pip package 3.1

You can skip this section unless you are seeing problems related
to the protobuf pip package.

**NOTE:** If your TensorFlow programs are running slowly, you might
have a problem related to the protobuf pip package.

The TensorFlow pip package depends on protobuf pip package version 3.1. The
protobuf pip package downloaded from PyPI (when invoking
<tt>pip install protobuf</tt>) is a Python-only library containing
Python implementations of proto serialization/deserialization that can run
**10x-50x slower** than the C++ implementation. Protobuf also supports a
binary extension for the Python package that contains fast
C++ based proto parsing.  This extension is not available in the
standard Python-only pip package.  We have created a custom binary
pip package for protobuf that contains the binary extension. To install
the custom binary protobuf pip package, invoke one of the following commands:

  * for Python 2.7:

  <pre> $ <b>pip install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp27-none-linux_x86_64.whl</b></pre>

  * for Python 3.n:

  <pre> $ <b>pip3 install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp35-none-linux_x86_64.whl</b> </pre>

Installing this protobuf package will overwrite the existing protobuf package.
Note that the binary pip package already has support for protobufs
larger than 64MB, which should fix errors such as these:

<pre>[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207]
A protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</pre>
