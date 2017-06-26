# Installing TensorFlow on Ubuntu

This guide explains how to install TensorFlow on Ubuntu. These instructions
might also work on other Linux variants, but we have only tested (and we
only support) these instructions on Ubuntu 14.04 or higher.


## Determine which TensorFlow to install

You must choose one of the following types of TensorFlow to install:

  * **TensorFlow with CPU support only**. If your system does not have a
    NVIDIA® GPU, you must install this version. Note that this version of
    TensorFlow is typically much easier to install (typically,
    in 5 or 10 minutes), so even if you have an NVIDIA GPU, we recommend
    installing this version first.
  * **TensorFlow with GPU support**. TensorFlow programs typically run
    significantly faster on a GPU than on a CPU. Therefore, if your
    system has a NVIDIA® GPU meeting the prerequisites shown below and you
    need to run performance-critical applications, you should ultimately
    install this version.

<a name="NVIDIARequirements"></a>
### NVIDIA requirements to run TensorFlow with GPU support

If you are installing TensorFlow with GPU support using one of the
mechanisms described in this guide, then the following NVIDIA software
must be installed on your system:

  * CUDA® Toolkit 8.0. For details, see
    [NVIDIA's documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A).
    Ensure that you append the relevant Cuda pathnames to the
    `LD_LIBRARY_PATH` environment variable as described in the
    NVIDIA documentation.
  * The NVIDIA drivers associated with CUDA Toolkit 8.0.
  * cuDNN v5.1. For details, see
    [NVIDIA's documentation](https://developer.nvidia.com/cudnn).
    Ensure that you create the `CUDA_HOME` environment variable as
    described in the NVIDIA documentation.
  * GPU card with CUDA Compute Capability 3.0 or higher.  See
    [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) for
    a list of supported GPU cards.
  * The libcupti-dev library, which is the NVIDIA CUDA Profile Tools Interface.
    This library provides advanced profiling support. To install this library,
    issue the following command:

    <pre>
    $ <b>sudo apt-get install libcupti-dev</b>
    </pre>

If you have an earlier version of the preceding packages, please upgrade to
the specified versions. If upgrading is not possible, then you may still run
TensorFlow with GPU support, but only if you do the following:

  * Install TensorFlow from sources as documented in
    @{$install_sources$Installing TensorFlow from Sources}.
  * Install or upgrade to at least the following NVIDIA versions:
    * CUDA toolkit 7.0 or greater
    * cuDNN v3 or greater
    * GPU card with CUDA Compute Capability 3.0 or higher.


## Determine how to install TensorFlow

You must pick the mechanism by which you install TensorFlow. The
supported choices are as follows:

  * [virtualenv](#InstallingVirtualenv)
  * ["native" pip](#InstallingNativePip)
  * [Docker](#InstallingDocker)
  * [Anaconda](#InstallingAnaconda)
  * installing from sources, which is documented in
    [a separate guide](https://www.tensorflow.org/install/install_sources).

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

Native pip installs TensorFlow directly on your system without going
through any container system. **We recommend the native pip install for
system administrators aiming to make TensorFlow available to everyone on a
multi-user system.** Since a native pip installation is not walled-off in
a separate container, the pip installation might interfere with other
Python-based installations on your system. However, if you understand pip
and your Python environment, a native pip installation often entails only
a single command.

Docker completely isolates the TensorFlow installation
from pre-existing packages on your machine. The Docker container contains
TensorFlow and all its dependencies. Note that the Docker image can be quite
large (hundreds of MBs). You might choose the Docker installation if you are
incorporating TensorFlow into a larger application architecture that already
uses Docker.

In Anaconda, you may use conda to create a virtual environment.
However, within Anaconda, we recommend installing TensorFlow with the
`pip install` command, not with the `conda install` command.

**NOTE:** The conda package is community supported, not officially supported.
That is, the TensorFlow team neither tests nor maintains the conda package.
Use that package at your own risk.


<a name="InstallingVirtualenv"></a>
## Installing with virtualenv

Take the following steps to install TensorFlow with Virtualenv:

  1. Install pip and virtualenv by issuing one of the following commands:

     <pre>$ <b>sudo apt-get install python-pip python-dev python-virtualenv</b> # for Python 2.7
     $ <b>sudo apt-get install python3-pip python3-dev python-virtualenv</b> # for Python 3.n</pre>

  2. Create a virtualenv environment by issuing one of the following commands:

     <pre>$ <b>virtualenv --system-site-packages</b> <i>targetDirectory</i> # for Python 2.7
     $ <b>virtualenv --system-site-packages -p python3</b> <i>targetDirectory</i> # for Python 3.n</pre>

     where <code><em>targetDirectory</em></code> specifies the top of the
     virtualenv tree.  Our instructions assume that
     <code><em>targetDirectory</em></code> is `~/tensorflow`, but you may
     choose any directory.

  3. Activate the virtualenv environment by issuing one of the following
     commands:

     <pre>$ <b>source ~/tensorflow/bin/activate</b> # bash, sh, ksh, or zsh
     $ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh</pre>

     The preceding <tt>source</tt> command should change your prompt
     to the following:

     <pre>(tensorflow)$ </pre>

  4. Ensure pip ≥8.1 is installed:

     <pre>(tensorflow)$ <b>easy_install -U pip</b></pre>

  5. Issue one of the following commands to install TensorFlow in the active
     virtualenv environment:

     <pre>(tensorflow)$ <b>pip install --upgrade tensorflow</b>      # for Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade tensorflow</b>     # for Python 3.n
     (tensorflow)$ <b>pip install --upgrade tensorflow-gpu</b>  # for Python 2.7 and GPU
     (tensorflow)$ <b>pip3 install --upgrade tensorflow-gpu</b> # for Python 3.n and GPU</pre>

     If the preceding command succeeds, skip Step 5. If the preceding
     command fails, perform Step 5.

  5. (Optional) If Step 4 failed (typically because you invoked a pip version
     lower than 8.1), install TensorFlow in the active virtualenv environment
     by issuing a command of the following format:

     <pre>(tensorflow)$ <b>pip install --upgrade</b> <i>tfBinaryURL</i>   # Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade</b> <i>tfBinaryURL</i>  # Python 3.n </pre>

     where <code><em>tfBinaryURL</em></code> identifies the URL of the
     TensorFlow Python package. The appropriate value of
     <code><em>tfBinaryURL</em></code>depends on the operating system,
     Python version, and GPU support. Find the appropriate value for
     <code><em>tfBinaryURL</em></code> for your system
     [here](#the_url_of_the_tensorflow_python_package).  For example, if you
     are installing TensorFlow for Linux, Python 2.7, and CPU-only support,
     issue the following command to install TensorFlow in the active
     virtualenv environment:

     <pre>(tensorflow)$ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp34-cp34m-linux_x86_64.whl</b></pre>

If you encounter installation problems, see
[Common Installation Problems](#common_installation_problems).


### Next Steps

After installing TensorFlow,
[validate the installation](#ValidateYourInstallation).

Note that you must activate the virtualenv environment each time you
use TensorFlow. If the virtualenv environment is not currently active,
invoke one of the following commands:

<pre>$ <b>source ~/tensorflow/bin/activate</b>      # bash, sh, ksh, or zsh
$ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh</pre>

When the virtualenv environment is active, you may run
TensorFlow programs from this shell.  Your prompt will become
the following to indicate that your tensorflow environment is active:

<pre>(tensorflow)$ </pre>

When you are done using TensorFlow, you may deactivate the
environment by invoking the `deactivate` function as follows:

<pre>(tensorflow)$ <b>deactivate</b> </pre>

The prompt will revert back to your default prompt (as defined by the
`PS1` environment variable).


### Uninstalling TensorFlow

To uninstall TensorFlow, simply remove the tree you created.
For example:

<pre>$ <b>rm -r</b> <i>targetDirectory</i> </pre>


<a name="InstallingNativePip"></a>
## Installing with native pip

You may install TensorFlow through pip, choosing between a simple
installation procedure or a more complex one.

**Note:** The
[REQUIRED_PACKAGES section of setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)
lists the TensorFlow packages that pip will install or upgrade.


### Prerequisite: Python and Pip

Python is automatically installed on Ubuntu.  Take a moment to confirm
(by issuing a `python -V` command) that one of the following Python
versions is already installed on your system:

  * Python 2.7
  * Python 3.3+

The pip or pip3 package manager is *usually* installed on Ubuntu.  Take a
moment to confirm (by issuing a `pip -V` or `pip3 -V` command)
that pip or pip3 is installed.  We strongly recommend version 8.1 or higher
of pip or pip3.  If Version 8.1 or later is not installed, issue the
following command, which will either install or upgrade to the latest
pip version:

<pre>$ <b>sudo apt-get install python-pip python-dev</b>   # for Python 2.7
$ <b>sudo apt-get install python3-pip python3-dev</b> # for Python 3.n
</pre>


### Install TensorFlow

Assuming the prerequisite software is installed on your Linux host,
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

     <pre>$ <b>sudo pip  install --upgrade</b> <i>tfBinaryURL</i>   # Python 2.7
     $ <b>sudo pip3 install --upgrade</b> <i>tfBinaryURL</i>   # Python 3.n </pre>

     where <code><em>tfBinaryURL</em></code> identifies the URL of the
     TensorFlow Python package. The appropriate value of
     <code><em>tfBinaryURL</em></code> depends on the operating system,
     Python version, and GPU support. Find the appropriate value for
     <code><em>tfBinaryURL</em></code>
     [here](#the_url_of_the_tensorflow_python_package).  For example, to
     install TensorFlow for Linux, Python 2.7, and CPU-only support, issue
     the following command:

     <pre>
     $ <b>sudo pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp34-cp34m-linux_x86_64.whl</b>
     </pre>

     If this step fails, see
     [Common Installation Problems](#common_installation_problems).


### Next Steps

After installing TensorFlow, [validate your installation](#ValidateYourInstallation).


### Uninstalling TensorFlow

To uninstall TensorFlow, issue one of following commands:

<pre>
$ <b>sudo pip uninstall tensorflow</b>  # for Python 2.7
$ <b>sudo pip3 uninstall tensorflow</b> # for Python 3.n
</pre>


<a name="InstallingDocker"></a>
## Installing with Docker

Take the following steps to install TensorFlow through Docker:

  1. Install Docker on your machine as described in the
     [Docker documentation](http://docs.docker.com/engine/installation/).
  2. Optionally, create a Linux group called <code>docker</code> to allow
     launching containers without sudo as described in the
     [Docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/).
     (If you don't do this step, you'll have to use sudo each time
     you invoke Docker.)
  3. To install a version of TensorFlow that supports GPUs, you must first
     install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), which
     is stored in github.
  4. Launch a Docker container that contains one of the
     [TensorFlow binary images](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

The remainder of this section explains how to launch a Docker container.


### CPU-only

To launch a Docker container with CPU-only support (that is, without
GPU support), enter a command of the following format:

<pre>
$ docker run -it <i>-p hostPort:containerPort TensorFlowCPUImage</i>
</pre>

where:

  * <tt><i>-p hostPort:containerPort</i></tt> is optional.
    If you plan to run TensorFlow programs from the shell, omit this option.
    If you plan to run TensorFlow programs as Jupyter notebooks, set both
    <tt><i>hostPort</i></tt> and <tt><i>containerPort</i></tt>
    to <tt>8888</tt>.  If you'd like to run TensorBoard inside the container,
    add a second `-p` flag, setting both <i>hostPort</i> and <i>containerPort</i>
    to 6006.
  * <tt><i>TensorFlowCPUImage</i></tt> is required. It identifies the Docker
    container. Specify one of the following values:
    * <tt>gcr.io/tensorflow/tensorflow</tt>, which is the TensorFlow CPU binary image.
    * <tt>gcr.io/tensorflow/tensorflow:latest-devel</tt>, which is the latest
      TensorFlow CPU Binary image plus source code.
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i></tt>, which is the
      specified version (for example, 1.1.0rc1) of TensorFlow CPU binary image.
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-devel</tt>, which is
      the specified version (for example, 1.1.0rc1) of the TensorFlow GPU
      binary image plus source code.

    <tt>gcr.io</tt> is the Google Container Registry. Note that some
    TensorFlow images are also available at
    [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/).

For example, the following command launches the latest TensorFlow CPU binary image
in a Docker container from which you can run TensorFlow programs in a shell:

<pre>
$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b>
</pre>

The following command also launches the latest TensorFlow CPU binary image in a
Docker container. However, in this Docker container, you can run TensorFlow
programs in a Jupyter notebook:

<pre>
$ <b>docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow</b>
</pre>

Docker will download the TensorFlow binary image the first time you launch it.


### GPU support

Prior to installing TensorFlow with GPU support, ensure that your system meets all
[NVIDIA software requirements](#NVIDIARequirements).  To launch a Docker container
with NVidia GPU support, enter a command of the following format:

<pre>
$ <b>nvidia-docker run -it</b> <i>-p hostPort:containerPort TensorFlowGPUImage</i>
</pre>

where:

  * <tt><i>-p hostPort:containerPort</i></tt> is optional. If you plan
    to run TensorFlow programs from the shell, omit this option. If you plan
    to run TensorFlow programs as Jupyter notebooks, set both
    <tt><i>hostPort</i></tt> and <code><em>containerPort</em></code> to `8888`.
  * <i>TensorFlowGPUImage</i> specifies the Docker container. You must
    specify one of the following values:
    * <tt>gcr.io/tensorflow/tensorflow:latest-gpu</tt>, which is the latest
      TensorFlow GPU binary image.
    * <tt>gcr.io/tensorflow/tensorflow:latest-devel-gpu</tt>, which is
      the latest TensorFlow GPU Binary image plus source code.
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-gpu</tt>, which is the
      specified version (for example, 0.12.1) of the TensorFlow GPU
      binary image.
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-devel-gpu</tt>, which is
      the specified version (for example, 0.12.1) of the TensorFlow GPU
      binary image plus source code.

We recommend installing one of the `latest` versions. For example, the
following command launches the latest TensorFlow GPU binary image in a
Docker container from which you can run TensorFlow programs in a shell:

<pre>
$ <b>nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash</b>
</pre>

The following command also launches the latest TensorFlow GPU binary image
in a Docker container. In this Docker container, you can run TensorFlow
programs in a Jupyter notebook:

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu</b>
</pre>

The following command installs an older TensorFlow version (0.12.1):

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:0.12.1-gpu</b>
</pre>

Docker will download the TensorFlow binary image the first time you launch it.
For more details see the
[TensorFlow docker readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker).


### Next Steps

You should now
[validate your installation](#ValidateYourInstallation).


<a name="InstallingAnaconda"></a>
## Installing with Anaconda

Take the following steps to install TensorFlow in an Anaconda environment:

  1. Follow the instructions on the
     [Anaconda download site](https://www.continuum.io/downloads)
     to download and install Anaconda.

  2. Create a conda environment named <tt>tensorflow</tt> to run a version
     of Python by invoking the following command:

     <pre>$ <b>conda create -n tensorflow</b></pre>

  3. Activate the conda environment by issuing the following command:

     <pre>$ <b>source activate tensorflow</b>
     (tensorflow)$  # Your prompt should change </pre>

  4. Issue a command of the following format to install
     TensorFlow inside your conda environment:

     <pre>(tensorflow)$ <b>pip install --ignore-installed --upgrade</b> <i>tfBinaryURL</i></pre>

     where <code><em>tfBinaryURL</em></code> is the
     [URL of the TensorFlow Python package](#the_url_of_the_tensorflow_python_package).
     For example, the following command installs the CPU-only version of
     TensorFlow for Python 2.7:

     <pre>
     (tensorflow)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp34-cp34m-linux_x86_64.whl</b></pre>


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

If you installed through Docker, start a Docker container
from which you can run bash. For example:

<pre>
$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b>
</pre>


### Run a short TensorFlow program

Invoke python from your shell as follows:

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

If you are new to TensorFlow, see @{$get_started/get_started$Getting Started with TensorFlow}.

If the system outputs an error message instead of a greeting, see [Common
installation problems](#common_installation_problems).

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
  <td><a href="http://stackoverflow.com/q/36371137">36371137</a> and
  <a href="#Protobuf31">here</a></td>
  <td><pre>libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
  protocol message was rejected because it was too big (more than 67108864 bytes).
  To increase the limit (or to disable these warnings), see
  CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/35252888">35252888</a></td>
  <td><pre>Error importing tensorflow. Unless you are using bazel, you should
  not try to import tensorflow from its source directory; please exit the
  tensorflow source tree, and relaunch your python interpreter from
  there.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><pre>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></pre>
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
  <td><a href="http://stackoverflow.com/questions/36933958">36933958</a></td>
  <td><pre>
  ...
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/
   Versions/2.7/Extras/lib/python/_markerlib'</pre>
  </td>
</tr>

</table>


<a name="TF_PYTHON_URL"></a>
## The URL of the TensorFlow Python package

A few installation mechanisms require the URL of the TensorFlow Python package.
The value you specify depends on three factors:

  * operating system
  * Python version
  * CPU only vs. GPU support

This section documents the relevant values for Linux installations.


### Python 2.7

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp27-none-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp27-none-linux_x86_64.whl
</pre>

Note that GPU support requires the NVIDIA hardware and software described in
[NVIDIA requirements to run TensorFlow with GPU support](#NVIDIARequirements).


### Python 3.4

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp34-cp34m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp34-cp34m-linux_x86_64.whl
</pre>

Note that GPU support requires the NVIDIA hardware and software described in
[NVIDIA requirements to run TensorFlow with GPU support](#NVIDIARequirements).


### Python 3.5

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp35-cp35m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp35-cp35m-linux_x86_64.whl
</pre>


Note that GPU support requires the NVIDIA hardware and software described in
[NVIDIA requirements to run TensorFlow with GPU support](#NVIDIARequirements).

### Python 3.6

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc2-cp36-cp36m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp36-cp36m-linux_x86_64.whl
</pre>


Note that GPU support requires the NVIDIA hardware and software described in
[NVIDIA requirements to run TensorFlow with GPU support](#NVIDIARequirements).

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

  <pre>
  $ <b>pip install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp27-none-linux_x86_64.whl</b></pre>

  * for Python 3.5:

  <pre>
  $ <b>pip3 install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp35-none-linux_x86_64.whl</b></pre>

Installing this protobuf package will overwrite the existing protobuf package.
Note that the binary pip package already has support for protobufs
larger than 64MB, which should fix errors such as these:

<pre>[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207]
A protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</pre>
