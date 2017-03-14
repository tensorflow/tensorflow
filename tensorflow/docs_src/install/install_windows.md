# Installing TensorFlow on Windows

This guide explains how to install TensorFlow on Windows.

## Determine which TensorFlow to install

You must choose one of the following types of TensorFlow to install:

  * **TensorFlow with CPU support only**. If your system does not have a
    NVIDIA® GPU, you must install this version. Note that this version of
    TensorFlow is typically much easier to install (typically,
    in 5 or 10 minutes), so even if you have an NVIDIA GPU, we recommend
    installing this version first.
  * **TensorFlow with GPU support**. TensorFlow programs typically run
    significantly faster on a GPU than on a CPU. Therefore, if your
    system has a NVIDIA® GPU meeting the prerequisites shown below
    and you need to run performance-critical applications, you should
    ultimately install this version.

### Requirements to run TensorFlow with GPU support

If you are installing TensorFlow with GPU support using one of the mechanisms
described in this guide, then the following NVIDIA software must be
installed on your system:

  * CUDA® Toolkit 8.0. For details, see
    [NVIDIA's
    documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
    Ensure that you append the relevant Cuda pathnames to the `%PATH%`
    environment variable as described in the NVIDIA documentation.
  * The NVIDIA drivers associated with CUDA Toolkit 8.0.
  * cuDNN v5.1. For details, see
    [NVIDIA's documentation](https://developer.nvidia.com/cudnn).
    Note that cuDNN is typically installed in a different location from the
    other CUDA DLLs. Ensure that you add the directory where you installed
    the cuDNN DLL to your `%PATH%` environment variable.
  * GPU card with CUDA Compute Capability 3.0 or higher.  See
    [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) for a
    list of supported GPU cards.

If you have an earlier version of the preceding packages, please
upgrade to the specified versions.


## Determine how to install TensorFlow

You must pick the mechanism by which you install TensorFlow. The
supported choices are as follows:

  * "native" pip
  * Anaconda

Native pip installs TensorFlow directly on your system without going
through a virtual environment.  Since a native pip installation is not
walled-off in a separate container, the pip installation might interfere
with other Python-based installations on your system. However, if you
understand pip and your Python environment, a native pip installation
often entails only a single command! Furthermore, if you install with
native pip, users can run TensorFlow programs from any directory on
the system.

In Anaconda, you may use conda to create a virtual environment.
However, within Anaconda, we recommend installing TensorFlow with the
`pip install` command, not with the `conda install` command.

**NOTE:** The conda package is community supported, not officially supported.
That is, the TensorFlow team neither tests nor maintains this conda package.
Use that package at your own risk.


## Installing with native pip

If the following version of Python is not installed on your machine,
install it now:

  * [Python 3.5.x from python.org](https://www.python.org/downloads/release/python-352/)

TensorFlow only supports version 3.5.x of Python on Windows.
Note that Python 3.5.x comes with the pip3 package manager, which is the
program you'll use to install TensorFlow.

To install TensorFlow, start a terminal. Then issue the appropriate
<tt>pip3 install</tt> command in that terminal.  To install the CPU-only
version of TensorFlow, enter the following command:

<pre>C:\> <b>pip3 install --upgrade tensorflow</b></pre>

To install the GPU version of TensorFlow, enter the following command:

<pre>C:\> <b>pip3 install --upgrade tensorflow-gpu</b></pre>


## Installing with Anaconda

**The Anaconda installation is community supported, not officially supported.**

Take the following steps to install TensorFlow in an Anaconda environment:

  1. Follow the instructions on the
     [Anaconda download site](https://www.continuum.io/downloads)
     to download and install Anaconda.

  2. Create a conda environment named <tt>tensorflow</tt>
     by invoking the following command:

     <pre>C:\> <b>conda create -n tensorflow</b> </pre>

  3. Activate the conda environment by issuing the following command:

     <pre>C:\> <b>activate tensorflow</b>
     (tensorflow)C:\>  # Your prompt should change </pre>

  4. Issue the appropriate command to install TensorFlow inside your conda
     environment. To install the CPU-only version of TensorFlow, enter the
     following command:

     <pre>(tensorflow)C:\> <b>pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl</b> </pre>

     To install the GPU version of TensorFlow, enter the following command
     (on a single line):

     <pre>(tensorflow)C:\> <b>pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl</b> </pre>

## Validate your installation

Start a terminal.

If you installed through Anaconda, activate your Anaconda environment.

Invoke python from your shell as follows:

<pre>$ <b>python</b></pre>

Enter the following short program inside the python interactive shell:

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

If the system outputs the following, then you are ready to begin writing
TensorFlow programs:

<pre>Hello, TensorFlow!</pre>

If you are new to TensorFlow, see @{$get_started$Getting Started with
TensorFlow}.

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
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\dso_loader.cc] Couldn't open CUDA library nvcuda.dll</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\cuda\cuda_dnn.cc] Unable to load cuDNN DSO</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
File "...\tensorflow\core\framework\graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/42011070">42011070</a></td>
  <td><pre>No module named "pywrap_tensorflow"</pre></td>
</tr>

<table>

