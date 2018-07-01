# Installing TensorFlow on Raspbian

This guide explains how to install TensorFlow on a Raspberry Pi running
Raspbian. Although these instructions might also work on other Pi variants, we
have only tested (and we only support) these instructions on machines meeting
the following requirements:

*   Raspberry Pi devices running Raspbian 9.0 or higher

## Determine how to install TensorFlow

You must pick the mechanism by which you install TensorFlow. The supported
choices are as follows:

*   "Native" pip.
*   Cross-compiling from sources.

**We recommend pip installation.**

## Installing with native pip

We have uploaded the TensorFlow binaries to piwheels.org. Therefore, you can
install TensorFlow through pip.

The [REQUIRED_PACKAGES section of
setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)
lists the packages that pip will install or upgrade.

### Prerequisite: Python

In order to install TensorFlow, your system must contain one of the following
Python versions:

*   Python 2.7
*   Python 3.4+

If your system does not already have one of the preceding Python versions,
[install](https://wiki.python.org/moin/BeginnersGuide/Download) it now. It
should already be included when Raspbian was installed though, so no extra steps
should be needed.

### Prerequisite: pip

[Pip](https://en.wikipedia.org/wiki/Pip_\(package_manager\)) installs and
manages software packages written in Python. If you intend to install with
native pip, then one of the following flavors of pip must be installed on your
system:

*   `pip3`, for Python 3.n (preferred).
*   `pip`, for Python 2.7.

`pip` or `pip3` was probably installed on your system when you installed Python.
To determine whether pip or pip3 is actually installed on your system, issue one
of the following commands:

<pre>$ <b>pip3 -V</b> # for Python 3.n
$ <b>pip -V</b>  # for Python 2.7</pre>

If it gives the error "Command not found", then the package has not been
installed yet. To install if for the first time, run:

<pre>$ sudo apt-get install python3-pip # for Python 3.n
sudo apt-get install python-pip # for Python 2.7</pre>

You can find more help on installing and upgrading pip in
[the Raspberry Pi documentation](https://www.raspberrypi.org/documentation/linux/software/python.md).

### Prerequisite: Atlas

[Atlas](http://math-atlas.sourceforge.net/) is a linear algebra library that
numpy depends on, and so needs to be installed before TensorFlow. To add it to
your system, run the following command:

<pre>$ sudo apt install libatlas-base-dev</pre>

### Install TensorFlow

Assuming the prerequisite software is installed on your Pi, install TensorFlow
by invoking **one** of the following commands:

     <pre> $ <b>pip3 install tensorflow</b>     # Python 3.n
     $ <b>pip install tensorflow</b>      # Python 2.7</pre>

This can take some time on certain platforms like the Pi Zero, where some Python
packages like scipy that TensorFlow depends on need to be compiled before the
installation can complete. The Python 3 version will typically be faster to
install because piwheels.org has pre-built versions of the dependencies 
available, so this is our recommended option.

### Next Steps

After installing TensorFlow, [validate your
installation](#ValidateYourInstallation) to confirm that the installation worked
properly.

### Uninstalling TensorFlow

To uninstall TensorFlow, issue one of following commands:

<pre>$ <b>pip uninstall tensorflow</b>
$ <b>pip3 uninstall tensorflow</b> </pre>

## Cross-compiling from sources

Cross-compilation means building on a different machine than than you'll be
deploying on. Since Raspberry Pi's only have limited RAM and comparatively slow
processors, and TensorFlow has a large amount of source code to compile, it's
easier to use a MacOS or Linux desktop or laptop to handle the build process.
Because it can take over 24 hours to build on a Pi, and requires external swap
space to cope with the memory shortage, we recommend using cross-compilation if
you do need to compile TensorFlow from source. To make the dependency management
process easier, we also recommend using Docker to help simplify building.

Note that we provide well-tested, pre-built TensorFlow binaries for Raspbian
systems. So, don't build a TensorFlow binary yourself unless you are very
comfortable building complex packages from source and dealing with the
inevitable aftermath should things not go exactly as documented

### Prerequisite: Docker

Install Docker on your machine as described in the [Docker
documentation](https://docs.docker.com/engine/installation/#/on-macos-and-windows).

### Clone the TensorFlow repository

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

### Build from source

To compile TensorFlow and produce a binary pip can install, do the following:

1.  Start a terminal.
2.  Navigate to the directory containing the tensorflow source code.
3.  Run a command to cross-compile the library, for example:

<pre>$ CI_DOCKER_EXTRA_PARAMS="-e CI_BUILD_PYTHON=python3 -e CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.4" \
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
 </pre>

This will build a pip .whl file for Python 3.4, with Arm v7 instructions that
will only work on the Pi models 2 or 3. These NEON instructions are required for
the fastest operation on those devices, but you can build a library that will
run across all Pi devices by passing `PI_ONE` at the end of the command line.
You can also target Python 2.7 by omitting the initial docker parameters. Here's
an example of building for Python 2.7 and Raspberry Pi model Zero or One
devices:

<pre>$ tensorflow/tools/ci_build/ci_build.sh PI tensorflow/tools/ci_build/pi/build_raspberry_pi.sh PI_ONE</pre>

This will take some time to complete, typically twenty or thirty minutes, and
should produce a .whl file in an output-artifacts sub-folder inside your source
tree at the end. This wheel file can be installed through pip or pip3 (depending
on your Python version) by copying it to a Raspberry Pi and running a terminal
command like this (with the name of your actual file substituted):

<pre>$ pip3 install tensorflow-1.9.0-cp34-none-linux_armv7l.whl</pre>

### Troubleshooting the build

The build script uses Docker internally to create a Linux virtual machine to
handle the compilation. If you do have problems running the script, first check
that you're able to run Docker tests like `docker run hello-world` on your
system.

If you're building from the latest development branch, try syncing to an older
version that's known to work, for example release 1.9, with a command like this:

<pre>$ <b>git checkout r1.0</b></pre>

<a name="ValidateYourInstallation"></a>

## Validate your installation

To validate your TensorFlow installation, do the following:

1.  Ensure that your environment is prepared to run TensorFlow programs.
2.  Run a short TensorFlow program.

### Prepare your environment

If you installed on native pip, Virtualenv, or Anaconda, then do the following:

1.  Start a terminal.
2.  If you installed TensorFlow source code, navigate to any directory *except*
    one containing TensorFlow source code.

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

If you're running with Python 3.5, you may see a warning when you first import
TensorFlow. This is not an error, and TensorFlow should continue to run with no
problems, despite the log message.

If the system outputs an error message instead of a greeting, see [Common
installation problems](#common_installation_problems).

To learn more, see the [TensorFlow tutorials](./tutorials/).

## Common installation problems

We are relying on Stack Overflow to document TensorFlow installation problems
and their remedies. The following table contains links to Stack Overflow answers
for some common installation problems. If you encounter an error message or
other installation problem not listed in the following table, search for it on
Stack Overflow. If Stack Overflow doesn't show the error message, ask a new
question about it on Stack Overflow and specify the `tensorflow` tag.

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>


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


</table>
