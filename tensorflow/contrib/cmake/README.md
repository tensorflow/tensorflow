TensorFlow CMake build
======================

This directory contains CMake files for building TensorFlow on Microsoft
Windows. [CMake](https://cmake.org) is a cross-platform tool that can
generate build scripts for multiple build systems, including Microsoft
Visual Studio.

**N.B.** We provide Linux build instructions primarily for the purpose of
testing the build. We recommend using the standard Bazel-based build on
Linux.

Current Status
--------------

The CMake files in this directory can build the core TensorFlow runtime, an
example C++ binary, and a PIP package containing the runtime and Python
bindings. Currently, only CPU builds are supported, but we are working on
providing a GPU build as well.

Note: Windows support is in an **alpha** state, and we welcome your feedback.

### Pre-requisites

* CMake version 3.1 or later

* [Git](http://git-scm.com)

* [SWIG](http://www.swig.org/download.html)

* Additional pre-requisites for Microsoft Windows:
  - Visual Studio 2015
  - Python 3.5
  - NumPy 1.11.0 or later

* Additional pre-requisites for Linux:
  - Python 2.7 or later
  - [Docker](https://www.docker.com/) (for automated testing)
  - NumPy 1.11.0 or later

### Known-good configurations

* Microsoft Windows 10
  - Microsoft Visual Studio Enterprise 2015 with Visual C++ 2015
  - [Anaconda 4.1.1 (Python 3.5 64-bit)](https://www.continuum.io/downloads)
  - [Git for Windows version 2.9.2.windows.1](https://git-scm.com/download/win)
  - [swigwin-3.0.10](http://www.swig.org/download.html)

* Ubuntu 14.04
  - Makefile generator
  - Docker 1.9.1 (for automated testing)

### Current known limitations

* CPU support only

  - We are in the process of porting the GPU code in
    `tensorflow/stream_executor` to build with CMake and work on non-POSIX
    platforms.

* Additional limitations for the Windows build:

  - The Python package supports **Python 3.5 only**, because that is the only
    version for which standard Python binaries exist and those binaries are
    compatible with the TensorFlow runtime. (On Windows, the standard Python
    binaries for versions earlier than 3.5 were compiled with older compilers
    that do not have all of the features (e.g. C++11 support) needed to compile
    TensorFlow. We welcome patches for making TensorFlow work with Python 2.7
    on Windows, but have not yet committed to supporting that configuration.)

  - The following Python APIs are not currently implemented:
    * Loading custom op libraries via `tf.load_op_library()`.
    * Path manipulation functions (such as `tf.gfile.ListDirectory()`) are not
      functional.

  - The `tf.contrib` libraries are not currently included in the PIP package.

  - The following operations are not currently implemented:
    * `DepthwiseConv2dNative`
    * `Digamma`
    * `Erf`
    * `Erfc`
    * `Igamma`
    * `Igammac`
    * `ImmutableConst`
    * `Lgamma`
    * `Polygamma`
    * `SparseMatmul`
    * `Zeta`

  - Google Cloud Storage support is not currently implemented. The GCS library
    currently depends on `libcurl` and `boringssl`, and the Windows version
    could use standard Windows APIs for making HTTP requests and cryptography
    (for OAuth). Contributions are welcome for this feature.

We are actively working on improving CMake and Windows support, and addressing
these limitations. We would appreciate pull requests that implement missing
ops or APIs.


Step-by-step Windows build
==========================

1. Install the pre-requisites detailed above, and set up your environment.

   * The following commands assume that you are using the Windows Command
     Prompt (`cmd.exe`). You will need to set up your environment to use the
     appropriate toolchain, i.e. the 64-bit tools. (Some of the binary targets
     we will build are too large for the 32-bit tools, and they will fail with
     out-of-memory errors.) The typical command to do set up your
     environment is:

     ```
     D:\temp> "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvarsall.bat"
     ```

   * We assume that `cmake` and `git` are installed and in your `%PATH%`. If
     for example `cmake` is not in your path and it is installed in
     `C:\Program Files (x86)\CMake\bin\cmake.exe`, you can add this directory
     to your `%PATH%` as follows:

     ```
     D:\temp> set PATH="%PATH%;C:\Program Files (x86)\CMake\bin\cmake.exe"
     ```

2. Clone the TensorFlow repository and create a working directory for your
   build:

   ```
   D:\temp> git clone https://github.com/tensorflow/tensorflow.git
   D:\temp> cd tensorflow\tensorflow\contrib\cmake
   D:\temp\tensorflow\tensorflow\contrib\cmake> mkdir build
   D:\temp\tensorflow\tensorflow\contrib\cmake> cd build
   D:\temp\tensorflow\tensorflow\contrib\cmake\build>
   ```

3. Invoke CMake to create Visual Studio solution and project files.

   **N.B.** This assumes that `cmake.exe` is in your `%PATH%` environment
   variable. The other paths are for illustrative purposes only, and may
   be different on your platform. The `^` character is a line continuation
   and must be the last character on each line.

   ```
   D:\...\build> cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release ^
   More? -DSWIG_EXECUTABLE=C:/tools/swigwin-3.0.10/swig.exe ^
   More? -DPYTHON_EXECUTABLE=C:/Users/%USERNAME%/AppData/Local/Continuum/Anaconda3/python.exe ^
   More? -DPYTHON_LIBRARIES=C:/Users/%USERNAME%/AppData/Local/Continuum/Anaconda3/libs/python35.lib
   ```

   Note that the `-DCMAKE_BUILD_TYPE=Release` flag must match the build
   configuration that you choose when invoking `msbuild`. The known-good
   values are `Release` and `RelWithDebInfo`. The `Debug` build type is
   not currently supported, because it relies on a `Debug` library for
   Python (`python35d.lib`) that is not distributed by default.

   There are various options that can be specified when generating the
   solution and project files:

   * `-DCMAKE_BUILD_TYPE=(Release|RelWithDebInfo)`: Note that the
     `CMAKE_BUILD_TYPE` option must match the build configuration that you
     choose when invoking MSBuild in step 4. The known-good values are
     `Release` and `RelWithDebInfo`. The `Debug` build type is not currently
     supported, because it relies on a `Debug` library for Python
     (`python35d.lib`) that is not distributed by default.

   * `-Dtensorflow_BUILD_ALL_KERNELS=(ON|OFF)`. Defaults to `ON`. You can
     build a small subset of the kernels for a faster build by setting this
     option to `OFF`.

   * `-Dtensorflow_BUILD_CC_EXAMPLE=(ON|OFF)`. Defaults to `ON`. Generate
     project files for a simple C++
     [example training program](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc).

   * `-Dtensorflow_BUILD_PYTHON_BINDINGS=(ON|OFF)`. Defaults to `ON`. Generate
     project files for building a PIP package containing the TensorFlow runtime
     and its Python bindings.

   * `-Dtensorflow_ENABLE_GRPC_SUPPORT=(ON|OFF)`. Defaults to `ON`. Include
     gRPC support and the distributed client and server code in the TensorFlow
     runtime.

   * `-Dtensorflow_ENABLE_SSL_SUPPORT=(ON|OFF)`. Defaults to `OFF`. Include
     SSL support (for making secure HTTP requests) in the TensorFlow runtime.
     This support is incomplete, and will be used for Google Cloud Storage
     support.

4. Invoke MSBuild to build TensorFlow.

   To build the C++ example program, which will be created as a `.exe`
   executable in the subdirectory `.\Release`:

   ```
   D:\...\build> MSBuild /p:Configuration=Release tf_tutorials_example_trainer.vcxproj
   D:\...\build> Release\tf_tutorials_example_trainer.exe
   ```

   To build the PIP package, which will be created as a `.whl` file in the
   subdirectory `.\tf_python\dist`:

   ```
   D:\...\build> MSBuild /p:Configuration=Release tf_python_build_pip_package.vcxproj
   ```


Linux Continuous Integration build
==================================

This build requires [Docker](https://www.docker.com/) to be installed on the
local machine.

```bash
$ git clone --recursive https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ tensorflow/tools/ci_build/ci_build.sh CMAKE tensorflow/tools/ci_build/builds/cmake.sh
```

That's it. Dependencies included.
