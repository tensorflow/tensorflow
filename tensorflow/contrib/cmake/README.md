TensorFlow CMake build
======================

CMAKE build is deprecated for TensorFlow. Please use `bazel` to build TF for all
platforms. For details, see the
[TensorFlow install guide](https://www.tensorflow.org/install/).

This directory contains CMake files for building TensorFlow on Microsoft
Windows. [CMake](https://cmake.org) is a cross-platform tool that can
generate build scripts for multiple build systems, including Microsoft
Visual Studio.

**N.B.** We provide Linux build instructions primarily for the purpose of
testing the build. We recommend using the standard Bazel-based build on
Linux.

Current Status
--------------

CMake can be used to build TensorFlow on Windows. See the [getting started documentation](https://www.tensorflow.org/install/source_windows)
for instructions on how to install a pre-built TensorFlow package on Windows.

### Current known limitations
* It is not possible to load a custom Op library.
* GCS file system is not supported.

## Building with CMake

The CMake files in this directory can build the core TensorFlow runtime, an
example C++ binary, and a PIP package containing the runtime and Python
bindings.

### Prerequisites

* CMake version 3.5 or later.

* [Git](https://git-scm.com)

* [SWIG](http://www.swig.org/download.html)

* Additional prerequisites for Microsoft Windows:
  - Visual Studio 2015
  - Python 3.5

* Additional prerequisites for Linux:
  - Python 2.7 or later
  - [Docker](https://www.docker.com/) (for automated testing)

* Python dependencies:
  - wheel
  - NumPy 1.11.0 or later

### Known-good configurations

* Microsoft Windows 10
  - Microsoft Visual Studio Enterprise 2015 with Visual C++ 2015
  - [Anaconda 4.1.1 (Python 3.5 64-bit)](https://www.anaconda.com/download/)
  - [Git for Windows version 2.9.2.windows.1](https://git-scm.com/download/win)
  - [swigwin-3.0.10](http://www.swig.org/download.html)
  - [NVidia CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
  - [NVidia CUDNN 5.1](https://developer.nvidia.com/cudnn)
  - [CMake 3.6](https://cmake.org/files/v3.6/cmake-3.6.3-win64-x64.msi)

* Ubuntu 14.04
  - Makefile generator
  - Docker 1.9.1 (for automated testing)

### Current known limitations
  - The Python package supports **Python 3.5 only**, because that is the only
    version for which standard Python binaries exist and those binaries are
    compatible with the TensorFlow runtime. (On Windows, the standard Python
    binaries for versions earlier than 3.5 were compiled with older compilers
    that do not have all of the features (e.g. C++11 support) needed to compile
    TensorFlow. We welcome patches for making TensorFlow work with Python 2.7
    on Windows, but have not yet committed to supporting that configuration.)

  - The following Python APIs are not currently implemented:
    * Loading custom op libraries via `tf.load_op_library()`. In order to use your
      custom op, please put the source code under the tensorflow/core/user_ops
      directory, and a shape function is required (not optional) for each op.
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

1. Install the prerequisites detailed above, and set up your environment.

   * The following commands assume that you are using the Windows Command
     Prompt (`cmd.exe`). You will need to set up your environment to use the
     appropriate toolchain, i.e. the 64-bit tools. (Some of the binary targets
     we will build are too large for the 32-bit tools, and they will fail with
     out-of-memory errors.) The typical command to do set up your
     environment is:

     ```
     D:\temp> "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvarsall.bat"
     ```

   * When building with GPU support after installing the CUDNN zip file from NVidia, append its
     bin directory to your PATH environment variable.
     In case TensorFlow fails to find the CUDA dll's during initialization, check your PATH environment variable.
     It should contain the directory of the CUDA dlls and the directory of the CUDNN dll.
     For example:

     ```
     D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
     D:\local\cuda\bin
     ```

   * When building with MKL support after installing [MKL](https://software.intel.com/en-us/mkl) from INTEL, append its bin directories to your PATH environment variable.

     In case TensorFlow fails to find the MKL dll's during initialization, check your PATH environment variable.
     It should contain the directory of the MKL dlls. For example:

     ```
     D:\Tools\IntelSWTools\compilers_and_libraries\windows\redist\intel64\mkl
     D:\Tools\IntelSWTools\compilers_and_libraries\windows\redist\intel64\compiler
     D:\Tools\IntelSWTools\compilers_and_libraries\windows\redist\intel64\tbb\vc_mt
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
   To build with GPU support add "^" at the end of the last line above following with:
   ```
   More? -Dtensorflow_ENABLE_GPU=ON ^
   More? -DCUDNN_HOME="D:\...\cudnn"
   ```
   To build with MKL support add "^" at the end of the last line above following with:

   ```
   More? -Dtensorflow_ENABLE_MKL_SUPPORT=ON ^
   More? -DMKL_HOME="D:\...\compilers_and_libraries"
   ```

   To enable SIMD instructions with MSVC, as AVX and SSE, define it as follows:

   ```
   More? -Dtensorflow_WIN_CPU_SIMD_OPTIONS=/arch:AVX
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

   * `-Dtensorflow_ENABLE_GPU=(ON|OFF)`. Defaults to `OFF`. Include
     GPU support. If GPU is enabled you need to install the CUDA 8.0 Toolkit and CUDNN 5.1.
     CMake will expect the location of CUDNN in -DCUDNN_HOME=path_you_unzipped_cudnn.

   * `-Dtensorflow_BUILD_CC_TESTS=(ON|OFF)`. Defaults to `OFF`. This builds cc unit tests.
     There are many of them and building will take a few hours.
     After cmake, build and execute the tests with
     ```
     MSBuild /p:Configuration=RelWithDebInfo ALL_BUILD.vcxproj
     ctest -C RelWithDebInfo
     ```

   * `-Dtensorflow_BUILD_PYTHON_TESTS=(ON|OFF)`. Defaults to `OFF`. This enables python kernel tests.
     After building the python wheel, you need to install the new wheel before running the tests.
     To execute the tests, use
     ```
     ctest -C RelWithDebInfo
     ```

   * `-Dtensorflow_BUILD_MORE_PYTHON_TESTS=(ON|OFF)`. Defaults to `OFF`. This enables python tests on
     serveral major packages. This option is only valid if this and tensorflow_BUILD_PYTHON_TESTS are both set as `ON`.
     After building the python wheel, you need to install the new wheel before running the tests.
     To execute the tests, use
     ```
     ctest -C RelWithDebInfo
     ```

   * `-Dtensorflow_ENABLE_MKL_SUPPORT=(ON|OFF)`. Defaults to `OFF`. Include MKL support. If MKL is enabled you need to install the [Intel Math Kernal Library](https://software.intel.com/en-us/mkl).
     CMake will expect the location of MKL in -MKL_HOME=path_you_install_mkl.

   * `-Dtensorflow_ENABLE_MKLDNN_SUPPORT=(ON|OFF)`. Defaults to `OFF`. Include MKL DNN support. MKL DNN is [Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](https://github.com/intel/mkl-dnn). You have to add `-Dtensorflow_ENABLE_MKL_SUPPORT=ON` before including MKL DNN support.


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
