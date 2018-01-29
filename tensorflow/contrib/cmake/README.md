TensorFlow CMake build
======================

This directory contains CMake files for building TensorFlow on Microsoft
Windows and Linux. [CMake](https://cmake.org) is a cross-platform tool that can
generate build scripts for multiple build systems, including Microsoft
Visual Studio and GCC. The method suppose could work on MacOS, but not tested yet.

**N.B.** We provide Linux build instructions primarily for the purpose of
testing the build. We recommend using the standard Bazel-based build on
Linux.

Current Status
--------------

CMake can be used to build TensorFlow on all platforms. See the [getting started documentation](https://www.tensorflow.org/install/install_windows)
for instructions on how to install a pre-built TensorFlow package on Windows and Linux. Procedure in MacOS is similar to the Linux build.

### Current known limitations
* It is not possible to load a custom Op library.
* GCS file system is not supported.
* Debug build is not available since python no longer release debug library.

## Building with CMake

The CMake files in this directory can build the core TensorFlow runtime, an
example C++ binary, and a PIP package containing the runtime and Python
bindings.

### Pre-requisites

* CMake version 3.5 or later.

* [Git](https://git-scm.com)

* [SWIG](http://www.swig.org/download.html)

* [Perl](https://www.perl.org/get.html)

* [Go](https://golang.org/)

* [NASM](http://www.nasm.us/)

* Additional pre-requisites for Microsoft Windows:
  - Visual Studio 2015 (latest version of MSVC 2017 is not suppored by CUDA yet, try it on your own)
  - Python 3.5
  - NumPy 1.11.0 or later

* Additional pre-requisites for Linux:
  - Python 2.7 or later
  - [Docker](https://www.docker.com/) (for automated testing)
  - NumPy 1.11.0 or later

### Known-good configurations

* Microsoft Windows 10
  - Microsoft Visual Studio Enterprise/ Community 2015 with Visual C++ 2015
  - [Anaconda 4.1.1 (Python 3.5 64-bit)](https://www.anaconda.com/download/)
  - [Git for Windows version 2.9.2.windows.1](https://git-scm.com/download/win)
  - [swigwin-3.0.10](http://www.swig.org/download.html)
  - [NVidia CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-downloads)
  - [NVidia CUDNN 7](https://developer.nvidia.com/cudnn)
  - [CMake 3.6](https://cmake.org/files/v3.6/cmake-3.6.3-win64-x64.msi)

* Ubuntu 14.04
  - Makefile generator
  - Docker 1.9.1 (for automated testing)

### Current known limitations
  - The Python package supports **Python 3.5/3.6 only**, because these are the only
    versions for which standard Python binaries exist and those binaries are
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

CMake GUI build (all platforms)
==================================
Install from CMake GUI would be a convenient way to generate C++ build projects. The software supports Windows, MacOS and Linux, while the posix platform provides an extra ccmake binary to run command line GUI. Both working principal of cmake, ccmake and cmake-gui are the same, the only difference is by providing suitable interface for project configuration and dependency setting.

0. Pre-buid checklist:
    The following binary/libraries should be setted in system path, otherwise you need to set manualy via cmake.
    * Compiler (GCC for Linux, MSVC for Windows)
    * Make sure compiler directory has been set to system path
    * CUDA 9.0 (GPU build)
    * CUDNN (GPU build)
    * NCCL (GPU build on Linux)
    * SWIG (python binding) 
    * Perl (GPU build on Windows, required by grpc)
    * Go (GPU build on Windows, required by grpc)
    * NASM (GPU build on Windows, required by grpc)
1. Start CMake GUI
2. Click on `Browse Source` and direct to the the folder `<tensorflow-source>/tensorflow/contrib/cmake`
3. Click on `Browse Build` and spectify a location that you want tensorflow to be build
4. Click on `Configure`, a new window will be prompted out, specify the generator mode for the project generation. For Windows, choose `Visual Studio <version> <year> Win64`, for Linux, choose `Unix Makefiles`, then press `Finish`. Wait for a moment, the default project dependecy would automatically generate.
5. There are a few options that you can customize your own build. **The setting here is crucial for a sucessful build, please check all items carefully.**
    * `tensorflow_BUILD_ALL_KERNELS` should alway be `on`
    * `tensorflow_BUILD_CC_EXAMPLE` is default to be `on`. This can help you to test build (optional)
    * `tensorflow_BUILD_CONTRIB_KERNELS` is default to be `on`, but it won't affect tensorflow function, turn it to `off` if you want a slim build. (optional)
    * `tensorflow_BUILD_PYTHON_BINDING` is default to be `on`. Set to `off` if you don't need python interaface. If SWIG is not in system path, you need set it manually. (optional)
    * `tensorflow_BUILD_SHARED_LIB` is default to be `off`. Set to `on` if you want the c++ interface. (optional)
    * `tensorflow_ENABLE_GPU` is default to be `off`. Set to `on` if you want GPU support. It will search CUDA and CUDNN dependecies if you have set them to system path, otherwise CMake would prompt error and request you to set it manually. (optional)
    * `tensorflow_ENABLE_GRPC_SUPPORT` is default to be `on`. For Linux build, this option must always be `on`. In Windows this need to be `on` for gpu build. Reminded that Perl, Go and NASM is required for this option in windows.
    * `tensorflow_ENABLE_POSITION_INDEPENDENT_CODE` should always be `on`
    * `tensorflow_ENABLE_SNAPPY_SUPPORT` should always be `on` 
    * `tensorflow_OPTIMIZE_FOR_NATIVE_ARCH` should always be `on`
    * `CMAKE_INSTALL_PREFIX` is the location where the final package will be installed. You may change it to you own preferred path (optional)

6. After changing the configuration in step 5, press `Configure` again
7. If not error is found, press `Generate`

#### Windows

1. Open `tensorflow.sln` in the build folder (Windows). Change build type from `Debug` to `Release`. Choose `Build`->`Build Solution`. This may take more than hours of compilation. If everything is alright, the output window would show no error.

    ##### Python

    In solution explorer, right click on `tf_python_build_pip_package` -> `build`. It will generate the wheel file in `<tensorflow-build>/tf_python/dist`. Install with following command:

     ```pip install --upgrade tensorflow-<config>.whl```

    ***The wheel name varies depends on you config. Change to your own wheel filename.***

    Remind that some pip installation requires administrator right command prompt.

    ##### C++

    You can directly use the build folder tree for C++ interface with cmake. If you want to do installation for api releasing, right click on `Install` -> `build`. The headers and library will be installed in the directory specify by `CMAKE_INSTALL_PREFIX` during configuration.

2. For smaller RAM computer, it is noticed that out of heap space error appears. Change to command prompt build is an alternative to do step 1. 

    Open `VS2015 x64 Native Tools Command Prompt`. You can open it by press `Start`, then type the binary name. Use `VS2017 x64 Native Tools Command Prompt` if you are using MSVC 2017.

    ##### Python

    Directly build python wheel package by following command:

    ```MSBuild /p:Configuration=Release <path-to-tf_python_build_pip_package.vcxproj>```

    Remember to change `<path-to-tf_python_build_pip_package.vcxproj>` to the actual path of the file, it can be found at the root of build directory

    Install the wheel file generated as instructed by step 1.

    ##### C++ interface
    Build from VS native toolchain with following command:
    ```MSBuild /p:Configuration=Release <path-to-ALL_BUILD.vcxproj>```

    Headers are discretely located in the build folders. Tensorflow library can be found at `<path-to-build>/Release`, namely `tensorflow.dll` and `tensorflow.lib`.

    * Build to install for api release (optional):
    ```MSBuild /p:Configuration=Release <path-to-INSTALL.vcxproj>```

    Remember to change `<path-to-ALL_BUILD.vcxproj>` and `<path-to-INSTALL.vcxproj>` to the actual path of the file, it can be found at the root of build directory.

#### Linux/MacOS (command line GNU build)

1. Open the terminal, change working directory to the one specified in step 3.

2. Type the following command:

    ```make -sj<number-of-threads> all```

    ##### Python

    **Important Note** CMake generated python wheel for Linux/MacOs is currently under development. Please use bazel build.

    Follow code is an expected Linux/MacOS python package build after development work is completed.

    ```
    make -sj<number-of-threads> tf_python_build_pip_package
    cd tf_python
    pip install --upgrade tensorflow-<config>.whl
    ```

    ##### C++ interface

    ```make -sj<number-of-threads> install```

    Where `<number-of-threads>` is the threads used for the compilation, change to any integer less or equal to your computer's maxiumum thread number.

     Headers are discretely located in the build folders. Tensorflow library can be found at `<path-to-build>`, namely `tensorflow.so` (Linux) or `tensorflow.dylib` (MacOS).

#### Start a Tensorflow C++ project with CMake
Here we assume that you have basic knowledge on gathering dependency with `CMakeLists.txt`. Here we introduce how the C++ api works with [official hello world tutorial](https://www.tensorflow.org/api_guides/cc/guide).

1. Create a new working directory and create a new text file named `CMakeLists.txt` and the c++ file `main.cxx`
2. Fill in the `main.cxx` with the code provided in [official c++ api basic](https://www.tensorflow.org/api_guides/cc/guide).
3. Fill in the `CMakeLists.txt` with following code:
    ``` cmake
    cmake_minimum_required (VERSION 2.6)
    project (tf_hello)

    # Tensorflow
    find_package(Tensorflow REQUIRED)
    include_directories(${TENSORFLOW_INCLUDE_DIRS})

    # compiler setting required by tensorflow, to be tested on all compilers
    # currently only tested on MSVC and GCC
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC) 
      add_definitions(-DCOMPILER_MSVC)
    elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "3")
        add_definitions(-DCOMPILER_GCC3)
      else()
        add_definitions(-D__GNUC__)
      endif()
    else()
      message(ERROR " compiler ${CMAKE_CXX_COMPILER_ID} not supported by this CMakeList.txt, under development")
    endif()

    add_executable(tf_hello main.cxx)
    target_link_libraries(tf_hello ${TENSORFLOW_LIBRARIES})
    ```
4. Configure the folder with cmake-gui, an error should be prompted out, requesting you to locate the folder containing `TensorflowConfig.cmake`. This file can be found at `<tensorflow-build>` or `<tensorflow-intall>` (for those have build install in previous steps).

5. Configure again, generate the project.
6. Compile the project with `Release` config (Windows). For Linux users, just compile the project.
7. Copy the `tensorflow.dll`(Windows)/`tensorflow.so`(Linux) from build directory to the build folder containing `tf_hello` binary.
8. Run `tf_hello` binary

Step-by-step Windows build (command prompt)
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

   * When building with GPU support after installing the CUDNN zip file from NVidia, append its
     bin directory to your PATH environment variable.
     In case TensorFlow fails to find the CUDA dll's during initialization, check your PATH environment variable.
     It should contain the directory of the CUDA dlls and the directory of the CUDNN dll.
     For example:

     ```
     D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
     D:\local\cuda\bin
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