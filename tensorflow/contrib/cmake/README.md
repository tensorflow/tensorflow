This directory contains *CMake* files that can be used to build TensorFlow
core library.


Current Status
--------------

CMake build is not yet ready for general usage!

We are actively working on CMake support. Please help us improve it.
Pull requests are welcomed!


Linux CMake + Docker (very simple)
----------------------------------

```bash
git clone --recursive https://github.com/tensorflow/tensorflow.git
cd tensorflow
tensorflow/tools/ci_build/ci_build.sh CPU tensorflow/tools/ci_build/builds/cmake.sh
```

That's it. Dependencies included. Otherwise read the rest of this readme...


Prerequisites
=============

You need to have [CMake](http://www.cmake.org) and [Git](http://git-scm.com)
installed on your computer before proceeding.

Most of the instructions will be given to the *Сommand Prompt*, but the same
actions can be performed using appropriate GUI tools.


Environment Setup
=================

Open the appropriate *Command Prompt* from the *Start* menu.

For example *VS2013 x64 Native Tools Command Prompt*:

    C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64>

Change to your working directory:

    C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64>cd C:\Path\to
    C:\Path\to>

Where *C:\Path\to* is the path to your real working directory.

Create a folder where Tensorflow headers/libraries/binaries will be installed
after they are built:

    C:\Path\to>mkdir install

If *cmake* command is not available from *Command Prompt*, add it to system
*PATH* variable:

    C:\Path\to>set PATH=%PATH%;C:\Program Files (x86)\CMake\bin

If *git* command is not available from *Command Prompt*, add it to system
*PATH* variable:

    C:\Path\to>set PATH=%PATH%;C:\Program Files\Git\cmd

Good. Now you are ready to continue.

Getting Sources
===============

You can get the latest stable source packages from the
[releases](https://github.com/tensorflow/tensorflow/releases) page.
Or you can type:

    C:\Path\to> git clone --recursive -b [release_tag] https://github.com/tensorflow/tensorflow.git

Where *[release_tag]* is a git tag like *v0.6.0* or a branch name like *master*
if you want to get the latest code.

Go to the project folder:

    C:\Path\to>cd tensorflow
    C:\Path\to\tensorflow>

Now go to *tensorflow\contrib\cmake* folder in Tensorflow's contrib sources:

    C:\Path\to\tensorflow>cd tensorflow\contrib\cmake
    C:\Path\to\tensorflow\tensorflow\contrib\cmake>

Good. Now you are ready to configure *CMake*.

CMake Configuration
===================

*CMake* supports a lot of different
[generators](http://www.cmake.org/cmake/help/latest/manual/cmake-generators.7.html)
for various native build systems. We are only interested in
[Makefile](http://www.cmake.org/cmake/help/latest/manual/cmake-generators.7.html#makefile-generators)
and
[Visual Studio](http://www.cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators)
generators.

We will use shadow building to separate the temporary files from the Tensorflow
source code.

Create a temporary *build* folder and change your working directory to it:

     C:\Path\to\tensorflow\tensorflow\contrib\cmake>mkdir build & cd build
     C:\Path\to\tensorflow\tensorflow\contrib\cmake\build>

The *Makefile* generator can build the project in only one configuration, so
you need to build a separate folder for each configuration.

To start using a *Release* configuration:

     [...]\contrib\cmake\build>mkdir release & cd release
     [...]\contrib\cmake\build\release>cmake -G "NMake Makefiles" ^
     -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_INSTALL_PREFIX=../../../../../../install ^
     ../..

It will generate *nmake* *Makefile* in current directory.

To use *Debug* configuration:

     [...]\contrib\cmake\build>mkdir debug & cd debug
     [...]\contrib\cmake\build\debug>cmake -G "NMake Makefiles" ^
     -DCMAKE_BUILD_TYPE=Debug ^
     -DCMAKE_INSTALL_PREFIX=../../../../../../install ^
     ../..

It will generate *nmake* *Makefile* in current directory.

To create *Visual Studio* solution file:

     [...]\contrib\cmake\build>mkdir solution & cd solution
     [...]\contrib\cmake\build\solution>cmake -G "Visual Studio 12 2013 Win64" ^
     -DCMAKE_INSTALL_PREFIX=../../../../../../install ^
     ../..

It will generate *Visual Studio* solution file *tensorflow.sln* in current
directory.

If the *gmock* directory does not exist, and/or you do not want to build
Tensorflow unit tests, you need to add *cmake* command argument
`-Dtensorflow_BUILD_TESTS=OFF` to disable testing.

Compiling
=========

To compile tensorflow:

     [...]\contrib\cmake\build\release>nmake

or

     [...]\contrib\cmake\build\debug>nmake

And wait for the compilation to finish.

If you prefer to use the IDE:

  * Open the generated tensorflow.sln file in Microsoft Visual Studio.
  * Choose "Debug" or "Release" configuration as desired.
  * From the Build menu, choose "Build Solution".

And wait for the compilation to finish.

Testing
=======

To run unit-tests:

     [...]\contrib\cmake\build\release>nmake check

or

     [...]\contrib\cmake\build\debug>nmake check

You can also build project *check* from Visual Studio solution.
Yes, it may sound strange, but it works.

You should see an output similar to:

     Running main() from gmock_main.cc
     [==========] Running 1546 tests from 165 test cases.
     
     ...
     
     [==========] 1546 tests from 165 test cases ran. (2529 ms total)
     [  PASSED  ] 1546 tests.

To run specific tests:

     C:\Path\to\tensorflow>tensorflow\contrib\cmake\build\release\tests.exe ^
     --gtest_filter=AnyTest*
     Running main() from gmock_main.cc
     Note: Google Test filter = AnyTest*
     [==========] Running 3 tests from 1 test case.
     [----------] Global test environment set-up.
     [----------] 3 tests from AnyTest
     [ RUN      ] AnyTest.TestPackAndUnpack
     [       OK ] AnyTest.TestPackAndUnpack (0 ms)
     [ RUN      ] AnyTest.TestPackAndUnpackAny
     [       OK ] AnyTest.TestPackAndUnpackAny (0 ms)
     [ RUN      ] AnyTest.TestIs
     [       OK ] AnyTest.TestIs (0 ms)
     [----------] 3 tests from AnyTest (1 ms total)
     
     [----------] Global test environment tear-down
     [==========] 3 tests from 1 test case ran. (2 ms total)
     [  PASSED  ] 3 tests.

Note that the tests must be run from the source folder.

If all tests are passed, safely continue.

Installing
==========

To install Tensorflow to the specified *install* folder:

     [...]\contrib\cmake\build\release>nmake install

or

     [...]\contrib\cmake\build\debug>nmake install

You can also build project *INSTALL* from Visual Studio solution.
It sounds not so strange and it works.

This will create the following folders under the *install* location:
  * bin - that contains tensorflow binaries;
  * include - that contains C++ headers and Tensorflow *.proto files;
  * lib - that contains linking libraries and *CMake* configuration files for
    *tensorflow* package.

Now you can if needed:
  * Copy the contents of the include directory to wherever you want to put
    headers.
  * Copy binaries wherever you put build tools (probably somewhere in your
    PATH).
  * Copy linking libraries libtensorflow[d].lib wherever you put libraries.

To avoid conflicts between the MSVC debug and release runtime libraries, when
compiling a debug build of your application, you may need to link against a
debug build of libtensorflowd.lib with "d" postfix.  Similarly, release builds
should link against release libtensorflow.lib library.

DLLs vs. static linking
=======================

Static linking is now the default for the Tensorflow Buffer libraries.  Due to
issues with Win32's use of a separate heap for each DLL, as well as binary
compatibility issues between different versions of MSVC's STL library, it is
recommended that you use static linkage only.  However, it is possible to
build libtensorflow as DLLs if you really want.  To do this, do the following:

  * Add an additional flag `-Dtensorflow_BUILD_SHARED_LIBS=ON` when invoking
    cmake
  * Follow the same steps as described in the above section.
  * When compiling your project, make sure to `#define TENSORFLOW_USE_DLLS`.

When distributing your software to end users, we strongly recommend that you
do NOT install libtensorflow.dll to any shared location.
Instead, keep these libraries next to your binaries, in your application's
own install directory.  C++ makes it very difficult to maintain binary
compatibility between releases, so it is likely that future versions of these
libraries will *not* be usable as drop-in replacements.

If your project is itself a DLL intended for use by third-party software, we
recommend that you do NOT expose Tensorflow objects in your library's
public interface, and that you statically link them into your library.

Notes on Compiler Warnings
==========================

The following warnings have been disabled while building the tensorflow
libraries and binaries.  You may have to disable some of them in your own
project as well, or live with them.

* [TODO]
