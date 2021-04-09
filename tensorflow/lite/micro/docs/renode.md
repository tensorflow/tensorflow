<!-- mdformat off(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
   * [Software Emulation with Renode](#software-emulation-with-renode)
   * [Installation](#installation)
   * [Running Unit Tests](#running-unit-tests)
      * [Under the hood of the Testing Infrastructure](#under-the-hood-of-the-testing-infrastructure)
   * [Running a non-test Binary with Renode](#running-a-non-test-binary-with-renode)
   * [Useful External Links for Renode and Robot Documentation](#useful-external-links-for-renode-and-robot-documentation)

<!-- Added by: advaitjain, at: Tue 10 Nov 2020 09:43:05 AM PST -->

<!--te-->

# Software Emulation with Renode

TensorFlow Lite Micro makes use of [Renode](https://github.com/renode/renode) to
for software emulation.

Here, we document how Renode is used as part of the TFLM project. For more
general use of Renode, please refer to the [Renode
documentation](https://renode.readthedocs.io/en/latest/).

You can also read more about Renode from a [publicly available slide deck](https://docs.google.com/presentation/d/1j0gjI4pVkgF9CWvxaxr5XuCKakEB25YX2n-iFxlYKnE/edit).

# Installation

Renode can be installed and used in a variety of ways, as documented in the
[Renode README](https://github.com/renode/renode/blob/master/README.rst#installation/). For the purpose of Tensorflow
Lite Micro, we make use of the portable version for Linux.

Portable renode will be automatically installed when using the TfLite Micro
Makefile to `tensorflow/lite/micro/tools/make/downloads/renode`.

The Makefile internally calls the `renode_download.sh` script:

```
tensorflow/lite/micro/testing/renode_download.sh tensorflow/lite/micro/tools/make/downloads
```

# Running Unit Tests

All the tests for a specific platform (e.g. bluepill) can be run with:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=bluepill test
```

 * This makes use of the robot framework from Renode.
 * Note that the tests can currently not be run in parallel.
 * It takes about 25 second to complete all tests, including around 3 seconds for suite startup/teardown and average 0.38 second per test.

## Under the hood of the Testing Infrastructure

Describe how we wait for a particular string on the UART. Some pointers into the
robot files as well as any relevant documentation from Renode.

A test failure is the absence of a specific string on the UART so the test will
wait for a specific timeout period (configured in the .robot) file before
failing.

 * What this means in practice is that a failing test will take longer to finish
   than a test that passes.

 * If needed, an optimization on this would be to have a specific failure
   message as well so that both success and failure can be detected quickly.

# Running a non-test Binary with Renode

Renode can also be used to run and debug binaries interactively. For example,
to debug `kernel_addr_test` on Bluepill platform, run Renode:

```
tensorflow/lite/micro/tools/make/downloads/renode/renode
```
and issue following commands:
```
# Create platform
include @tensorflow/lite/micro/testing/bluepill_nontest.resc
# Load ELF file
sysbus LoadELF @tensorflow/lite/micro/tools/make/gen/bluepill_cortex-m3_default/bin/keyword_benchmark
# Start simulation
start

# To run again:
Clear
include @tensorflow/lite/micro/testing/bluepill_nontest.resc
sysbus LoadELF @tensorflow/lite/micro/tools/make/gen/bluepill_cortex-m3_default/bin/keyword_benchmark
start

```

To make repeat runs a bit easier, you can put all the commands into a
single line (up arrow will show the last command in the Renode terminal):
```
Clear; include @tensorflow/lite/micro/testing/bluepill_nontest.resc; sysbus LoadELF @tensorflow/lite/micro/tools/make/gen/bluepill_cortex-m3_default/bin/keyword_benchmark; start
```

You can also connect GDB to the simulation.
To do that, start the GDB server in Renode before issuing the `start` command:
```
machine StartGdbServer 3333
```
Than you can connect from GDB with:
```
target remote localhost:3333
```

For further reference please see the [Renode documentation](https://renode.readthedocs.io/en/latest/).

# Useful External Links for Renode and Robot Documentation

 * [Testing with Renode](https://renode.readthedocs.io/en/latest/introduction/testing.html?highlight=robot#running-the-robot-test-script)

 * [Robot Testing Framework on Github](https://github.com/robotframework/robotframework). For someone new to
   the Robot Framework, the documentation  can be a bit hard to navigate, so
   here are some links that are relevant to the use of the Robot Framework with
   Renode for TFLM:

   * [Creating Test Data](http://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#creating-test-data)
     section of the user guide.

   * Renode-specific additions to the Robot test description format are in the
     [RobotFrameworkEngine directory](https://github.com/renode/renode/tree/master/src/Renode/RobotFrameworkEngine). For example,

       * [Start Emulation](https://github.com/renode/renode/blob/master/src/Renode/RobotFrameworkEngine/RenodeKeywords.cs#L41-L42)
       * [Wait For Line On Uart](https://github.com/renode/renode/blob/master/src/Renode/RobotFrameworkEngine/UartKeywords.cs#L62-L63)
     is where `Wait For Line On Uart` is defined.

   * Some documentation for all the [Standard Libraries](http://robotframework.org/robotframework/#standard-libraries)
     that define commands such as:

       * [Remove File](http://robotframework.org/robotframework/latest/libraries/OperatingSystem.html#Remove%20File)
       * [List Files In Directory](https://robotframework.org/robotframework/latest/libraries/OperatingSystem.html#List%20Files%20In%20Directory)
