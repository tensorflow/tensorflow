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

<!-- Added by: advaitjain, at: Fri 23 Oct 2020 04:40:49 PM PDT -->

<!--te-->

# Software Emulation with Renode

TensorFlow Lite Micro makes use of [Renode](https://github.com/renode/renode) to
for software emulation.

Here, we document how Renode is used as part of the TFLM project. For more
general use of Renode, please refer to the [Renode
documentation](https://renode.readthedocs.io/en/latest/).

# Installation

Renode can be installed and used in a variety of ways, as documented
[here](https://renode.readthedocs.io/en/latest/). For the purpose of Tensorflow
Lite Micro, we make use of a portable version for Linux.

 1. Download portable version of Renode for Linux:

    ```
    tensorflow/lite/micro/testing/download_renode.sh tensorflow/lite/micro/tools/make/downloads/renode
    ```

 2. Install the Renode test dependencies

    ```
    pip3 install -r tensorflow/lite/micro/tools/make/downloads/renode/tests/requirements.txt
    ```

At this point in time you will be ready to run TFLM tests with Renode.

# Running Unit Tests

All the tests for a specific platform (e.g. bluepill) can be run with:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=bluepill test
```

 * This makes use of the robot framework from Renode.
 * Note that the tests can currently not be run in parallel.

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

It may be useful to run binaries on Renode that are not tests, independent of
the robot framework. We will be adding some documentation for that in this
section.
