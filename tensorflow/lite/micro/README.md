<!-- mdformat off(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
   * [TensorFlow Lite for Microcontrollers](#tensorflow-lite-for-microcontrollers)
   * [Continuous Build Status](#continuous-build-status)
      * [Official Builds](#official-builds)
      * [Community Supported Builds](#community-supported-builds)
   * [Getting Help and Involved](#getting-help-and-involved)
   * [Additional Documentation](#additional-documentation)

<!-- Added by: advaitjain, at: Mon 23 Nov 2020 03:32:57 PM PST -->

<!--te-->

# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is a port of TensorFlow Lite designed to
run machine learning models on microcontrollers and other devices with only
kilobytes of memory.

To learn how to use the framework, visit the developer documentation at
[tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers).

# Continuous Build Status

## Official Builds
Build Type | Status      | Artifacts
---------- | ----------- | ---------
Linux      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/tflite-micro.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/tflite-micro.html) |

## Community Supported Builds
Build Type | Status      | Artifacts
---------- | ----------- | ---------
Arduino    | [![Status](https://github.com/antmicro/tensorflow-arduino-examples/actions/workflows/test_examples.yml/badge.svg)](https://github.com/antmicro/tensorflow-arduino-examples/actions/workflows/test_examples.yml) |
Xtensa     | [![Status](https://github.com/advaitjain/tensorflow/blob/local-continuous-builds/tensorflow/lite/micro/docs/local_continuous_builds/xtensa-build-status.svg)](https://github.com/advaitjain/tensorflow/tree/local-continuous-builds/tensorflow/lite/micro/docs/local_continuous_builds/xtensa.md) |


# Getting Help and Involved

A
[TF Lite Micro Github issue](https://github.com/tensorflow/tensorflow/issues/new?labels=comp%3Amicro&template=70-tflite-micro-issue.md)
should be the primary method of getting in touch with the TensorFlow Lite Micro
(TFLM) team.

The following resources may also be useful:

1.  SIG Micro [email group](https://groups.google.com/a/tensorflow.org/g/micro)
    and
    [monthly meetings](http://doc/1YHq9rmhrOUdcZnrEnVCWvd87s2wQbq4z17HbeRl-DBc).

1.  SIG Micro [gitter chat room](https://gitter.im/tensorflow/sig-micro).

If you are interested in contributing code to TensorFlow Lite for
Microcontrollers then please read our [contributions guide](CONTRIBUTING.md).

# Additional Documentation

For developers that are interested in more details of the internals of the
project, we have additional documentation in the [docs](docs/) folder.

*   [Benchmarks](benchmarks/README.md)
*   [Profiling](docs/profiling.md)
*   [Memory Management](docs/memory_management.md)
*   [Optimized Kernel Implementations](docs/optimized_kernel_implementations.md)
*   [New Platform Support](docs/new_platform_support.md)
*   [Software Emulation with Renode](docs/renode.md)
*   [Pre-allocated tensors](docs/preallocated_tensors.md)
