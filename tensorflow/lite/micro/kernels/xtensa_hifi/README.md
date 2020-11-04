# Building TensorFlow Lite for Microcontrollers for Cadence Tensilica HiFi DSPs

This document describes the steps to build and run the Tensorflow Lite Micro on
the Cadence HiFi DSPs.

## Pre-requisites

The Xtensa development tools and the target processor configurations should be
installed on the system. Please check [https://tensilicatools.com] for more
information about downloading and installing the required tools.

Set environment variable XTENSA_BASE to point to the Xtensa developer tools 
installation directory.
(Please refer tensorflow/lite/micro/tools/make/targets/xtensa_hifi_makefile.inc) 

## Optimized HiFi NN Library support  
  
Two HiFi NN Libraries are integerated into TensorFlow Lite for Microcontrollers:
HiFi 4 NN Library
HiFi 5 NN Library

These libraries contain HiFi optimized implementation of various low level 
NN kernels.
The folder, tensorflow/lite/micro/kernels/xtensa_hifi, contains the integration
code for using the HiFi NN Libraries.
  
HiFi NN Library support is available for following HiFi DSPs:
Using HiFi 4 NN Library:
  HiFi 4 
  HiFi 3Z
  Fusion F1
Using HiFi 5 NN Library:
  HiFi 5
  

## Building for HiFi Processors

To build the code using Xtensa tools for the processor configuration selected by
XTENSA_CORE, set TARGET=xtensa_hifi. Additionally TARGET_ARCH can be used to
select optimized HiFi NN kernels specific to the processor configuration.

For HiFi 4, HiFi 3z, Fusion F1 DSPs:
setenv XTENSA_BASE <XtDevTools>
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa_hifi TARGET_ARCH=hifi4 XTENSA_TOOLS_VERSION=RI.2019.2 XTENSA_CORE=<CORE> <target>

For HiFi 5 DSP:
setenv XTENSA_BASE <XtDevTools>
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa_hifi TARGET_ARCH=hifi5 XTENSA_TOOLS_VERSION=RI.2019.2 XTENSA_CORE=<CORE> <target>

A scratch memory allocation is needed for the HiFi optimized kernels. This
allocation is currently done on stack and it's size can be controlled by
defining 'XTENSA_NNLIB_MAX_SCRATCH_SIZE' appropriately in the file
'tensorflow/lite/micro/tools/make/ext_libs/xtensa_hifi_nn_library.inc

