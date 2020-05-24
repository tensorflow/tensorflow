# Building TensorFlow Lite for Microcontrollers for Cadence Tensilica HiFi DSPs

This document describes the steps to build and run the Tensorflow Lite Micro on
the Cadence HiFi DSPs.

## Pre-requisites

The Xtensa development tools and the target processor configurations should be
installed on the system. Please check [https://tensilicatools.com] for more
information about downloading and installing the required tools.

The PATH variable should be set to include the <xtensa_tools_root>/bin
directory. The XTENSA_SYSTEM and XTENSA_CORE environment variables should be set
to the required tools version and the required processor configuration.

## Building for HiFi Processors

To build the code using Xtensa tools for the processor configuration selected by
XTENSA_CORE , set TARGET=xtensa_hifi. Additionally TARGET_ARCH can be used to
select optimized HiFi NN kernels specific to the processor configuration.
Currently the HiFi4 NN kernels are provided which can be enabled as follows:

make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test
TARGET=xtensa_hifi TARGET_ARCH=hifi4

Xtensa specific TF Lite Micro kernels are implemented in this folder:
tensorflow/lite/micro/kernels/xtensa_hifi/

A scratch memory allocation is needed for the HiFi optimized kernels. This
allocation is currently done on stack and it's size can be controlled by
defining 'XTENSA_NNLIB_MAX_SCRATCH_SIZE' approproately in the file
'tensorflow/lite/micro/tools/make/ext_libs/xtensa_hifi_nn_library.inc

The files containing the HiFi optimized NN kernels are present in this folder:
tensorflow/lite/micro/kernels/xtensa_hifi/xa_nnlib/
