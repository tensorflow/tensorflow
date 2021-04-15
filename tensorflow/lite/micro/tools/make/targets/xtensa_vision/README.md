# Building TensorFlow Lite for Microcontrollers for Cadence Tensilica Vision DSPs

This document describes the steps to build and run the Tensorflow Lite Micro on
the Cadence Vision DSPs.

## Pre-requisites

The Xtensa development tools and the target processor configurations should be
installed on the system. Please check [https://tensilicatools.com] for more
information about downloading and installing the required tools.

The PATH variable should be set to include the <xtensa_tools_root>/bin  
directory. The XTENSA_SYSTEM and XTENSA_CORE environment variables should
be set to the required tools version and the required processor configuration.
XTENSA_TOOLS_VERSION should point to tools version.

## Building for Vision Processors

To build the code using Xtensa tools for vision processor configuration set
TARGET=xtensa_vision. Additionally TARGET_ARCH can be used to
select optimized Vision NN kernels specific to the processor configuration.
Currently the Vision P6 kernels are provided which can be enabled as follows:

make -f tensorflow/lite/micro/tools/make/Makefile person_detection_test_int8
TARGET=xtensa_vision TARGET_ARCH=visionp6_ao OPTIMIZED_KERNEL_DIR=xtensa_vision

Xtensa specific TF Lite Micro kernels are implemented in this folder:
tensorflow/lite/micro/kernels/xtensa_vision/

Optimizations done in kernels need additional tensor_arena memory. In
application like person_detection, ensure to increase 'tensor_arena_size'
which is passed to interpreter init call. Size of extra memory will depend
various factors like kernels used, size of filters used in kernels and number of
layers of operation. For uint8 case of person detect using 3 optimized kernel
(pooling, depthwise_conv and conv) for input images size 96x96 needs 216KB extra
memory.

During building the code, files containing the Vision optimized kernels are
downloaded to this folder:
tensorflow/lite/micro/tools/make/downloads/xi_annlib_vision_p6 (for Vision P6)
