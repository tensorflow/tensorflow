<!-- mdformat off(b/169948621#comment2) -->

# Generic Cortex-Mx customizations

The customization requires a definition where the debug log goes to. The purpose
of the generic Cortex-Mx target is to generate a TFLM library file for use in
application projects outside of this repo. As the chip HAL and the board
specific layer are only defined in the application project, the TFLM library
cannot write the debug log anywhere. Instead, we allow the application layer to
register a callback function for writing the TFLM kernel debug log.

# Usage

See debug_log_callback.h

# How to build

Required parameters:

  - TARGET: cortex_m_generic
  - TARGET_ARCH: cortex-mXX (For all options see: tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc)

Optional parameters:

  - TOOLCHAIN: gcc (default) or armmclang
  - For Cortex-M55, ARM Compiler 6.14 or later is required.

Some examples:

Building with arm-gcc

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m7 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m7 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite

make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
```

Building with armclang

```
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55+nofp OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
```

The Tensorflow Lite Micro makefiles download a specific version of the arm-gcc
compiler to tensorflow/lite/micro/tools/make/downloads/gcc_embedded.

If desired, a different version can be used by providing `TARGET_TOOLCHAIN_ROOT`
option to the Makefile:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp TARGET_TOOLCHAIN_ROOT=/path/to/arm-gcc/ microlite
```

Similarly, `OPTIMIZED_KERNEL_DIR=cmsis_nn` downloads a specific version of CMSIS to
tensorflow/lite/micro/tools/make/downloads/cmsis. While this is the only version
that is regularly tested, you can use your own version of CMSIS as well by
providing `CMSIS_PATH` to the Makefile:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn CMSIS_PATH=/path/to/own/cmsis microlite
```
