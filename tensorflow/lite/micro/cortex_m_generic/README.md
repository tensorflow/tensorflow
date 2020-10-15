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
   - TOOLCHAIN: armclang or armgcc
   - TARGET: cortex_m_generic
   - TARGET_ARCH: cortex-mXX (For all options see: tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc)

For Cortex-M55, ARM Compiler 6.14 or later is required.

Some examples:
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m55+nofp microlite
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armclang TARGET=cortex_m_generic TARGET_ARCH=cortex-m7 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armgcc TARGET=cortex_m_generic TARGET_ARCH=cortex-m7 microlite
make -f tensorflow/lite/micro/tools/make/Makefile TOOLCHAIN=armgcc TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 microlite

Example of using own CMSIS-NN, instead of the default downloaded one:
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn CMSIS_PATH=/path/to/own/cmsis TOOLCHAIN=armgcc TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 microlite
