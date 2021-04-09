 <!-- mdformat off(b/169948621#comment2) -->

# Running a fixed virtual platform based on Corstone-300 software

This target makes use of a fixed virtual platform (FVP) based on Arm Cortex-300
based software. More info about Arm Corstone-300 software:
https://developer.arm.com/ip-products/subsystem/corstone/corstone-300. More info
about FVPs:
https://developer.arm.com/tools-and-software/simulation-models/fixed-virtual-platforms.

To fullfill the needed requirements it is depending the following projects:

-   Arm Ethos-U Core Platform:
    https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-platform.
    -   Arm Ethos-U Core Platform provides the linker file as well as UART and
        retarget functions.
-   CMSIS: https://github.com/ARM-software/CMSIS_5.
    -   CMSIS provides startup functionality, e.g. for setting up interrupt
        handlers and clock speed.

# General build info

This target is based on the cortex_m_generic target and except that for now the
only supported toolchain is GCC, the same general build info applies:
tensorflow/lite/micro/cortex_m_generic/README.md.

Required parameters:

-   TARGET: cortex_m_corstone_300
-   TARGET_ARCH: cortex-mXX (For all options see:
    tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_makefile.inc)

# How to run

Note that Corstone-300 is targetted for Cortex-M55 but it is backwards
compatible. This means one could potentially run it for example with a
Cortex-M7. Note that the clock speed would be that of an Cortex-M55. This may
not matter when running unit tests or for debugging.

Some examples:

```
make -j -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_kernel_fully_connected_test
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_kernel_fully_connected_test
make -j -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m7+fp test_kernel_fully_connected_test
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m3 test_kernel_fully_connected_test
```
