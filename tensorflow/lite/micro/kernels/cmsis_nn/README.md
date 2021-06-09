<!-- mdformat off(b/169948621#comment2) -->

# Info
CMSIS-NN is a library containing kernel optimizations for Arm(R) Cortex(TM)-M
processors. To use CMSIS-NN optimized kernels instead of reference kernels, add
`OPTIMIZED_KERNEL_DIR=cmsis_nn` to the make command line. See examples below.

For more information about the optimizations, check out
[CMSIS-NN documentation](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/README.md)

# Example 1

A simple way to compile a binary with CMSIS-NN optimizations.

```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn \
TARGET=sparkfun_edge person_detection_int8_bin
```

# Example 2 - MBED

Using mbed you'll be able to compile for the many different targets supported by
mbed. Here's an example on how to do that. Start by generating an mbed project.

```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn \
generate_person_detection_int8_mbed_project
```

Go into the generated mbed project folder, currently:

```
tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/prj/person_detection_int8/mbed
```

and setup mbed.

```
mbed new .
```

Note: Mbed has a dependency to an old version of arm_math.h. Therefore you need
to copy the newer version as follows:

```
cp tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/DSP/Include/\
arm_math.h mbed-os/cmsis/TARGET_CORTEX_M/arm_math.h
```

There's also a dependency to an old cmsis_gcc.h, which you can fix with the
following:

```
cp tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include/\
cmsis_gcc.h mbed-os/cmsis/TARGET_CORTEX_M/cmsis_gcc.h
```

This issue will be resolved soon.

Now type:

```
mbed compile -m DISCO_F746NG -t GCC_ARM
```

and that gives you a binary for the DISCO_F746NG with CMSIS-NN optimized
kernels.
