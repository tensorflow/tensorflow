# Info

To use CMSIS-NN optimized kernels instead of reference kernel add TAGS=cmsis-nn
to the make line. Some micro architectures have optimizations (M4 or higher),
others don't. The kernels that doesn't have optimization for a certain micro
architecture fallback to use TFLu reference kernels.

The optimizations are almost exclusively made for int8 (symmetric) model. For
more details, please read
[CMSIS-NN doc](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/README.md)

# Example 1

A simple way to compile a binary with CMSIS-NN optimizations.

```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn \
TARGET=sparkfun_edge person_detection_int8_bin
```

# Example 2 - MBED

Using mbed you'll be able to compile for the many different targets supported by
mbed. Here's an example on how to do that. Start by generating an mbed project.

```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn \
generate_person_detection_mbed_project
```

Go into the generated mbed project folder, currently:

```
tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/person_detection_int8/mbed
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
tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include/\
cmsis_gcc.h mbed-os/cmsis/TARGET_CORTEX_M/cmsis_gcc.h
```

This issue will be resolved soon.

Now type:

```
mbed compile -m DISCO_F746NG -t GCC_ARM
```

and that gives you a binary for the DISCO_F746NG with CMSIS-NN optimized
kernels.
