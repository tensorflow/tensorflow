# Info

To use CMSIS-NN optimized kernels instead of reference kernel add TAGS=cmsis-nn
to the make line. Some micro architectures have optimizations (M4 or higher),
others don't. The kernels that doesn't have optimization for a certain micro
architecture fallback to use TFLu reference kernels.

The optimizations are almost exclusively made for int8 (symmetric) model. For
more details, please read
[CMSIS-NN doc](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/README.md)

# Example 1

```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn
TARGET=apollo3evb person_detection_bin
```

# Example 2 - MBED

```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn
generate_person_detection_mbed_project
```

Go into the generated project's mbed folder.

Note: Mbed has a dependency to an old version of arm_math.h. Therefore you need
to copy the newer version as follows:

```
cp tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/DSP/Include/
arm_math.h mbed-os/cmsis/TARGET_CORTEX_M/arm_math.h
```

This issue will be resolved soon. Now type

```
mbed new .
mbed compile -m DISCO_F746NG -DARM_MATH_LOOPUNROLL
```

Note: ARM_MATH_LOOPUNROLL requirement will be removed
