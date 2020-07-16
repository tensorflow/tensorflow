# Description of files

*   **create_constants.py**: Python file used to create hanning.cc, hanning.h,
    sin_1k.cc, and sin_1k.h
*   **hanning.cc**: Precomputed
    [Hann window](https://en.wikipedia.org/wiki/Hann_function) for use in the
    preprocessor. This file is created in ../create_constants.py
*   **hanning.h**: Header file for hanning.cc
*   **preprocessor.cc**: CMSIS version of the preprocessor
*   **sin_1k.cc**: A 1 kHZ sinusoid used for comparing the CMSIS preprocessor
    with the Micro-Lite fixed_point preprocessor
*   **sin_1k.h**: Header file for sin_1k.cc

# Description of externally downloaded files in ../CMSIS_ext

*   **arm_cmplx_mag_squared_q10p6.c**: Modified version of the ARM CMSIS
    function
    [arm_cmplx_mag_squared.c](http://arm-software.github.io/CMSIS_5/DSP/html/group__cmplx__mag__squared.html#ga45537f576102d960d467eb722b8431f2).
    The modification is that we have changed the amount of right-shift to make
    sure our data is in the correct range. We redistribute because the original
    content was created with the Apache 2.0 license.
*   **arm_cmplx_mag_squared_q10p6.h**: Header file for
    arm_cmplx_mag_squared_q10p6.c
