# TFLite for Microcontrollers Benchmarks

These benchmarks are for measuring the performance of key models and workloads.
They are meant to be used as part of the model optimization process for a given
platform.

## Table of contents

-   [Keyword Benchmark](#keyword-benchmark)
-   [Run on x86](#run-on-x86)
-   [Run on Xtensa XPG Simulator](#run-on-xtensa-xpg-simulator)

## Keyword benchmark

The keyword benchmark contains a model for keyword detection with scrambled
weights and biases.  This model is meant to test performance on a platform only.
Since the weights are scrambled, the output is meaningless. In order to validate
the accuracy of optimized kernels, please run the kernel tests.

## Run on x86

To run the keyword benchmark on x86, run
```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=posix test_keyword_benchmark
```

## Run on Xtensa XPG Simulator

To run the keyword benchmark on the Xtensa XPG simulator, you will need a valid
Xtensa toolchain and license.  With these set up, run:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa-xpg XTENSA_CORE=<xtensa core>  TAGS=xtensa_hifimini test_keyword_benchmark -j18
```
