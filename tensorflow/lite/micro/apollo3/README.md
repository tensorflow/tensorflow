# TensorFlow Lite Micro Support for Ambiq Apollo3

## Maintainers

*   [SparkFun Electronics](https://github.com/sparkfun) 
*   [rambiqmicro](https://github.com/rambiqmicro)
*   [oclyke](https://github.com/oclyke)

## Introduction

[Ambiq Micro's](https://ambiq.com/) [Apollo3](https://ambiq.com/apollo3-blue/) MCU is an ultra-low power, highly integrated microcontroller platform based on Ambiq’s patented Subthreshold Power Optimized Technology. Several supported boards, which are now consolidated in this directory, rely on the Apollo3.

## Usage

```make -f tensorflow/lite/micro/tools/make/Makefile TARGET=apollo3 BOARD=${board_id}```

## Boards

board_id | description
---------|------------
apollo3evb | Ambiq Apollo3 evaluation board
sparkfun_edge | Apollo3-based coin-cell powered demo board by SparkFun Electronics

## License

TensorFlow's code is covered by the Apache2 License included in the repository,
and third party dependencies are covered by their respective licenses, in the
third_party folder of this package.