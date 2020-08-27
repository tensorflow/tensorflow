# VexRISC-V

## Maintainers

*   [danielyou0230](https://github.com/danielyou0230)
*   [tal-x](https://github.com/tcal-x)

## Background

The optimized kernels for
[VexRISC-V](https://github.com/SpinalHDL/VexRiscv)/[Litex](https://github.com/enjoy-digital/litex)
are used to run Tensorflow Lite Micro in Zephyr on either

*   Digilent Arty board (e.g. Arty A7)
*   [Renode](https://github.com/renode/renode): Open source simulation framework
    (no hardware required)

To run on Digilent Arty board (FPGA,) you'll also need a soft-CPU gateware for
the FPGA, please see
[Tensorflow lite demo running in Zephyr on Litex/VexRiscv SoC](https://github.com/antmicro/litex-vexriscv-tensorflow-lite-demo)
by Antmicro for more details.

## Info

To use VexRISC-V optimized kernels instead of reference kernel add
`TAGS=vexriscv` to the make command. The kernels that doesn't have optimization
for a certain micro architecture fallback to use TFLM reference kernels.

# Example

To compile the binary file with VexRISC-V optimizations, one can use the
following command

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TAGS=vexriscv \
TARGET=zephyr_vexriscv \
person_detection_int8_bin
```
