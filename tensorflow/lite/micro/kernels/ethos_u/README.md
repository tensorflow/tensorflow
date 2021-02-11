<!-- mdformat off(b/169948621#comment2) -->

# Info

To use Ethos-U kernel add CO_PROCESSOR=ethos_u to the make line. A tflite file
compiled by ARM's offline tool Vela is required for it to work. Armclang 6.14 is
required as compiler as well.

## Vela example workflow

```
     | tensor0
     |
     v
+------------+
| ethos-u    |
| custom op  |
+------------+
     +
     |
     | tensor1
     |
     v
+-----------+
| transpose |
|           |
+----|------+
     |
     | tensor2
     |
     v
```

Note that ethousu_init() need to be called once during startup.

(TODO: Add link to driver readme.) __FPU_PRESENT need to be set in target
makefile.

# Example 1

Compile a binary with Ethos-U kernel.

```
make -f tensorflow/lite/micro/tools/make/Makefile network_tester_test CO_PROCESSOR=ethos_u \
TARGET=<ethos_u_enabled_target> NETWORK_MODEL=<ethos_u_enabled_tflite>
```
