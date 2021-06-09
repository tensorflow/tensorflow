<!-- mdformat off(b/169948621#comment2) -->

# Info
Arm(R) Ethos(TM)-U is a new class of machine learning processors, called a
microNPU, specifically designed to accelerate ML inference in area-constrained
embedded and IoT devices. This readme briefly describes how to integrate Ethos-U
related hardware and software into TFLM. See also [Ethos-U ML Evaluation kit examples](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ml-embedded-evaluation-kit).

To enable the Ethos-U software stack, add `CO_PROCESSOR=ethos_u` to the make
command line. See example below.

## Requirements:
- Armclang 6.14 or later
- GCC 10.2.1 or later

## Ethos-U custom operator
The TFLM runtime will dispatch workloads to Ethos-U when it encounters an
Ethos-U custom op in the tflite file. The Ethos-U custom op is added by a tool
called Ethos-U Vela and contains information the Ethos-U hardware need to execute
the workload. More info in the [Vela repo](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela).

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

Note that the `ethousu_init()` API of the Ethos-U driver need to be called at
startup, before calling the TFLM API. More info in the [Ethos-U driver repo](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-core-driver).

For even more info regarding Vela and Ethos-U, checkout [Ethos-U landing page](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u/+/refs/heads/master).

# Some examples of compiling a binary and running a network with Ethos-U support.
In order to run a test with Ethos-U55 enabled, a platform with corresponding hardware support is required. One such platform is the fixed virtual platform (FVP) based on Arm Corstone-300 software. See tensorflow/lite/micro/cortex_m_corstone_300/README.md for more info.

On top of that the .tflite model needs to be modified according subchapter "Ethos-U custom operator" above.

The log level of the Ethos-U driver can be set in the build command. For example: ETHOSU_LOG_SEVERITY=ETHOSU_LOG_INFO.

## Example using network tester
See tensorflow/lite/micro/examples/network_tester/README.md for more info.

```
make -f tensorflow/lite/micro/tools/make/Makefile network_tester_test CO_PROCESSOR=ethos_u TARGET=cortex_m_corstone_300 \
TARGET_ARCH=cortex-m55 test_network_tester_test NETWORK_MODEL=path/to/network_model.h INPUT_DATA=path/to/input_data.h \
OUTPUT_DATA=path/to/expected_output_data.h

make -f tensorflow/lite/micro/tools/make/Makefile network_tester_test CO_PROCESSOR=ethos_u TARGET=cortex_m_corstone_300 \
TARGET_ARCH=cortex-m55 test_network_tester_test
```
