# Info

To use Ethos-U kernel add TAGS="ethos-u" to the make line.
A tflite file compiled by ARM's offline tool Vela is required for it to work.
Armclang 6.14 is required as compiler as well.

Vela example workflow
---------------------
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
    +---------+
    | softmax |
    |         |
    +----|----+
         |
         | tensor2
         |
         v

Note that ethousu_init() need to be called once during startup.

(TODO: Add link to driver readme.) __FPU_PRESENT need to be set in target makefile.


# Example 1

Clone driver repo from here (this will soon be available).

```
https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-driver
```

Copy it to the downloads folder like this. This step will be done automatically once
https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-driver is up & running.
It will be added to third_party downloads.inc.

```
cp -rp  /path/to/driver/* ./tensorflow/lite/micro/tools/make/downloads/ethosu

```

Then to compile a binary with Ethos-U kernel.

```
make -f tensorflow/lite/micro/tools/make/Makefile network_tester_test TAGS="ethos-u" \
TARGET=<ethos_u_enabled_target> NETWORK_MODEL=<ethos_u_enabled_tflite>
```

