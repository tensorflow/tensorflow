# TensorFlow Security Advisories

[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)

We regularly publish security advisories about using TensorFlow.

*Note*: In conjunction with these security advisories, we strongly encourage
TensorFlow users to read and understand TensorFlow's security model as outlined
in [SECURITY.md](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md).

| Advisory Number | Type               | Versions affected | Reported by           | Additional Information      |
|-----------------|--------------------|:-----------------:|-----------------------|-----------------------------|
| [TFSA-2020-001](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2020-001.md)   | Segmentation fault when converting a Python string to `tf.float16` | >= 12.0, <= 2.1 | (found internally) |  |
| [TFSA-2019-002](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2019-002.md)   | Heap buffer overflow in `UnsortedSegmentSum` | <= 1.14 | (found internally) |  |
| [TFSA-2019-001](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2019-001.md)   | Null Pointer Dereference Error in Decoding GIF Files | <= 1.12 | Baidu Security Lab |  |
| [TFSA-2018-006](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-006.md)   | Crafted Configuration File results in Invalid Memory Access | <= 1.7 | Blade Team of Tencent |  |
| [TFSA-2018-005](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-005.md)   | Old Snappy Library Usage Resulting in Memcpy Parameter Overlap | <= 1.7 | Blade Team of Tencent |  |
| [TFSA-2018-004](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-004.md)   | Checkpoint Meta File Out-of-Bounds Read | <= 1.7 | Blade Team of Tencent |  |
| [TFSA-2018-003](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-003.md)   | TensorFlow Lite TOCO FlatBuffer Parsing Vulnerability | <= 1.7 | Blade Team of Tencent |  |
| [TFSA-2018-002](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-002.md)   | GIF File Parsing Null Pointer Dereference Error | <= 1.5 | Blade Team of Tencent |  |
| [TFSA-2018-001](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2018-001.md)   | BMP File Parser Out-of-bounds Read | <= 1.6 | Blade Team of Tencent |  |
| -               | Out Of Bounds Read |             <= 1.4 | Blade Team of Tencent | [issue report](https://github.com/tensorflow/tensorflow/issues/14959) |

