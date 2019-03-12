# Performance benchmarks

This document lists TensorFlow Lite performance benchmarks when running well
known models on some Android and iOS devices.

These performance benchmark numbers were generated with the
[Android TFLite benchmark binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
and the [iOS benchmark app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios).

# Android performance benchmarks

For Android benchmarks, the CPU affinity is set to use big cores on the device to
reduce variance (see [details](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android)).

It assumes that models were download and unzipped to the
`/data/local/tmp/tflite_models` directory. The benchmark binary is built
using [these instructions](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android)
and assumed in the `/data/local/tmp` directory.

To run the benchmark:

```
adb shell taskset ${CPU_MASK} /data/local/tmp/benchmark_model \
  --num_threads=1 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50 \
  --use_nnapi=false
```

Here, `${GRAPH}` is the name of model and `${CPU_MASK}` is the CPU affinity
chosen according to the following table:

Device | CPU_MASK |
-------| ----------
Pixel 2 | f0 |
Pixel xl | 0c |

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Device </th>
      <th>Mean inference time</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 2>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td>Pixel 2 </td>
    <td>123.3 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>113.3 ms</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>
    </td>
    <td>Pixel 2 </td>
    <td>65.4 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>74.6 ms</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
    </td>
    <td>Pixel 2 </td>
    <td>273.8 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>210.8 ms</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
    </td>
    <td>Pixel 2 </td>
    <td>234.0 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>158.0 ms</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
    </td>
    <td>Pixel 2 </td>
    <td>2846.0 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>1973.0 ms </td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
    </td>
    <td>Pixel 2 </td>
    <td>3180.0 ms</td>
  </tr>
   <tr>
     <td>Pixel XL </td>
     <td>2262.0 ms</td>
  </tr>

 </table>

# iOS benchmarks

To run iOS benchmarks, the [benchmark
app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)
was modified to include the appropriate model and `benchmark_params.json` was
modified  to set `num_threads` to 1.

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Device </th>
      <th>Mean inference time</th>
    </tr>
  </thead>
  <tr>
    <td>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td>iPhone 8 </td>
    <td>32.2 ms</td>
  </tr>
  <tr>
    <td>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>
    </td>
    <td>iPhone 8 </td>
    <td>24.4 ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
    </td>
    <td>iPhone 8 </td>
    <td>60.3 ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
    </td>
    <td>iPhone 8 </td>
    <td>44.3</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
    </td>
    <td>iPhone 8</td>
    <td>562.4 ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
    </td>
    <td>iPhone 8 </td>
    <td>661.0 ms</td>
  </tr>
 </table>
