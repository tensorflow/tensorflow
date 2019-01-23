# TFLite iOS benchmark app.

## Description

An iOS app to benchmark TFLite models.

The app reads benchmark parameters from a JSON file named `benchmark_params.json`
in its `benchmark_data` directory. Any downloaded models for benchmarking should
also be placed in `benchmark_data` directory.

The JSON file specifies the name of the model file and other benchmarking
parameters like inputs to the model, type of inputs, number of iterations,
number of threads. The default values in the JSON file are for the
Mobilenet_1.0_224 model
([paper](https://arxiv.org/pdf/1704.04861.pdf),
[tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz))

## To build/install/run

- Follow instructions at
[iOS build for TFLite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/ios.md)
to build TFLite.

Running

```bash
tensorflow/lite/tools/make/build_ios_universal_lib.sh
```
will also build `tensorflow/lite/gen/lib/benchmark-lib.a` .

- Now copy the downloaded model file to `benchmark_data` directory. 

- Modify `benchmark_params.json` change the `input_layer`, `input_layer_shape`
and other benchmark parameters.

- Change `Build Phases -> Copy Bundle Resources` and add the model file to the
resources that need to be copied.

- Ensure that `Build Phases -> Link Binary With Library` contains the 
`Accelerate framework` and `tensorflow/lite/gen/lib/benchmark-lib.a`.

- Now try running the app. The app has a single button that runs the benchmark
  on the model and displays results in a text view below.

## Profiling

If you want detailed profiling, use the following command:

```bash
tensorflow/lite/build_ios_universal_lib.sh -p
```

Then following the same steps above and run the benchmark app. You will see the
detailed profiling results in the outputs.
