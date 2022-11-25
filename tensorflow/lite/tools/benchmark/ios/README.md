# TFLite iOS benchmark app.

## Description

An iOS app to benchmark TFLite models.

The app reads benchmark parameters from a JSON file named
`benchmark_params.json` in its `benchmark_data` directory. Any downloaded models
for benchmarking should also be placed in `benchmark_data` directory.

The JSON file specifies the name of the model file and other benchmarking
parameters like inputs to the model, type of inputs, number of iterations,
number of threads. The default values in the JSON file are for the
Mobilenet_1.0_224 model ([paper][mobilenet-paper],
[tflite&pb][mobilenet-model]).

## Building / running the app

*   Follow the [iOS build instructions][build-ios] to configure the Bazel
    workspace and `.bazelrc` file correctly.

*   Run `build_benchmark_framework.sh` script to build the benchmark framework.
    This script will build the benchmark framework targeting iOS arm64 and put
    it under `TFLiteBenchmark/TFLiteBenchmark/Frameworks` directory.

*   If you want more detailed profiling, run the build script with `-p` option:
    `build_benchmark_framework.sh -p`.

*   Modify `benchmark_params.json` change the `input_layer`, `input_layer_shape`
    and other benchmark parameters.

*   Change `Build Phases -> Copy Bundle Resources` and add the model file to the
    resources that need to be copied.

*   Ensure that `Build Phases -> Link Binary With Library` contains the
    `Accelerate framework` and `TensorFlowLiteBenchmarkC.framework`.

*   Now try running the app. The app has a single button that runs the benchmark
    on the model and displays results in a text view below. You can also see the
    console output section in your Xcode to see more detailed benchmark
    information.


[build-ios]: https://tensorflow.org/lite/guide/build_ios
[mobilenet-model]: https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
[mobilenet-paper]: https://arxiv.org/pdf/1704.04861.pdf
