# TFLite iOS evaluation app.

## Description

An iOS app to evaluate TFLite models. This is mainly for running different
evaluation tasks on iOS. Right now it only supports evaluating the inference
difference between cpu and delegates.

The app reads evaluation parameters from a JSON file named
`evaluation_params.json` in its `evaluation_data` directory. Any downloaded
models for evaluation should also be placed in `evaluation_data` directory.

The JSON file specifies the name of the model file and other evaluation
parameters like number of iterations, number of threads, delegate name. The
default values in the JSON file are for the Mobilenet_v2_1.0_224 model
([tflite&pb][mobilenet-model]).

## Building / running the app

*   Follow the [iOS build instructions][build-ios] to configure the Bazel
    workspace and `.bazelrc` file correctly.

*   Run `build_evaluation_framework.sh` script to build the evaluation
    framework. This script will build the evaluation framework targeting iOS
    arm64 and put it under `TFLiteEvaluation/TFLiteEvaluation/Frameworks`
    directory.

*   Update evaluation parameters in `evaluation_params.json`.

*   Change `Build Phases -> Copy Bundle Resources` and add the model file to the
    resources that need to be copied.

*   Ensure that `Build Phases -> Link Binary With Library` contains the
    `Accelerate framework` and `TensorFlowLiteInferenceDiffC.framework`.

*   Now try running the app. The app has a single button that runs the
    evaluation on the model and displays results in a text view below. You can
    also see the console output section in your Xcode to see more detailed
    information.

[build-ios]: https://tensorflow.org/lite/guide/build_ios
[mobilenet-model]: https://github.com/tensorflow/tflite-support/raw/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite
[mobilenet-paper]: https://arxiv.org/pdf/1704.04861.pdf