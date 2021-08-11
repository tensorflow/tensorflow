## Inference Diff tool

**NOTE: This is an experimental tool to analyze TensorFlow Lite behavior on
delegates.**

For a given model, this binary compares TensorFlow Lite execution (in terms of
latency & output-value deviation) in two settings:

*   Single-threaded CPU Inference
*   User-defined Inference

To do so, the tool generates random gaussian data and passes it through two
TFLite Interpreters - one running single-threaded CPU kernels and the other
parameterized by the user's arguments.

It measures the latency of both, as well as the absolute difference between the
output tensors from each Interpreter, on a per-element basis.

The final output (logged to stdout) typically looks like this:

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

There is one instance of `OutputDiff` for each output tensor in the model, and
the statistics in `OutputDiff[i]` correspond to the absolute difference in raw
values across all elements for the `i`th output.

## Parameters

(In this section, 'test Interpreter' refers to the User-defined Inference
mentioned above. The reference setting is always single-threaded CPU).

The binary takes the following parameters:

*   `model_file` : `string` \
    Path to the TFlite model file.

and the following optional parameters:

*   `num_runs`: `int` \
    How many runs to perform to compare execution in reference and test setting.
    Default: 50. The binary performs runs 3 invocations per 'run', to get more
    accurate latency numbers.

*   `num_interpreter_threads`: `int` (default=1) \
    This modifies the number of threads used by the test Interpreter for
    inference.

*   `delegate`: `string` \
    If provided, tries to use the specified delegate on the test Interpreter.
    Valid values: "nnapi", "gpu", "hexagon".

    NOTE: Please refer to the
    [Hexagon delegate documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md)
    for instructions on how to set it up for the Hexagon delegate. The tool
    assumes that `libhexagon_interface.so` and Qualcomm libraries lie in
    `/data/local/tmp`.

*   `output_file_path`: `string` \
    The final metrics are dumped into `output_file_path` as a serialized
    instance of `tflite::evaluation::EvaluationStageMetrics`

This script also supports runtime/delegate arguments introduced by the
[delegate registrar](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates).
If there is any conflict (for example, `num_threads` vs
`num_interpreter_threads` here), the parameters of this
script are given precedence.

When **multiple delegates** are specified to be used in the commandline flags
via the support of delegate registrar, the order of delegates applied to the
TfLite runtime will be same as their enabling commandline flag is specified. For
example, "--use_xnnpack=true --use_gpu=true" means applying the XNNPACK delegate
first, and then the GPU delegate secondly. In comparison,
"--use_gpu=true --use_xnnpack=true" means applying the GPU delegate first, and
then the XNNPACK delegate secondly.

Note, one could specify `--help` when launching the binary to see the full list
of supported arguments.

## Running the binary on Android

(1) Build using the following command:

```
bazel build -c opt \
  --config=android_arm64 \
  //tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval
```

(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):

```
adb push bazel-bin/third_party/tensorflow/lite/tools/evaluation/tasks/inference_diff/run_eval /data/local/tmp
```

(3) Push the TFLite model that you need to test. For example:

```
adb push mobilenet_v1_1.0_224.tflite /data/local/tmp
```

(3) Run the binary.

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_v1_1.0_224.tflite \
  --delegate=gpu
```

(5) Pull the results.

```
adb pull /data/local/tmp/inference_diff.txt ~/accuracy_tool
```
