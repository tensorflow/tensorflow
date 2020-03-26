## Inference Diff tool

**NOTE: This is an experimental tool to analyze TensorFlow Lite behavior on
delegates.**

For a given model, this binary compares TensorFlow Lite execution (in terms of
latency & output-value deviation) in two settings:

*   Single-threaded CPU Inference
*   User-defined Inference

To do so, the tool generates random gaussian data and passes it through two
TFLite Interpreters - one running single-threaded CPU kernels and the other
parametrized by the user's arguments.

It measures the latency of both, as well as the absolute difference between the
output tensors from each Interpreter, on a per-element basis.

The final output typically looks like this:

```
num_runs: 50
process_metrics {
  inference_profiler_metrics {
    reference_latency {
      last_us: 43111
      max_us: 49314
      min_us: 42965
      sum_us: 6525771
      avg_us: 43505.14
    }
    test_latency {
      last_us: 26906
      max_us: 107118
      min_us: 26454
      sum_us: 5286197
      avg_us: 35241.313333333332
    }
    output_errors {
      max_value: 0.000999001
      min_value: 0
      avg_value: 1.9980019424110651e-05
      std_deviation: 0.00013986013
    }
  }
}
```

The values in `test_latency` denote the inference latency statistics in
milliseconds. `reference_latency` denotes single-threaded CPU behavior.

There is one instance of `output_errors` for each output tensor in the model,
and the statistics in `output_errors[i]` correspond to the absolute difference
in raw values across all elements for the `i`th output.

## Parameters

(In this section, 'test Interpreter' refers to the User-defined Inference
mentioned above. The reference setting is always single-threaded CPU).

The binary takes the following parameters:

*   `model_file` : `string` \
    Path to the TFlite model file.

*   `output_file_path`: `string` \
    The final metrics are dumped into `output_file_path` as a string-serialized
    instance of `tflite::evaluation::EvaluationStageMetrics`.

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
  --output_file_path=/data/local/tmp/inference_diff.txt \
  --delegate=gpu
```

(5) Pull the results.

```
adb pull /data/local/tmp/inference_diff.txt ~/accuracy_tool
```
