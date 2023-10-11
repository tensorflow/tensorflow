# TensorFlow Lite Sample Stable Delegate

## Description

An example delegate for stable delegate testing that supports only a few
operations: addition, subtraction, multiplication, equality check, and while
loops.

The sample stable delegate implementation uses the stable delegate API, which is
based around `TfLiteOpaqueDelegate`. `TfLiteOpaqueDelegate` is an opaque version
of `TfLiteDelegate`; which allows delegation of nodes to alternative backends.
This is an abstract type that is intended to have the same role as
`TfLiteDelegate` from
[common.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h),
but without exposing the implementation details of how delegates are
implemented.

`TfLiteOpaqueDelegate`s can be loaded dynamically (see
`sample_stable_delegate_external_test.cc`) and then be supplied to the TFLite
runtime, in the same way as statically linked delegates can.

Note however that open-source TF Lite does not (yet) provide a binary stable
interface between delegates and the TF Lite runtime itself. Therefore any opaque
delegate that is loaded dynamically into TF Lite *must* have been built against
the same version (and commit) that the TF Lite runtime itself has been built at.
Any other configuration can lead to undefined behavior.

## Delegate implementation

The sample stable delegate uses two supporting interfaces
[SimpleOpaqueDelegateInterface and SimpleOpaqueDelegateKernelInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h).
These APIs make it easier to implement an opaque TF Lite delegate, though their
usage is entirely optional.

The `sample_stable_delegate_test` driver (see next section) makes use of the
[TfLiteOpaqueDelegateFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h)
facility, which provides static methods that deal with delegate creation and
deletion.

## Testing

See
[sample_stable_delegate_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_test.cc)
for a standalone test driver that links the sample stable delegate statically
and runs inference on a TF Lite model.

See
[sample_stable_delegate_external_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_external_test.cc)
for a standalone test driver that loads the sample stable delegate dynamically
and runs inference on a TF Lite model.

### Delegate Test Suite

The Delegate Test Suite provides correctness testing for a delegate at the
operation level. It checks whether a delegate produces results that meet the
accuracy thresholds of the supported operations.

Support for stable delegate binaries has been integrated into the Delegate Test
Suite.

#### Run on Android

The following instructions show how to run the test suite on Android.

First, we build the sample stable delegate shared library file,
`libtensorflowlite_sample_stable_delegate.so`, which we will later load
dynamically as part of the test:

```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate

adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so /data/local/tmp
```

Next, we create a configuration file for the component that loads the stable
delegate:

```bash
adb shell 'echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"/data/local/tmp/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"> /data/local/tmp/stable_delegate_settings.json'
```

We create a configuration file for the delegate test suite to verify that the
models in the specified test cases have been delegated:

```bash
adb shell 'echo "
  # The sample stable delegate supports static-sized addition and subtraction operations.
  FloatSubOpModel.NoActivation
  FloatSubOpModel.VariousInputShapes
  FloatAddOpModel.NoActivation
  FloatAddOpModel.VariousInputShapes
"> /data/local/tmp/stable_delegate_acceleration_test_config.json'
```

Then, we build the test suite itself:

```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/stable_delegate:stable_delegate_test_suite

adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_test_suite /data/local/tmp
```

Now, we can execute the test suite with providing the settings file:

```bash
adb shell "/data/local/tmp/stable_delegate_test_suite \
  --stable_delegate_settings_file=/data/local/tmp/stable_delegate_settings.json \
  --acceleration_test_config_path=/data/local/tmp/stable_delegate_acceleration_test_config.json"
```

You can also specify `gunit_filter` to only run a subset of tests. This can be
used to skip non release-blocking test cases (e.g. fp16 precision issues) which
are subject to Android ML team's approval. For example, the following command
would skip TestA, TestB and TestC.

```bash
adb shell "/data/local/tmp/stable_delegate_test_suite \
  --stable_delegate_settings_file=/data/local/tmp/stable_delegate_settings.json \
  --acceleration_test_config_path=/data/local/tmp/stable_delegate_acceleration_test_config.json \
  --gunit_filter=-TestA:TestB:TestC"
```

The test suite will show the following output in console after all tests are
passed:

```
...
[==========] 3338 tests from 349 test suites ran. (24555 ms total)
[  PASSED  ] 3338 tests.
```

### Benchmark Tools

#### Delegate Performance Benchmark app

The
[Delegate Performance Benchmark app](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md)
is the recommended tool to test the latency and accuracy of a stable delegate.

#### TF Lite Benchmark Tool

During early development stages of a new stable delegate it can also be useful
to directly load the delegate's shared library file into TF Lite's
`benchmark_model` tool, because this development workflow works on regular linux
desktop machines and also allows users to benchmark any TF Lite model file they
are interested in.

Support for stable delegate binaries has been integrated into TF Lite's
[`benchmark_model`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
CLI tool. We can use this tool to test the sample stable delegate with a
provided TF Lite model file.

##### A) Run on a regular linux host

The following instructions show how to run the tool on regular desktop linux
machine.

First, we build the sample stable delegate shared library file,
`libtensorflowlite_sample_stable_delegate.so`, which we will later load
dynamically with the `benchmark_model` tool:

```bash
bazel build -c opt //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate
```

Next, we create a configuration file for the component that loads the stable
delegate:

```bash
echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"$(bazel info -c opt bazel-bin)/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"> stable_delegate_settings.json
```

Then, we build the `benchmark_model` tool itself:

```bash
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Now, we can execute the benchmark tool. We provide the settings file together
with a TF Lite file that contains ADD operations. We do this because the sample
stable delegate only support ADD, SUB, MUL, EQUAL, and WHILE:

```bash
$(bazel info -c opt bazel-bin)/tensorflow/lite/tools/benchmark/benchmark_model \
  --stable_delegate_settings_file=$(pwd)/stable_delegate_settings.json \
    --graph=$(pwd)/tensorflow/lite/testdata/add.bin
```

Note that when you make changes to the sample delegate you need to rebuild the
delegate's shared library file, in order for benchmark_model to pick up the new
delegate code.

##### B) Run on Android

The following instructions show how to run the tool on Android.

First, we build the sample stable delegate shared library file,
`libtensorflowlite_sample_stable_delegate.so`, which we will later load
dynamically with the `benchmark_model` tool:

```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate

adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so /data/local/tmp
```

Next, we create a configuration file for the component that loads the stable
delegate:

```bash
adb shell 'echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"/data/local/tmp/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"> /data/local/tmp/stable_delegate_settings.json'
```

Then, we build the `benchmark_model` tool itself:

```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model

adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
```

Now, we can execute the benchmark tool. We provide the settings file together
with a TF Lite file that contains ADD operations. We do this because the sample
stable delegate only support ADD, SUB, MUL, EQUAL, and WHILE:

```bash
adb push tensorflow/lite/testdata/add.bin /data/local/tmp/add.bin
adb shell "/data/local/tmp/benchmark_model \
  --stable_delegate_settings_file=/data/local/tmp/stable_delegate_settings.json \
  --graph=/data/local/tmp/add.bin"
```

## Sample app

To show how to use the sample stable delegate, we have included a sample app
that uses it.  You can build and run the sample app as follows:

```
bazel run -c opt \
    //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:sample_app_using_stable_delegate \
    tensorflow/lite/testdata/add.tflite
```
