When speaking of a TFLite delegate, how to create it and how to reuse existing
TFLite testing and tooling with the new delegate are two major challenging
issues. Here, we show a dummy delegate implementation to illustrate our
recommended approaches to address these issues.

## Delegate Creation

We recommend using
[SimpleDelegateInterface and SimpleDelegateKernelInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h).
We believe such APIs will make it easier to create a TFLite delegate. At a high
level, developers only need to focus on

* Whether a TFLite node in the graph is supported by the delegate or not.
* Given the set of supported nodes (i.e. a subgraph of the original model
graph), implement a delegate kernel that executes this set of nodes.

The dummy delegate implementation here is a good starting point to understand
the ideas above. For more sophisticated examples, refer to [Flex delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex),
    [Hexagon delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/hexagon).

## Testing & Tooling

There are currently **two options** to plug in a newly created TFLite delegate
to reuse existing TFLite kernel tests and tooling:

- Utilize the **[delegate registrar](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)**
mechanism
- Utilize the
**[external delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external)**
mechanism.

The former approach requires few changes as detailed below. The latter one
requires even fewer changes and works with pre-built Tensorflow Lite tooling
binaries. However, it is less explicit and it might be more complicated to set
up in automated integration tests. Therefore, for better clarity, the
delegate-registrar approach is slightly preferred here.

We now describe each option above in more details in the following sections.

### Option 1: Utilize Delegate Registrar
In this approach, create a delegate provider like the
[`dummy_delegate_provider.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate_provider.cc)
here, and then add it as an extra dependency when building the binary. Refer
[here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)
for more delegate provider examples. Now we look at using this provider for
testing and evaluation.

#### Kernel Tests
Tests referred here are defined in [tensorflow/lite/kernels](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels).
They are based on the
 [test_util library](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/test_util.h)
 and the [testing main function stub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/test_main.cc).

To plug in the newly created delegate and reuse these tests, simply add the
created delegate provider as an extra dependency to
[`test_util_delegate_providers`](https://github.com/tensorflow/tensorflow/blob/f09dc5cf6e7fde978f9891638f529cd52a3c878f/tensorflow/lite/kernels/BUILD#L203)
and remove others that are not relevant, like the following:

```
cc_library(
    name = "tflite_driver_delegate_providers",
    deps = [
        # Existing delegate providers that might be still relevant.
        ":dummy_delegate_provider",
    ],
    alwayslink = 1,
)
```

Then build a kernel test, and specify the commandline flags defined in the
delegate provider when executing the test. Take this case as an example,

```
bazel build -c opt tensorflow/lite/kernels:add_test

# Setting --use_dummy_delegate=true will apply the dummy delegate to the
# TFLite model graph
bazel-bin/tensorflow/lite/kernels/add_test --use_dummy_delegate=true
```

#### Benchmark and Task Evaluation Tools

In TFLite, we have developed
[model benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
and
[evaluation tools](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks)
that already have integrated existing various TFLite delegates. To reuse these
tools for the new delegate, similar to the kernel testing above, we simply add
the created delegate provider as an additional dependency when building the
binary. See rules in the
[BUILD](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/BUILD)
file for details.

Take reusing the TFLite model benchmark tool as an example, after the delegate
provider is created, define the BUILD rule like the following:

```
cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        # Simply add the delegate provider as an extra dep.
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)
```

Now build the binary, and specify the commandline flags defined in this new
delegate provider and others detailed in the benchmark model tool
[doc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md)
when running the benchmark tool like the following:

```
bazel build -c opt tensorflow/lite/delegates/utils/dummy_delegate:benchmark_model_plus_dummy_delegate

# Setting --use_dummy_delegate=true will apply the dummy delegate to the
# TFLite model graph.
bazel-bin/tensorflow/lite/delegates/utils/dummy_delegate/benchmark_model_plus_dummy_delegate --graph=/tmp/mobilenet-v2.tflite --use_dummy_delegate=true

```

### Option 2: Utilize Tensorflow Lite External Delegate
In this **alternative approach to reuse existing Tensorflow Lite kernel testing
and tooling**, we first create an external delegate adaptor like the [`external_delegate_adaptor.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc) here, and create the corresponding BUILD target
to build a dynamic library.

Afterwards, one could build binaries or use pre-built ones to run with the
dummy delegate as long as the binary is linked with the
[`external_delegate_provider`](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159)
library which supports command-line flags as described
[here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider).
Note this external delegate provider has already been linked to existing testing
and tooling binaries.

For example, the following illustrates how to benchmark the dummy delegate here
via this external-delegate approach. We could use similar commands for testing
and evaluation tools.

```
bazel build -c opt tensorflow/lite/delegates/utils/dummy_delegate:dummy_external_delegate.so

# Copy the .so file to the directory that the external delegate will be loaded
# from at your choice.
cp bazel-bin/tensorflow/lite/delegates/utils/dummy_delegate/dummy_external_delegate.so /tmp

bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model

# Setting a non-empty --external_delegate_path value will trigger applying
# the external delegate during runtime.
bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=/tmp/mobilenet-v2.tflite \
  --external_delegate_path=/tmp/dummy_external_delegate.so \
  --external_delegate_options='error_during_init:true;error_during_prepare:true'
```

It is worth noting the *external delegate* is the corresponding C++
implementation of the *delegate* in Tensorflow Lite Python binding as shown
[here](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42).
Therefore, the dynamic external delegate adaptor library created here could be
directly used with Tensorflow Lite Python APIs.

More detailed guide on TFLite delegate is coming soon.
