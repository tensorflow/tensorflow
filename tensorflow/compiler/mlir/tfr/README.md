# Composable Tensorflow

## Composable Tensorflow

Composable TensorFlow (TF) is the framework for defining portable TF ops with
composition in the authoring language.

The set of standard TF ops is currently open. New ops are defined for special
purposes but it is hard to make them work end-to-end: The op
needs to be handled separately by several backends (tf2xla bridge, tflite
converter, CPU kernels, etc.). Writing shape functions and gradients for these
ops is extremely difficult. `tf.function` makes some parts of the implementation
simpler, but it introduces runtime overhead and it cannot easily be used to
apply dedicated optimizations to op kernels.

The composable TF framework allows the user to define portable TF ops as
ompositions of other TF ops. It translates a Python function used to define the
composition directly into a portable IR at build time, and uses it to expand the
composite op in the TF program during compilation / execution. By using this
expansion mechanism, new op are readily available on different platforms without
extra work. Moreover, since the expansion is optional, the backend can easily
treat it as a monolithic op when needed, for instance to apply optimizations or
associate it with a custom kernel.

### Benefits

Using the Composable TF API to define a new op and its composition can bring the
following benefits:

* *Automatic backend support*: As long as it is composed of ops supported by the
backend, the new op is automatcally supported (as a `tf.function` alternative);
* *Reduced tracing overhead*: Unlike `tf.function`, the composition function is
compiled at build time, hence TF only needs to trace a single op to build the
`graph`;
* *Easy fused op/kernel optimization*: Even if it has complex
semantics, the new op is presented as a single node in the graph, thus
optimization passes and kernels can easily be specialized to this op for better
performance.
* *Automatic shape/type inference support*: No shape functions are required for
the new op;
* *Automatic gradient support (WIP)*: The user doesn't need to author
gradient a function of the op for training.

### Use Cases

* (Portablity) User wants to add a new op and run this op on different
platforms (CPU, TPU, TFLite, etc.) to be portable.
 * *Solution*: The user should define the new op as a composition. The ops used
 inside the composition should have support for these platforms. These ops can
 also be composite ops.

* (Performance) User defines a custom kernel for a regular structure
(i.e. LSTM), but it is hard to add the logic to fuse the individual ops to
target this kernel in the inference graph.
 * *Solution*: The user should define a new TF op, which corresponds to the
 fused kernel, with composition, and use this op to build the model for both
 training and inference. For the platforms where a fused kernel is not
 available, the execution will use the composition instead.

## Gradient
(TODO)

## Authoring Op Composition in Python

The composable TF provides a single API to define a new op with its composition
at the same time. For example, the following code defines a new
`FusedFullyConnected` op, which have `MatMul`, `Add` and some
`activation function` (specified by an op attribute) fused.


```python
import tensorflow as tf

@Composite(
    'FusedFullyConnected',
    inputs=['input_: T', 'filter_: T', 'bias: T'],
    attrs=['act: {"", "RELU", "RELU6", "TANH"} = ""'],
    derived_attrs=['T: {float, int8}'],
    outputs=['o: T'])
def _composite_fully_connected(input_, filter_, bias, act):
  res = tf.raw_ops.MatMul(
      a=input_, b=filter_, transpose_a=False, transpose_b=True)
  res = tf.raw_ops.Add(x=res, y=bias)
  if act == 'RELU':
    return tf.raw_ops.Relu(features=res)
  elif act == 'RELU6':
    return tf.raw_ops.Relu6(features=res)
  elif act == 'TANH':
    return tf.raw_ops.Tanh(x=res)
  else:
    return res

```

Besides defining new ops, composition can be specified for an existing op
for portability. The following code defines the semantics of `AddNOp`:

```python
@Composite('AddNOp')
def _my_op_c(ins):
  N = len(ins)
  if N == 1:
    return ins[0]
  sum = ins[0]
  for i in range(1, N):
    sum += ins[i]
  return sum
```

Utilities have been built to compile the Python composition functions down to
the backend IR. The project also provides a set of graph optimization passes to
expand the composite ops in the graph by using the input backend IR. These
passes have been added to the TF
[common runtime](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime)
for graph execution and
[eager runtime](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager)
for eager execution.

## Compiling Op Composition

### Ahead-Of-Time (AOT) mode

Like the op kernels, the op composition can be pre-compiled to the backend IR
so the decomposition can be invoked at runtime. A Python [define_op_template.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tfr/define_op_template.py)
file is provided as an example to build composite ops in the users project
directory. All the targets required to build the new ops are created by the
following target:


```BUILD
load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")

gen_op_libraries(
    name = "test_ops",
    src = "define_op_template.py",
    deps = [
        "//third_party/py/tensorflow",
    ],
)
```

More composite op definitions and usages are here included in the
[examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tfr/examples)
directory.

### Just-In-Time (JIT) mode
(TODO)

## Known Limitations

* `while` statement
* condition of `if` statement couldn't be a tensor

## RFC
This project is an alternative implementaion of [RFC:Standardizing composite ops in tensorflow to support efficient inference](https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md).
This project doesn't rely on the tracing functionality provided by `tf.function`
to avoid all its pitfalls and it helps to build more general transformations in
the backends.

## Team

* Feng Liu
* Dan Moldovan

