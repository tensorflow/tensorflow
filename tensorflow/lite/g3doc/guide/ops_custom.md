# Custom operators

Since the TensorFlow Lite builtin operator library only supports a limited
number of TensorFlow operators, not every model is convertible. For details,
refer to [operator compatibility](ops_compatibility.md).

To allow conversion, users can provide their own custom implementation of an
unsupported TensorFlow operator in TensorFlow Lite, known as a custom operator.
*If instead, you wish to combine a series of unsupported (or supported)
TensorFlow operators into a single fused optimized custom operator, refer to
[operator fusing](https://www.tensorflow.org/lite/convert/operation_fusion).*

Using custom operators consists of four steps.

*   [Create a TensorFlow Model.](#create-a-tensorflow-model) Make sure the Saved
    Model (or Graph Def) refers to the correctly named TensorFlow Lite operator.

*   [Convert to a TensorFlow Lite Model.](#convert-to-a-tensorflow-lite-model)
    Make sure you set the right TensorFlow Lite converter attribute in order to
    successfully convert the model.

*   [Create and register the operator.](#create-and-register-the-operator) This
    is so that the TensorFlow Lite runtime knows how to map your operator and
    parameters in your graph to executable C/C++ code.

*   [Test and profile your operator.](#test-and-profile-your-operator) If you
    wish to test just your custom operator, it is best to create a model with
    just your custom operator and use the
    [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc)
    program.

Let’s walk through an end-to-end example of running a model with a custom
operator `tf.sin` (named as `Sin`, refer to #create-a-tensorflow-model) which is
supported in TensorFlow, but unsupported in TensorFlow Lite.

Note: In reality, `tf.sin` is **not** a custom operator. It is regular operator
which is supported by both TensorFlow and TensorFlow Lite. But we **assume**
that it is a custom operator in the following example in order to demonstrate a
simple workflow.

## Example: Custom `Sin` operator

Let’s walk through an example of supporting a TensorFlow operator that
TensorFlow Lite does not have. Assume we are using the `Sin` operator and that
we are building a very simple model for a function `y = sin(x + offset)`, where
`offset` is trainable.

### Create a TensorFlow Model

The following code snippet trains a simple TensorFlow model. This model just
contains a custom operator named `Sin`, which is a function `y = sin(x +
offset)`, where `offset` is trainable.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-0.6569866 ,  0.99749499,  0.14112001, -0.05837414,  0.80641841]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Sin`
@tf.function
def sin(x):
  return tf.sin(x + offset, name="Sin")

  # Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = sin(x)
      loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))

for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())
```

```python
The actual offset is: 1.0
The predicted offset is: 1.0000001
```

At this point, if you try to generate a TensorFlow Lite model with the default
converter flags, you will get the following error message:

```none
Error:
Some of the operators in the model are not supported by the standard TensorFlow
Lite runtime...... Here is
a list of operators for which you will need custom implementations: Sin.
```

### Convert to a TensorFlow Lite Model

Create a TensorFlow Lite model with custom operators, by setting the converter
attribute `allow_custom_ops` as shown below:

<pre>
converter = tf.lite.TFLiteConverter.from_concrete_functions([sin.get_concrete_function(x)])
<b>converter.allow_custom_ops = True</b>
tflite_model = converter.convert()
</pre>

At this point, if you run it with the default interpreter, you will get the
following error messages:

```none
Error:
Didn't find custom operator for name 'Sin'
Registration failed.
```

### Create and register the operator.

All TensorFlow Lite operators (both custom and builtin) are defined using a
simple pure-C interface that consists of four functions:

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

Refer to
[`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)
for details on `TfLiteContext` and `TfLiteNode`. The former provides error
reporting facilities and access to global objects, including all the tensors.
The latter allows implementations to access their inputs and outputs.

When the interpreter loads a model, it calls `init()` once for each node in the
graph. A given `init()` will be called more than once if the op is used multiple
times in the graph. For custom ops a configuration buffer will be provided,
containing a flexbuffer that maps parameter names to their values. The buffer is
empty for builtin ops because the interpreter has already parsed the op
parameters. Kernel implementations that require state should initialize it here
and transfer ownership to the caller. For each `init()` call, there will be a
corresponding call to `free()`, allowing implementations to dispose of the
buffer they might have allocated in `init()`.

Whenever the input tensors are resized, the interpreter will go through the
graph notifying implementations of the change. This gives them the chance to
resize their internal buffer, check validity of input shapes and types, and
recalculate output shapes. This is all done through `prepare()`, and
implementations can access their state using `node->user_data`.

Finally, each time inference runs, the interpreter traverses the graph calling
`invoke()`, and here too the state is available as `node->user_data`.

Custom ops can be implemented in exactly the same way as builtin ops, by
defining those four functions and a global registration function that usually
looks like this:

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

Note that registration is not automatic and an explicit call to
`Register_MY_CUSTOM_OP` should be made. While the standard `BuiltinOpResolver`
(available from the `:builtin_ops` target) takes care of the registration of
builtins, custom ops will have to be collected in separate custom libraries.

### Defining the kernel in the TensorFlow Lite runtime

All we need to do to use the op in TensorFlow Lite is define two functions
(`Prepare` and `Eval`), and construct a `TfLiteRegistration`:

```cpp
TfLiteStatus SinPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node,0);
  TfLiteTensor* output = GetOutput(context, node,0);

  float* input_data = input->data.f;
  float* output_data = output->data.f;

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = sin(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {nullptr, nullptr, SinPrepare, SinEval};
  return &r;
}
```

When initializing the `OpResolver`, add the custom op into the resolver (see
below for an example). This will register the operator with Tensorflow Lite so
that TensorFlow Lite can use the new implementation. Note that the last two
arguments in `TfLiteRegistration` correspond to the `SinPrepare` and `SinEval`
functions you defined for the custom op. If you used `SinInit` and `SinFree`
functions to initialize variables used in the op and to free up space,
respectively, then they would be added to the first two arguments of
`TfLiteRegistration`; those arguments are set to `nullptr` in this example.

### Register the operator with the kernel library

Now we need to register the operator with the kernel library. This is done with
an `OpResolver`. Behind the scenes, the interpreter will load a library of
kernels which will be assigned to execute each of the operators in the model.
While the default library only contains builtin kernels, it is possible to
replace/augment it with a custom library op operators.

The `OpResolver` class, which translates operator codes and names into actual
code, is defined like this:

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

Regular usage requires that you use the `BuiltinOpResolver` and write:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

To add the custom op created above, you call `AddOp` (before you pass the
resolver to the `InterpreterBuilder`):

```c++
resolver.AddCustom("Sin", Register_SIN());
```

If the set of builtin ops is deemed to be too large, a new `OpResolver` could be
code-generated based on a given subset of ops, possibly only the ones contained
in a given model. This is the equivalent of TensorFlow's selective registration
(and a simple version of it is available in the `tools` directory).

If you want to define your custom operators in Java, you would currently need to
build your own custom JNI layer and compile your own AAR
[in this jni code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/builtin_ops_jni.cc).
Similarly, if you wish to define these operators available in Python you can
place your registrations in the
[Python wrapper code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

Note that a similar process as above can be followed for supporting a set of
operations instead of a single operator. Just add as many `AddCustom` operators
as you need. In addition, `BuiltinOpResolver` also allows you to override
implementations of builtins by using the `AddBuiltin`.

### Test and profile your operator

To profile your op with the TensorFlow Lite benchmark tool, you can use the
[benchmark model tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool)
for TensorFlow Lite. For testing purposes, you can make your local build of
TensorFlow Lite aware of your custom op by adding the appropriate `AddCustom`
call (as show above) to
[register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/register.cc)

## Best practices

1.  Optimize memory allocations and de-allocations cautiously. Allocating memory
    in `Prepare` is more efficient than in `Invoke`, and allocating memory
    before a loop is better than in every iteration. Use temporary tensors data
    rather than mallocing yourself (see item 2). Use pointers/references instead
    of copying as much as possible.

2.  If a data structure will persist during the entire operation, we advise
    pre-allocating the memory using temporary tensors. You may need to use
    OpData struct to reference the tensor indices in other functions. See the
    example in the
    [kernel for convolution](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc).
    A sample code snippet is below

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3.  If it doesn't cost too much wasted memory, prefer using a static fixed size
    array (or a pre-allocated `std::vector` in `Resize`) rather than using a
    dynamically allocated `std::vector` every iteration of execution.

4.  Avoid instantiating standard library container templates that don't already
    exist, because they affect binary size. For example, if you need a
    `std::map` in your operation that doesn't exist in other kernels, using a
    `std::vector` with direct indexing mapping could work while keeping the
    binary size small. See what other kernels use to gain insight (or ask).

5.  Check the pointer to the memory returned by `malloc`. If this pointer is
    `nullptr`, no operations should be performed using that pointer. If you
    `malloc` in a function and have an error exit, deallocate memory before you
    exit.

6.  Use `TF_LITE_ENSURE(context, condition)` to check for a specific condition.
    Your code must not leave memory hanging when `TF_LITE_ENSURE` is used, i.e.,
    these macros should be used before any resources are allocated that will
    leak.
