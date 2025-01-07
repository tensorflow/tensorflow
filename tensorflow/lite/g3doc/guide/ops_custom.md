# Custom operators

Since the TensorFlow Lite builtin operator library only supports a limited
number of TensorFlow operators, not every model is convertible. For details,
refer to [operator compatibility](ops_compatibility.md).

To allow conversion, users can provide their own custom implementation of an
unsupported TensorFlow operator in TensorFlow Lite, known as a custom operator.
*If instead, you wish to combine a series of unsupported (or supported)
TensorFlow operators into a single fused optimized custom operator, refer to
[operator fusing](https://www.tensorflow.org/lite/models/convert/operation_fusion).*

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
operator `tf.atan` (named as `Atan`, refer to #create-a-tensorflow-model) which
is supported in TensorFlow, but unsupported in TensorFlow Lite.

Note: The `tf.atan` function is **not** a custom operator. It is a regular
operator
which is supported by both TensorFlow and TensorFlow Lite. But we **assume**
that it is a custom operator in the following example in order to demonstrate a
simple workflow.

The TensorFlow Text operator is an example of a custom operator. See the
<a href="https://tensorflow.org/text/guide/text_tf_lite" class="external">
  Convert TF Text to TF Lite</a> tutorial for a code example.

## Example: Custom `Atan` operator

Let’s walk through an example of supporting a TensorFlow operator that
TensorFlow Lite does not have. Assume we are using the `Atan` operator and that
we are building a very simple model for a function `y = atan(x + offset)`, where
`offset` is trainable.

### Create a TensorFlow Model

The following code snippet trains a simple TensorFlow model. This model just
contains a custom operator named `Atan`, which is a function `y = atan(x +
offset)`, where `offset` is trainable.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-1.4288993, 0.98279375, 1.2490457, 1.2679114, 1.5658458]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Atan`
@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def atan(x):
  return tf.atan(x + offset, name="Atan")

# Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = atan(x)
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
The predicted offset is: 0.99999905
```

At this point, if you try to generate a TensorFlow Lite model with the default
converter flags, you will get the following error message:

```none
Error:
error: 'tf.Atan' op is neither a custom op nor a flex op.
```

### Convert to a TensorFlow Lite Model

Create a TensorFlow Lite model with custom operators, by setting the converter
attribute `allow_custom_ops` as shown below:

<pre>
converter = tf.lite.TFLiteConverter.from_concrete_functions([atan.get_concrete_function()], atan)
<b>converter.allow_custom_ops = True</b>
tflite_model = converter.convert()
</pre>

At this point, if you run it with the default interpreter using commands such as
follows:

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

You will still get the error:

```none
Encountered unresolved custom op: Atan.
```

### Create and register the operator.

```c++
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
```

TensorFlow Lite custom operators are defined using a simple pure-C API that
consists of an opaque type (`TfLiteOperator`) and related functions.

`TfLiteOperator` is an opaque type:

```c++
typedef struct TfLiteOperator TfLiteOperator;
```

`TfLiteOperator` stores the operator's identity and implementation.
(Note that the operator is distinct from its operands, which are stored in the
TF Lite graph nodes for nodes that call the operator.)

Instances of this type are constructed with calls to
`TfLiteOperatorCreate` and can be destroyed by calling
`TfLiteOperatorDelete`.

The operator's identity is set via the parameters to the constructor function
`TfLiteOperatorCreate`:

```c++
TfLiteOperator*
TfLiteOperatorCreate(
    TfLiteBuiltinOperator builtin_code,  // Normally `TfLiteBuiltinCustom`.
    const char* custom_name,  // The name of the custom op.
    int version  // Normally `1` for the first version of a custom op.
);
```

The operator implementation can define "methods" with the following signatures.
All of these methods are optional, but for an operator to be successfully
evaluated, the operator implementation needs to define and set (using the setter
functions) at least the `Prepare` and `Invoke` methods.

```c++
// Initializes the op from serialized data.
void* Init(TfLiteOpaqueContext* context, const char* buffer, size_t length);

// Deallocates the op.
// The pointer `buffer` is the data previously returned by an Init invocation.
void Free(TfLiteOpaqueContext* context, void* buffer);

// Called when the inputs that this node depends on have been resized.
TfLiteStatus Prepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

// Called when the node is executed. (Should read node inputs and write to
// node outputs).
TfLiteStatus Invoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

// Retrieves the async kernel.
TfLiteAsyncKernel AsyncKernel(TfLiteOpaqueContext* context,
                              TfLiteOpaqueNode* node);
```

The function *names* (or namespace prefixes, for C++) in your op implementation
don't have to match the function names in the above code snippet, since the TF
Lite custom ops API will only use their addresses. Indeed we recommend that you
declare them in an anonymous namespace or as static functions.

But it is a good idea to include your operator name as a namespace or prefix on
these function names:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-cpp">
namespace my_namespace::my_custom_op {
  void* Init(TfLiteOpaqueContext* context,
             const char* buffer, size_t length) { ... }
  // ... plus definitions of Free, Prepare, and Invoke ...
}
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-cpp">
void* MyCustomOpInit(TfLiteOpaqueContext* context,
                     const char* buffer, size_t length) { ... }
// ... plus definitions of MyCustomOpFree, MyCustomOpPrepare, and
// MyCustomOpInvoke.
      </pre></p>
    </section>
  </devsite-selector>
</div>

Since this is a C API, these "methods" are implemented as C function pointers in
the `TfLiteOperator` type, which are set by passing the addresses of
your implementation functions to the corresponding setter functions
`TfLiteOperatorSet`*MethodName*:

```c++
void TfLiteOperatorSetInit(
    TfLiteOperator* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length));
void TfLiteOperatorSetFree(
    TfLiteOperator* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data));
void TfLiteOperatorSetPrepare(
    TfLiteOperator* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));
void TfLiteOperatorSetInvoke(
    TfLiteOperator* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));
void TfLiteOperatorSetAsyncKernel(
    TfLiteOperator* registration,
    struct TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node));
```

Refer to
[`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/c/common.h)
for details on `TfLiteContext` and `TfLiteNode`. `TfLiteContext` provides error
reporting facilities and access to global objects, including all the tensors.
`TfLiteNode` allows operator implementations to access their inputs and outputs.

When the interpreter loads a model, it calls the `Init()` method once for each
node in the graph. A given `Init()` will be called more than once if the op is
used multiple times in the graph. For custom ops a configuration buffer will be
provided, containing a flexbuffer that maps parameter names to their values. The
buffer is empty for builtin ops because the interpreter has already parsed the
op parameters. Kernel implementations that require state should initialize it
here and transfer ownership to the caller. For each `Init()` call, there will be
a corresponding call to `Free()`, allowing implementations to dispose of the
buffer they might have allocated in `Init()`.

Whenever the input tensors are resized, the interpreter will go through the
graph notifying implementations of the change. This gives them the chance to
resize their internal buffer, check validity of input shapes and types, and
recalculate output shapes. This is all done through the `Prepare()` method, and
implementations can access their state using
`TfLiteOpaqueNodeGetUserData(node)`.

Finally, each time inference runs, the interpreter traverses the graph calling
the `Invoke()` method, and here too the state is available as
`TfLiteOpaqueNodeGetUserData(node)`.

Custom ops can be implemented by defining those "method" functions, and then
defining a function that returns an instance of `TfLiteOperator`
constructed by calling `TfLiteOperatorCreate` and then the relevant
setter methods:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-cpp">
namespace my_namespace::my_custom_op {
  namespace {
    void* Init(TfLiteOpaqueContext* context,
               const char* buffer, size_t length) { ... }
    void Free(TfLiteOpaqueContext* context, void* buffer) { ... }
    TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                         TfLiteOpaqueNode* node) { ... }
    TfLiteStatus Invoke(TfLiteOpaqueContext* context,
                        TfLiteOpaqueNode* node) {... }
  };

  const TfLiteOperator* MyCustomOpRegistrationExternal() {
    // Singleton instance, intentionally never destroyed.
    static const TfLiteOperator* my_custom_op = ()[] {
        TfLiteOperator* r =
            TfLiteOperatorCreate(
                kTfLiteBuiltinCustom, "MyCustomOp", /*version=*/ 1);
        TfLiteOperatorSetInit(r, Init);
        TfLiteOperatorSetFree(r, Free);
        TfLiteOperatorSetPrepare(r, Prepare);
        TfLiteOperatorSetInvoke(r, Eval);
        return r;
      };
    return my_custom_op;
  }
}  // namespace my_namespace
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-cpp">
static void* MyCustomOpInit(TfLiteOpaqueContext* context, const char* buffer,
                     size_t length) { ... }
static void MyCustomOpFree(TfLiteOpaqueContext* context, void* buffer) { ... }
static TfLiteStatus MyCustomOpPrepare(TfLiteOpaqueContext* context,
                                      TfLiteOpaqueNode* node) { ... }
static TfLiteStatus MyCustomOpInvoke(TfLiteOpaqueContext* context,
                                     TfLiteOpaqueNode* node) {... }

static TfLiteOperator* MyCustomOpCreate() {
  const TfLiteOperator* r =
      TfLiteOperatorCreate(
          kTfLiteBuiltinCustom, "MyCustomOp", /*version=*/ 1);
  TfLiteOperatorSetInit(r, MyCustomOpInit);
  TfLiteOperatorSetFree(r, MyCustomOpFree);
  TfLiteOperatorSetPrepare(r, MyCustomOpPrepare);
  TfLiteOperatorSetInvoke(r, MyCustomOpEval);
  return r;
}

const TfLiteOperator* MyCustomOpRegistrationExternal() {
  // Singleton instance, intentionally never destroyed.
  static const TfLiteOperator* my_custom_op = MyCustomOpCreate();
  return my_custom_op;
}
      </pre></p>
    </section>
  </devsite-selector>
</div>

Note that registration is not automatic and an explicit call to your
`MyCustomOpRegistration` function should be made (see details below). While the
standard `BuiltinOpResolver` (available from the `:builtin_ops` target) takes
care of the registration of builtins, custom ops will have to be collected in
separate custom libraries.

### Defining the kernel in the TensorFlow Lite runtime

All we need to do to use the op in TensorFlow Lite is define two functions
(`Prepare` and `Eval`), and a third to construct a `TfLiteOperator`:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-cpp">
namespace atan_op {
  namespace {
    TfLiteStatus AtanPrepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
      TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumInputs(node), 1);
      TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumOutputs(node), 1);

      const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
      TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);

      int num_dims = TfLiteOpaqueTensorNumDimensions(input);

      TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
      for (int i=0; i < num_dims; ++i) {
        output_size->data[i] = input->dims->data[i];
      }

      return TfLiteOpaqueContextResizeTensor(context, output, output_size);
    }

    TfLiteStatus AtanEval(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
      const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
      TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);

      float* input_data = static_cast<float*>(TfLiteOpaqueTensorData(input));
      float* output_data = static_cast<float*>(TfLiteOpaqueTensorData(output));

      size_t count = 1;
      int num_dims = TfLiteOpaqueTensorNumDimensions(input);
      for (int i = 0; i < num_dims; ++i) {
        count *= input->dims->data[i];
      }

      for (size_t i = 0; i < count; ++i) {
        output_data[i] = atan(input_data[i]);
      }
      return kTfLiteOk;
    }
  }  // anonymous namespace

  const TfLiteOperator* AtanOpRegistrationExternal() {
    // Singleton instance, intentionally never destroyed.
    static const TfLiteOperator* atan_op = ()[] {
        auto* r = TfLiteOperatorCreate(
            kTfLiteBuiltinCustom, "ATAN", /*version=*/ 1);
        TfLiteOperatorSetPrepare(r, Prepare);
        TfLiteOperatorSetInvoke(r, Eval);
        return r;
      };
    return atan_op;
  }
}  // namespace atan_op
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-cpp">
static TfLiteStatus AtanPrepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumInputs(node), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumOutputs(node), 1);

  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);

  int num_dims = TfLiteOpaqueTensorNumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return TfLiteOpaqueContextResizeTensor(context, output, output_size);
}

static TfLiteStatus AtanEval(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);

  float* input_data = static_cast<float*>(TfLiteOpaqueTensorData(input));
  float* output_data = static_cast<float*>(TfLiteOpaqueTensorData(output));

  size_t count = 1;
  int num_dims = TfLiteOpaqueTensorNumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i = 0; i < count; ++i) {
    output_data[i] = atan(input_data[i]);
  }
  return kTfLiteOk;
}

static const TfLiteOperator* AtanOpCreate() {
  TfLiteOperator* r = TfLiteOperatorCreate(
          kTfLiteBuiltinCustom, "ATAN", /*version=*/ 1);
  TfLiteOperatorSetPrepare(r, Prepare);
  TfLiteOperatorSetInvoke(r, Eval);
  return r;
}

const TfLiteOperator* AtanOpRegistrationExternal() {
  // Singleton instance, intentionally never destroyed.
  static const TfLiteOperator* atan_op = AtanOpCreate();
  return atan_op;
}
      </pre></p>
    </section>
  </devsite-selector>
</div>

When initializing the `OpResolver`, add the custom op into the resolver (see
below for an example). This will register the operator with Tensorflow Lite so
that TensorFlow Lite can use the new implementation. Note that the last two
arguments in `TfLiteRegistration` correspond to the `AtanPrepare` and `AtanEval`
functions you defined for the custom op. If you used `AtanInit` and `AtanFree`
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
 public:
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  ...
};
```

Note that for backwards compatibility, this class uses the older concrete type
`TfLiteRegistration` rather than the opaque type `TfLiteOperator`,
but the `TfLiteRegistration` struct contains a `registration_external` field of
type `TfLiteOperator*`.

The `MutableOpResolver` and `BuiltinOpResolver` classes are derived from
`OpResolver`:

```c++
class MutableOpResolver : public OpResolver {
 public:
  MutableOpResolver();  // Constructs an initially empty op resolver.
  void AddAll(const MutableOpResolver& other);
  ...
};

class BuiltinOpResolver : public MutableOpResolver {
 public:
  BuiltinOpResolver();  // Constructs an op resolver with all the builtin ops.
};
```

Regular usage (without custom ops) requires that you use the `BuiltinOpResolver`
and write:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

To add the custom op created above, you can instead use a `MutableOpResolver`,
and call `AddCustom` (before you pass the resolver to the
`InterpreterBuilder`):

```c++
tflite::ops::builtin::MutableOpResolver resolver;
resolver.AddAll(tflite::ops::builtin::BuiltinOpResolver());
tflite::AddOp(&resolver, AtanOpRegistration());
```

If the set of builtin ops is deemed to be too large, a new `OpResolver` could be
code-generated based on a given subset of ops, possibly only the ones contained
in a given model. This is the equivalent of TensorFlow's selective registration
(and a simple version of it is available in the `tools` directory).

If you want to define your custom operators in Java, you would currently need to
build your own custom JNI layer and compile your own AAR
[in this jni code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc).
Similarly, if you wish to define these operators available in Python you can
place your registrations in the
[Python wrapper code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

Note that a similar process as above can be followed for supporting a set of
operations instead of a single operator. Just add as many `AddCustom` operators
as you need. In addition, `MutableOpResolver` also allows you to override
implementations of builtins by using `AddBuiltin`.

### Test and profile your operator

To profile your op with the TensorFlow Lite benchmark tool, you can use the
[benchmark model tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool)
for TensorFlow Lite. For testing purposes, you can make your local build of
TensorFlow Lite aware of your custom op by adding the appropriate `AddCustom`
call (as show above) to
[register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/core/kernels/register.cc)

## Best practices

1.  Optimize memory allocations and de-allocations cautiously. Allocating memory
    in `Prepare` is more efficient than in `Invoke`, and allocating memory
    before a loop is better than in every iteration. Use temporary tensors data
    rather than mallocing yourself (see item 2). Use pointers/references instead
    of copying as much as possible.

2.  If a data structure will persist during the entire operation, we advise
    pre-allocating the memory using temporary tensors. You may need to use an
    OpData struct to reference the tensor indices in other functions. See the
    example in the
    [kernel for convolution](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc).
    A sample code snippet is below.

    ```c++
    struct MyOpData {
      int temp_tensor_index;
      ...
    };

    void* Init(TfLiteOpaqueContext* context,
        const char* buffer, size_t length) {
      auto* op_data = new MyOpData{};
      ...
      return op_data;
    }
    void Free(TfLiteOpaqueContext* context, void* buffer) {
      ...
      delete reinterpret_cast<MyOpData*>(buffer);
    }
    TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                         TfLiteOpaqueNode* node) {
      ...
      auto* op_data =
          reinterpret_cast<MyOpData*>(TfLiteOpaqueNodeGetUserData(node));
      const int num_temporaries = 1;
      int temporary_tensor_indices[num_temporaries];
      TfLiteOpaqueTensorBuilder* builder = TfLiteOpaqueTensorBuilderCreate();
      TfLiteOpaqueTensorBuilderSetType(builder, kTfLiteFloat32);
      TfLiteOpaqueTensorBuilderSetAllocationType(builder, kTfLiteArenaRw);
      TfLiteOpaqueContextAddTensor(context, builder,
          &temporary_tensor_indices[0]);
      TfLiteOpaqueTensorBuilderDelete(builder);
      TfLiteOpaqueNodeSetTemporaries(node, temporary_tensor_indices,
          num_temporaries);
      op_data->temp_tensor_index = temporary_tensor_indices[0];
      ...
      return kTfLiteOk;
    }
    TfLiteStatus Invoke(TfLiteOpaqueContext* context,
                        TfLiteOpaqueNode* node) {
      ...
      auto* op_data = reinterpret_cast<MyOpData*>(
          TfLiteOpaqueNodeGetUserData(node));
      TfLiteOpaqueTensor* temp_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context,
              op_data->temp_tensor_index);
      TF_LITE_OPAQUE_ENSURE(context,
          TfLiteTensorType(temp_tensor) == kTfLiteFloat32);
      TF_LITE_OPAQUE_ENSURE(context,
          TfLiteTensorGetAllocationType(temp_Tensor) == kTfLiteArenaRw);
      void *temp_data = TfLiteTensorData(temp_tensor);
      TF_LITE_OPAQUE_ENSURE(context, temp_data != nullptr);
      ...
      return kTfLiteOk;
    }
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

6.  Use `TF_LITE_OPAQUE_ENSURE(context, condition)` to check for a specific
    condition. Your code must not leave memory hanging when
    `TF_LITE_OPAQUE_ENSURE` is used, i.e., these macros should be used before
    any resources are allocated that will leak.
