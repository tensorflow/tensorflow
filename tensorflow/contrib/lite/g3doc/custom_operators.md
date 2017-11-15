# How to use custom operators

TensorFlow Lite currently supports a subset of TensorFlow operators. However, it
does support the use of user-provided implementations (as known as custom
implementations) if the model contains an operator that is not supported.

Let’s walk through this via an example. Assume we are using the `Sin` operator
and that we are building a very simple model for a function `y = sin(x +
offset)`, where `offset` is trainable.

The code to train the TensorFlow model will be something like:

```python
offset = tf.get_variable("offset", [1,], tf.float32)
x = tf.placeholder(tf.float32, shape=(None,))
y = tf.sin(x + offset)
y_ = tf.placeholder(tf.float32, shape=(None,))
loss = tf.reduce_sum(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
```

If you convert this model to Tensorflow Lite format using the TensorFlow Lite
Optimizing Converter with `--allow_custom_ops` argument, and run it with the
default interpreter, the interpreter will raise the following error messages:

```
Didn't find custom op for name 'Sin'
Registration failed.
```

All we need to do to use the op in TensorFlow Lite is define two functions
(`Prepare` and `Eval`), and construct a `TfLiteRegistration`. This code would
look something like this:

```cpp
TfLiteStatus SinPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input = GetInput(context, node, 0);
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
  TfLiteTensor* input = GetInput(context, node,0);
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
  static TfLiteRegistration r = {nullptr, nullptr, SinResize, SinEval};
  return &r;
}
```

When initializing the OpResolver, add the custom op into the resolver, this will
register the operator with Tensorflow Lite so that TensorFlow Lite can use the
new implementation.

```cpp
tflite::ops::builtin::BuiltinOpResolver builtins;
builtins.AddCustom("Sin", Register_SIN());
```

Note that a similar process as above can be followed for supporting for a set of
operations instead of a single operator.
