This folder contains a convenience library called *tf-shim* over TF and TFLite
op kernel APIs.

## Summary

This library creates a shim over the custom op APIs of TF and TFLite so the
developer can write the custom op once with minimal binary or runtime overhead.

An example usage is an input preprocessing op kernel that can be used in
both TF and TFLite.

## Background

When there is a need to implement a logic that is not supported by the TF
builtin ops the alternative is to build a custom op. If that op needs to
run on-device then it needs to be written in C++ against the client API for
custom ops.

For example, feature processing especially for textual input in an ML model
can involve operations that don't lend themselves well to vectorization and the
code, if written as a C++ function, would be much shorter and more readable.

However, Tensorflow and TFLite APIs for creating op kernels are, at the moment,
not identical. This library offers a convenient way to write the kernel once and
adapt it to both TF and TFLite with minimal binary and runtime overhead.

## Implementation

This folder contains two pieces:

1.  `TensorView` as a shim over `::tensorflow::Tensor` and `TfLiteTensor`

2.  `OpKernelShim` class which abstracts the TF and TFLite op kernel APIs.

### TensorView

This class is a *view* over an already allocated tensor in TF or TFLite without
taking any ownership. In that sense it is similar to `absl::string_view` but with
the difference that the underlying buffer can be mutable.

Example Usage:

```
::tensorflow::Tensor tf_tensor;
auto t = TensorView::New(&tf_tensor);

auto t_str_mat = t.As<::tensorflow::tstring, /*RANK=*/ 2>();
t(0, 0) = "ab";
t(0, 1) = "cde"


auto t_buffer = t.Data<::tensorflow::tstring>();
t[0] = "ab";
t[1] = "cde"
```

```
TfLiteTensor tflite_tensor;
auto t = TensorView::New(&tflite_tensor);

auto t_int_vec = t.As<int32, /*RANK=*/ 1>();
t(0) = 123;
t(1) = 456

auto t_buffer = t.Data<int32>();
t[0] = 123;
t[1] = 456
```

The `New` is the factory function which based on the type of the input returns
either a `TfTensorView` or a `TfLiteTensorView`.

See the unit tests `tf_tensor_view_test.cc` and `tflite_tensor_view_test.cc` for
more usage.

The string tensor in `TfLiteTensorView` is a bit of special case. Since string
tensors in TfLite are serialized in a specific format, while writing to those
tensors an intermediate buffer is needed to hold intermediate values before all
the strings get serialized. The intermediate string buffers are serialized back
to the TfLite string format once the last remaining `TfLiteTensorView` goes out
of scope. Only then the user can see the string values in the underlying
`TfLiteTensor`. That said, when implementing an op kernel, there is rarely a
need to read back the contents of a mutable output `TfLiteTensor` within the
same code block.

### OpKernelShim

*WARNING: Experimental interface, subject to change*

This class defines the interface which when implemented allows for convenient
adaptation to TF and TFLite op kernels.

Here is an example op kernel implementing this interface:

```
template<TfRuntime R>
class MyOp : public OpKernelShim<MyOp, R> {

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs();

  // Input tensors declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* ctx);

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx);
};
```

The class `MyOp` is passing itself to `OpKernelShim` as a template parameter.
This is because `OpKernelShim` is a static interface using the CRTP pattern.
Similarly, the context classes: `InitContext`, `InvokeContext` and
`ShapeInferenceContext` are all static interfaces in the same way.

The class `MyOp` can also be templatized. See `test_op/tmpl_op.h` for an
example.

### Context Interfaces

An op kernel written using this library has access to a number of *context*
objects at various stages of its lifecycle. These context objects are
effectively shims over the existing context objects in TF and TFLite.

#### InitContext
An instance of this class is passed to the op kernel during its initialization.

```
template <typename SubType>
class InitContext {
 public:
  // Read the given attribute and populate the given value.
  template <typename AttrType>
  absl::Status GetAttr(const std::string& attr_name, AttrType* value) const;
};
```

#### InvokeContext
An instance of this class is passed to the op kernel during its invocation.

```
template <typename SubType>
class InvokeContext {
 public:
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const;
};
```

#### ShapeInferenceContext
An instance of this class is passed to the op kernel during its shape inference.

```
template <typename SubType>
class ShapeInferenceContext {
 public:
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;
};
```
