<!-- LINT.IfChange -->
# Create a custom multiplexer op with GPU support

This guide provides an end-to-end example for adding a custom multiplexer op
with both CPU and GPU support.

For a simpler example of a TensorFlow multiplexer custom op, refer to
`multiplex_1`. The `multiplex_2` operation builds on the `multiplex_1` operation
in the following ways:

*   This op includes support for both GPU and CPU, while `multiplex_1` only
    supports CPU.
*   This op uses the
    [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview)
    library to access tensor values and compute the multiplex operation. The
    `multiplex_1` op only uses Eigen to access tensor values.

This example uses `multiplex_2_kernel.cc` to register the op for CPU, and
`multiplex_2_kernel.cu.cc` to register the op for GPU. Excluding the
`multiplex_2_kernel.cu.cc` file from this op will result in a multiplexer similar
to multiplex_1.

The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## Creating a custom multiplexer op with GPU support

This example demonstrates how you can create a Python custom multiplexer,
`multiplex_2_op`, similar to
[`tf.where`](https://tensorflow.org/api_docs/python/tf/where?version=nightly).
It returns elements chosen from either of the two input tensors (`x` or `y`),
depending on the `condition`. You can call the op with the following:

<!-- test_snippets_in_readme skip -->
```python
multiplex_2_op.multiplex(condition, x, y)
```

This simplified `multiplex_2` op has the following limitations that are not
present in `tf.where`:

*   Support only for dense tensors
*   No broadcasting capabilities
*   No extensibility through optional parameters

This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.

### Step 1 - Define the op interface

Define the op interface and register it using the `REGISTER_OP` macro.

```
REGISTER_OP("Examples>MultiplexDense")
    .Input("cond: bool")
    .Input("a: T")
    .Input("b: T")
    .Output("output_values: T")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      // Determine the output shape and also assert that inputs 0 and 1 have
      // the same shape.
      tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
      // Assert that inputs 0 and 2 have the same shape, i.e. that all inputs
      // have the same shape. This is optional, but it is desirable
      // to raise errors about inconsistent input shapes early when using
      // graph mode.
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));

      c->set_output(0, out);
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Return elements chosen from `a` or `b` depending on `cond`.

This is similar to `np.where` and `tf.where`, but simplified to only handle
the case of dense tensors, no optional parameters, no broadcasting, etc..
This uses cond.select from the Eigen library and supports GPU (and CPU).

cond: tf.Tensor of type bool.
a: tf.Tensor with the same type and shape as `b`.
b: tf.Tensor with the same type and shape as `a`.

      Where True, yield `a`, otherwise yield `b`.
output_values: A tf.Tensor with elements from `a` where `cond` is True, and
               elements from `b` elsewhere.
)doc");
```

Note that:

*   This op has three input tensors - one boolean tensor for selecting which
    values to choose from the two other input tensors of matching type `T`, and
    one output tensor of type `T`.
*   The `Attr` for this op is defined as `.Attr("T: type")` which specifies `T`
    as an `Attr` of type `type`. In the subsequent steps, you will use `T` with
    a template class to define the type of the contents of tensors.
*   The docstring for this op is specified by passing a string to `.Doc()`.
*   The shape function for this op uses the `Merge` method of the
    [`tensorflow::shape_inference::InferenceContext`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h#:~:text=class%20InferenceContext)
    object which is a helper function to set the output shape to be the same as
    the identical shapes of the two inputs (for example, if it is used for
    binary ops) and has error checking to ensure that the two inputs have the
    same shape. Since `multiplex_2` has three inputs, two calls to `Merge` are
    used to assert that all three inputs are the same shape.
    
### Step 2 - Register the op implementation (kernel)

This example registers the kernel for both CPU and GPU. You can register the
kernel for only CPU using `multiplex_2_kernel.cc`.
This will result in a kernel similar to the `multiplex_1` custom op.
The types supported by GPU kernels are a subset of the types supported by CPU
kernels.

Register the kernel by calling the `REGISTER_KERNEL_BUILDER` macro.

```
#define REGISTER_KERNELS_GPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_GPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<GPUDevice, type>)

REGISTER_KERNELS_GPU(bool);
REGISTER_KERNELS_GPU(Eigen::half);
REGISTER_KERNELS_GPU(float);
REGISTER_KERNELS_GPU(double);
REGISTER_KERNELS_GPU(int64);
REGISTER_KERNELS_GPU(complex64);
REGISTER_KERNELS_GPU(complex128);

#undef REGISTER_KERNELS_GPU
```

### Step 3 - Implement the op kernel(s)

In the op kernel (`multiplex_2_kernel.h`), create a class derived from
[`OpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel)
that implements a `Compute` method to get and validate input tensors, perform
computation, and create the output tensors. This file is included by both
`multiplex_2_kernel.cu.cc` (for GPU) and `multiplex_2_kernel.cc` (for CPU).

```
template <typename Device, typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    const auto& cond_tensor = ctx->input(0);
    const auto& a_values_tensor = ctx->input(1);
    const auto& b_values_tensor = ctx->input(2);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::tensorflow::TensorShapeUtils has some useful functions
    // for checking shapes.
    OP_REQUIRES(ctx, a_values_tensor.shape() == b_values_tensor.shape(),
                ::tensorflow::errors::InvalidArgument(
                    "a and b must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " b shape: ", b_values_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.shape() == cond_tensor.shape(),
                ::tensorflow::errors::InvalidArgument(
                    "a and cond must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " cond shape: ", cond_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.NumElements() > 0,
                ::tensorflow::errors::InvalidArgument(
                    "Inputs must have at least one element."));

    const auto a_values = a_values_tensor.flat<T>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto cond = cond_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, a_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    // Here is an example of processing tensors using the Eigen library.
    // This supports both CPU and GPU.
    // For CPU, it supports chunking into blocks and multi-threading.
    // See
    // https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
    output.device(ctx->eigen_device<Device>()) =
        cond.select(a_values, b_values);
  }
};
```

For intensive mathematical operations, it is a good practice to use
[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview) to
perform the computation. Eigen is vectorized, avoids dynamic memory allocation
and is faster on tensors.The definitions related to Eigen are:

<!-- test_snippets_in_readme skip -->
```c++
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif
```

[Selection](https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55)
from Eigen supports CPU and GPU devices, as well as chunking data into blocks
and multi-threading. The `multiplex_2` op contains the following:

<!-- test_snippets_in_readme skip -->
```c++
output.device(ctx->eigen_device<Device>()) =
     cond.select(a_values, b_values);
```

Using Eigen simplified this example. Alternatively, Custom Ops may implement
kernels for GPU directly in the `*.cu.cc` files using C++.

#### Compile the op

Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.

Create a `BUILD` file for the op which declares the dependencies and the output
build targets. Refer to
[building for OSS](https://www.tensorflow.org/guide/create_op#build_the_op_library).

### Step 4 - Create the Python wrapper

To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.

```
def multiplex(cond, a, b, name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.

  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of dense tensors, no optional parameters, no broadcasting, etc..

  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>

  Args:
    cond: tf.Tensor of type bool. Where True, yield `a`, otherwise yield `b`.
    a: tf.Tensor with the same type and shape as `b`.
    b: tf.Tensor with the same type and shape as `a`.
    name: An optional name for the op.

  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  return gen_multiplex_2_op.examples_multiplex_dense(
      cond=cond, a=a, b=b, name=name)
```

### Step 5 - Test the op

Create op tests using classes derived from
[`tf.test.TestCase`](https://www.tensorflow.org/api_docs/python/tf/test/TestCase).

When writing tests to ensure that the op works correctly in both graph and eager
executions, it is important to note that errors in the op code may be detected
in two distinct phases of code execution depending on how it is executed (eager
or graph executions). Errors may be detected early by the shape function or a
bit later from the logic in the `Compute` method. This may lead to differing
error types and/or messages.

Below are test excerpts showing how to handle errors for different scenarios.
The first test case demonstrates error handling when errors are common across
eager and graph executions and the second test case demonstrates error handling
when the errors are different in eager and graph executions.

```
  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)
```

```
  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples>MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int64 tensor '
        r'\[Op:Examples>MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b' of 'Examples>MultiplexDense' Op has type int64 that "
        r"does not match type float32 of argument 'a'.)"):
      self.evaluate(multiplex_2_op.multiplex(cond, a, b))
```

Refer to `multiplex_2_test.py` for the full source code which contains all the
test cases.

Reuse the `BUILD` file to add build rules for the Python API wrapper and the op
test.

```
py_strict_library(
    name = "multiplex_2_op",
    srcs = ["multiplex_2_op.py"],
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow/examples/custom_ops_doc:__subpackages__"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

cuda_py_test(
    name = "multiplex_2_test",
    size = "small",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_2_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```

Test the op in the following ways:

*   Build for CPU and test on CPU

    <!-- test_snippets_in_readme skip -->
    ```shell
    bazel test //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test
    ```

*   Build for GPU and CPU; test on CPU

    <!-- test_snippets_in_readme skip -->
    ```shell
    $ bazel test --config=cuda //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test
    ```

*   Build for GPU and CPU; test on GPU (note the `_gpu` suffix in the target)

    <!-- test_snippets_in_readme skip -->
    ```shell
    $ bazel test --config=cuda //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test_gpu
    ```

Testing and building exclusively on CPU only requires the multiplex_2_kernel.cc
file when registering the op. For all other cases, include both
multiplex_2_kernel.cc and multiplex_2_kernel.cu.cc files.

### Use the op

Import the op and call it using the following example:

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op

a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
cond = tf.constant([True, False, True, False, True], dtype=bool)
# expected result is [1, 20, 3, 40, 5]
result = multiplex_2_op.multiplex(cond, a, b)
```

Here, `multiplex_2_op` is the name of the Python wrapper that was created in
this example.

When running an op on GPU, use inputs with types supported by the GPU kernels
(e.g. this example uses `tf.int64` for `a` and `b` since this type was
registered).

### Summary

In this example, you learned how to define and use a custom multiplexer op for
GPU. The image below summarizes the files created for this op.

The table below summarizes the build rules and targets for building and testing
the `multiplex_2` op.

Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | `tf_custom_op_library` | `multiplex_2_kernel` | `multiplex_2_kernel.cu.cc`, `multiplex_2_kernel.cc`, `multiplex_2_op.cc`, `multiplex_2_kernel.h`
Wrapper (automatically generated)       | N/A                    | `gen_multiplex_2_op` | N/A
Wrapper (with public API and docstring) | `py_strict_library`    | `multiplex_2_op`     | `multiplex_2_op.py`
Tests                                   | `cuda_py_test`         | `multiplex_2_test`   | `multiplex_2_test.py`
<!-- LINT.ThenChange(multiplex_2.md) -->
