# Create a custom multiplexer op

This page provides an end-to-end example for adding a custom multiplexer op to
TensorFlow. For additional context,
read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## Creating a custom multiplexer op

This examples demonstrates how you can create a Python custom multiplexer
`multiplex_1_op`, similar to
[`tf.where`](https://tensorflow.org/api_docs/python/tf/where?version=nightly)
which you can call as:

<!-- test_snippets_in_readme skip -->
```python
multiplex_1_op.multiplex(condition, x, y)                                        # doctest: skip
```

This custom op returns elements chosen from either of the two input tensors `x`
or `y` depending on the `condition`.

Example usage:

<!-- test_snippets_in_readme skip -->
```python
from tensorflow.examples.custom_ops_doc.multiplex_1 import multiplex_1_op

m = multiplex_1_op.multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
m.numpy()
```

<!-- test_snippets_in_readme skip -->
```
array([  1, 200, 300,   4], dtype=int32)
```

Note that this simplified `multiplex_1` op has limitations that are not present
in `tf.where` such as:

*   Support only for dense tensors
*   Support only for CPU computations
*   No broadcasting capabilities
*   No extensibility through optional parameters

The example below contains C++ and Python code snippets to illustrate the code
flow. These snippets are not all complete; some are missing namespace
declarations, imports, and test cases.

### Step 1 - Define op interface

Define the op interface and register it using the `REGISTER_OP` macro.

<!-- test_snippets_in_readme skip -->
```
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
```

```
REGISTER_OP("Examples1>MultiplexDense")
    .Input("cond: bool")
    .Input("a_values: T")
    .Input("b_values: T")
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

cond: tf.Tensor of type bool.
a_values: tf.Tensor with the same type and shape as `b_values`.
b_values: tf.Tensor with the same type and shape as `a_values`.

      Where True, yield `a_values`, otherwise yield `b_values`.
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
    binary ops) and has error checking that the two inputs have the same shape.
    Since `multiplex_1` has three inputs, two calls to `Merge` are used to
    assert that all three inputs are the same shape.

### Step 2 - Register the op implementation (kernel)

Register the kernel by calling the `REGISTER_KERNEL_BUILDER` macro.

```
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples1>MultiplexDense")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
```

### Step 3 - Implement the op kernel(s)

In the op kernel in `multiplex_1_kernel.cc`, create a class derived from
[`OpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel)
that implements a `Compute` method to get and validate input tensors, perform
computation, and create the output tensors.

```
template <typename T>
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
                InvalidArgument(
                    "a_values and b_values must have the same shape. "
                    "a_values shape: ",
                    a_values_tensor.shape().DebugString(), " b_values shape: ",
                    b_values_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, a_values_tensor.shape() == cond_tensor.shape(),
        InvalidArgument("a_values and cond must have the same shape. "
                        "a_values shape: ",
                        a_values_tensor.shape().DebugString(),
                        " cond shape: ", cond_tensor.shape().DebugString()));

    const auto a_values = a_values_tensor.flat<T>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto cond = cond_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, a_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    const int64_t N = a_values_tensor.NumElements();

    // Here is an example of processing tensors in a simple loop directly
    // without relying on any libraries. For intensive math operations, it is
    // a good practice to use libraries such as Eigen that support
    // tensors when possible, e.g. "output = cond.select(a_values, b_values);"
    // Eigen supports chunking into blocks and multi-threading.
    // See
    // https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
    for (int64_t i = 0; i < N; i++) {
      if (cond(i)) {
        output(i) = a_values(i);
      } else {
        output(i) = b_values(i);
      }
    }
  }
};
```

A common way to access the values in tensors for manipulation is to get
flattened rank-1
[`Eigen::Tensor`](https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html)
objects. In the example code, this is done for all three inputs and the output.
The example also processes tensors in a simple loop directly without relying on
any libraries.

Using Eigen, the `for` loop above could have been written simply as:

<!-- test_snippets_in_readme skip -->
```c++
output = cond.select(a_values, b_values);
```

[Selection](https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55)
from Eigen supports chunking into blocks and multi-threading.

For intensive mathematical operations, it is a good practice to use libraries
such as [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview)
that support tensors to do the computation when possible. Eigen is vectorized,
avoids dynamic memory allocation and therefore is typically faster than using
simple `for` loops.

Eigen provides the following for accessing tensor values (for both inputs and
outputs):

*   `flat<T>()(index)` for element access for tensors of any rank
*   `scalar<T>()()` for rank 0 tensors
*   `vec<T>()(index)` for rank 1 tensors
*   `matrix<T>()(i, j)` for rank 2 tensors
*   `tensor<T, 3>()(i, j, k)` for tensors of known rank (e.g. 3).

#### Compile the op (optional)

Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.

Create a `BUILD` file for the op which declares the dependencies and the output
build targets. Refer to
[building for OSS](https://www.tensorflow.org/guide/create_op#build_the_op_library).

### Step 4 - Create the Python wrapper (optional)

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
  return gen_multiplex_1_op.examples1_multiplex_dense(
      cond=cond, a_values=a, b_values=b, name=name)
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
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([10, 20, 30, 40, 50])
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_1_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)
```

```
  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50])  # int32
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples1>MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int32 tensor '
        r'\[Op:Examples1>MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b_values' of 'Examples1>MultiplexDense' Op has type int32 that "
        r"does not match type float32 of argument 'a_values'.)"):
      self.evaluate(multiplex_1_op.multiplex(cond, a, b))
```

Refer to `multiplex_1_test.py` for the full source code which contains all the
test cases.

Reuse the `BUILD` file created in Step 3a above to add build
rules for the Python API wrapper and the op test.

<!-- test_snippets_in_readme skip -->
```
load("//third_party/tensorflow:strict.default.bzl", "py_strict_library")
load("//third_party/tensorflow:tensorflow.default.bzl", "tf_py_test")
```

```
py_strict_library(
    name = "multiplex_1_op",
    srcs = ["multiplex_1_op.py"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow/google/g3doc:__subpackages__"],
    deps = [
        ":gen_multiplex_1_op",
        ":multiplex_1_kernel",
        "//third_party/py/tensorflow",
    ],
)

tf_py_test(
    name = "multiplex_1_test",
    size = "small",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":multiplex_1_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```

Test the op by running:

<!-- test_snippets_in_readme skip -->
```shell
$ bazel test //third_party/tensorflow/google/g3doc/example/multiplex_1:multiplex_1_test
```

### Use the op

Use the op by importing and calling it as follows:

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_1 import multiplex_1_op

a = tf.constant([1, 2, 3, 4, 5])
b = tf.constant([10, 20, 30, 40, 50])
cond = tf.constant([True, False, True, False, True], dtype=bool)

result = multiplex_1_op.multiplex(cond, a, b)
result.numpy()
```

<!-- test_snippets_in_readme skip -->
```
 array([ 1, 20,  3, 40,  5], dtype=int32)
```

Here, `multiplex_1_op` is the name of the Python wrapper that was created in
this example.

### Summary

In this example, you learned how to define and use a custom multiplexer op. The
image below summarizes the files created for this op.

The table below summarizes the build rules and targets for building and testing
the `multiplex_1` op.

Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | `tf_kernel_library`    | `multiplex_1_kernel` | `multiplex_1_kernel.cc`, `multiplex_1_op.cc`
Wrapper (automatically generated)       | `tf_gen_op_wrapper.py` | `gen_multiplex_1_op` | N/A
Wrapper (with public API and docstring) | `py_strict_library`    | `multiplex_1_op`     | `multiplex_1_op.py`
Tests                                   | `tf_py_test`           | `multiplex_1_test`   | `multiplex_1_test.py`

