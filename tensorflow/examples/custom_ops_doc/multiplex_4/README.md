<!-- LINT.IfChange -->
# Create a custom multiplexer op with C++ backward compatibility

This guide provides an end-to-end implementation of a new custom op that is
backwards compatible with an existing custom op.

The example in this guide implements a new custom op that handles inputs that
are Python lists of tensors, and is backwards compatible with an existing op
that only handles inputs that are single tensors.

The existing op is a multiplexer that returns elements chosen from either of two
single input tensors `x` or `y` depending on a single `condition` tensor.

The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## A backwards compatible kernel that handles lists of tensors

This example demonstrates how you can create a custom multiplexer,
`multiplex_4`, to register a new kernel that is backward compatible with an
existing multiplex_2` op.

The new custom op registers a kernel
(multiplex_4_kernel.cc) that takes lists of tensors as inputs, and is backwards
compatible with the existing kernel (multiplex_2_kernel.cc) that takes only
single tensors as inputs.

The `multiplex_4` op is similar to
[numpy.select](https://numpy.org/doc/stable/reference/generated/numpy.select.html),
while the `multiplex_2` op is similar to
[numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html).

The lists of tensors that the new op takes as inputs are of a particular fixed
size. Since the list size is defined in `Attr`, it is fixed at graph
construction time when the constructor of the C++ kernel is called. Therefore,
the size of the list cannot be data dependent. See
[Ragged tensors](https://www.tensorflow.org/guide/ragged_tensor) for
variable length lists.

This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.

### Prerequsites - Implement `multiplex_2` and `SavedModel`

This example uses a [`SavedModel`](https://www.tensorflow.org/guide/saved_model)
from an existing `multiplex_2` custom op.

The `muliplex_2_save.py` file uses `save` from `model_using_muliplex.py` to
create a `SavedModel` named `model_using_multiplex` in the current working
directory.

```
def save(multiplex_op, path):
  """Save a model that contains the given `multiplex_op`.

  Args:
    multiplex_op: A multiplex Custom Op, e.g. multiplex_4_op.multiplex. This is
      parameterized so it can also be used to create an "old" model with an
      older version of the op, e.g. multiplex_2_op.multiplex.
    path: Directory to save model to.
  """
  example_cond, example_a, example_b = _get_example_tensors()

  class UseMultiplex(tf.Module):

    @tf.function(input_signature=[
        tf.TensorSpec.from_tensor(example_cond),
        tf.TensorSpec.from_tensor(example_a),
        tf.TensorSpec.from_tensor(example_b)
    ])
    def use_multiplex(self, cond, a, b):
      return multiplex_op(cond, a, b)

  model = UseMultiplex()
  tf.saved_model.save(
      model,
      path,
      signatures=model.use_multiplex.get_concrete_function(
          tf.TensorSpec.from_tensor(example_cond),
          tf.TensorSpec.from_tensor(example_a),
          tf.TensorSpec.from_tensor(example_b)))
```

This `SavedModel` has the old version of the custom op (`multiplex_2`) that only
supports individual tensors as inputs. The following steps will register a
kernel that accepts lists of tensors as inputs, while maintaining backward
compatability with the previous op.

### Step 1 - Define the op interface

Define the op interface and register it using the `REGISTER_OP` macro.

```
REGISTER_OP("Examples>MultiplexDense")
    .Input("cond: N * bool")
    .Input("a_values: N * T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .Attr("N: int = 1")
    .SetShapeFn(MultiplexShapeFunction)
    .Doc(R"doc(
Return elements chosen from `a_values` or `b_values` depending on `cond`.

When `a_values` and `cond` are tenors (i.e. N=1), this is similar to `np.where`
and `tf.where`. When `a_values` and `cond` are lists of tensors (i.e. N>1),
this is similar to `np.select`. In either case these are simplified to only
handle dense tensors, no optional parameters, no broadcasting, etc..

cond: tf.Tensor or list of tf.Tensor of type bool. If it is a list, `a_values`
      must be a list of the same length. Where True, yield the corresponding
      element from `a_values` (with priority to the first one encountered in
      lists), otherwise yield `b_values`.
a_values: tf.Tensor or list of tf.Tensor. Each tensor has the same type and
          shape as `b_values`. If it is a list, `cond` must be a list of the
          same length.
b_values: tf.Tensor with the same type and shape as the `a_values` if it is a
          tensor or as every element of `a_values` if `a_values` is a list.
output_values: A tf.Tensor with elements from `a_values` where `cond` is True,
               and elements from `b` elsewhere.
)doc");
```

While the `multiplex_2` op defined inputs as single tensors, such as `cond:
bool` and `a_values: T`, this op supports lists of tensors by adding `N*`, where
`N` is the length of the lists.

The default list size (`N`) is set to 1 with the following: `.Attr("N: int =
1")`. If the inputs are single tensors, then `N` is equal to 1, which is
backwards compatible with a previous definition of `.Input("x: T")`.

All lists in this example are of equal length (`N`). To support lists of
different lengths, define an attribute for each unique length. For example:

<!-- test_snippets_in_readme skip -->
```c++
.Input("short_list: short_len * float")
.Input("long_list: long_len * float")
.Attr("short_len: int = 1")
.Attr("long_len: int >= 10")
```

### Step 2 - Register the op implementation (kernel)

The C++ kernel in `multiplex_4_kernel.cc` implements a multiplexer that accepts
lists of tensors as inputs. Register the kernel by calling the
`REGISTER_KERNEL_BUILDER` macro.

```
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
```

### Step 3 - Implement the op kernel

In the `multiplex_4_kernel.cc` op kernel, create a class derived from
[`OpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel)
that implements a `Compute` method. This method retrieves and validates input
tensors, performs computation, and creates output tensors.

```
template <typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_cond_a_));
  }

  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    // Optional error checking: cond and a_values are lists of N, so there are
    // a total of 2N+1 inputs. Check that the  number of inputs and the
    // `N` Attr is consistent.
    const int64_t expected_inputs = 2 * num_cond_a_ + 1;
    OP_REQUIRES(ctx, expected_inputs == ctx->num_inputs(),
                Internal("expected_inputs != num_inputs(): ", expected_inputs,
                         " != ", ctx->num_inputs()));
    VLOG(1) << "N " << num_cond_a_;

    const auto& first_cond_tensor = ctx->input(0);
    const auto& first_a_values_tensor = ctx->input(num_cond_a_);
    const auto& b_values_tensor = ctx->input(2 * num_cond_a_);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::tensorflow::TensorShapeUtils has some useful functions
    // for checking shapes.
    for (int64_t i = 0; i < num_cond_a_; i++) {
      const auto& cond_tensor_i = ctx->input(i);
      const auto& a_values_tensor_i = ctx->input(num_cond_a_ + i);
      OP_REQUIRES(
          ctx, a_values_tensor_i.shape() == b_values_tensor.shape(),
          InvalidArgument(
              "a_values[", i,
              "] and b_values must have the same shape. "
              "a_values[",
              i, "] shape: ", a_values_tensor_i.DebugString(),
              " b_values shape: ", b_values_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx, cond_tensor_i.shape() == b_values_tensor.shape(),
          InvalidArgument(
              "cond_values[", i,
              "] and b_valuesmust have the same shape. "
              "cond_values[",
              i, "] shape: ", first_a_values_tensor.shape().DebugString(),
              " b_values shape: ", first_cond_tensor.shape().DebugString()));
    }

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, b_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();

    const auto b_values = b_values_tensor.template flat<T>();
    // np.select style behavior, `cond` and `a_values` are lists of tensors.
    // Also works for the np.where style case where there is only one `cond`
    // and one `a_values` tensor.
    const int64_t N = first_a_values_tensor.NumElements();
    for (int64_t i = 0; i < N; i++) {
      bool flag = false;
      for (int64_t list_index = 0; list_index < num_cond_a_; list_index++) {
        const auto& cond_tensor = ctx->input(list_index);
        const auto& a_values_tensor = ctx->input(num_cond_a_ + list_index);
        const auto cond = cond_tensor.template flat<bool>();
        const auto a_values = a_values_tensor.template flat<T>();
        if (cond(i)) {
          output(i) = a_values(i);
          flag = true;
          VLOG(1) << "A " << list_index << " for " << i;
          break;
        }
      }
      if (!flag) {
        output(i) = b_values(i);
        VLOG(1) << "B for " << i;
      }
    }
  }

 private:
  int64_t num_cond_a_;  // the number of `cond` and `a` input tensors
};
```

The kernel uses a private member variable (`num_cond_a_`) to hold the length of
`cond` and `a`. The constructor saves the `N` attribute into the variable.

<!-- test_snippets_in_readme skip -->
```c++
private:
  int64_t num_cond_a_;  // the number of cond and a input tensors
```

<!-- test_snippets_in_readme skip -->
```c++
explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_cond_a_));
}
```

The `num_cond_a_` variable is used to index the inputs in the following order:
`cond`, `a`, `b`. The op interfaces specify that `cond` and `a` are tensor lists
of length `N`, and `b` is a single tensor. The inputs are indexed as follows:

1.  `cond`: [0 ... N-1]
2.  `a`: [N ... 2*N-1]
3.  `b`: [2*N]

When `num_cond_a_` is equal to 1, the kernel implements `numpy.where` as it
would in the `multiplex_2` op. When `num_cond_a_` is greater than 1, the kernel
implements `numpy.select`. This is achieved with the following `for` loop.

<!-- test_snippets_in_readme skip -->
```c++
for (int64_t i = 0; i < N; i++) {
  bool flag = false;
  for (int64_t list_index = 0; list_index < num_cond_a_; list_index++) {
    const auto& cond_tensor = ctx->input(list_index);
    const auto& a_values_tensor = ctx->input(num_cond_a_ + list_index);
    const auto cond = cond_tensor.flat<bool>();
    const auto a_values = a_values_tensor.flat<T>();
    if (cond(i)) {
      output(i) = a_values(i);
      flag = true;
      break;
    }
  }
  if (!flag) {
    output(i) = b_values(i);
  }
}
```

#### Compile the op

Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.

Create a `BUILD` file for the op which declares the dependencies and the output
build targets. Refer to
[building for OSS](https://www.tensorflow.org/guide/create_op#build_the_op_library).

### Step 4 - Create the Python wrapper

To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.

If `cond` and `a` are not already lists, the wrapper in `multiplex_4_op.py`
puts the variables in lists before the `numpy.where` implementation.

Note: The generated Python wrapper automatically sets the `N` attribute based on
the length of the input lists.

```
def multiplex(cond, a, b, name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.

  This is similar to `np.where` and `tf.where` if `cond` and `a` are tensors.
  This is similar to `np.select` if `cond` and `a` are lists of tensors.
  In either case, this is simplified to only handle the case of dense tensors,
  no optional parameters, no broadcasting, etc..

  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>

  >>> a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
  >>> a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
  >>> a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
  >>> b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
  >>> cond1 = tf.constant([False, False, True, False, False], dtype=bool)
  >>> cond2 = tf.constant([False, False, False, False, True], dtype=bool)
  >>> cond3 = tf.constant([True, False, True, False, True], dtype=bool)
  >>> multiplex_4_op.multiplex([cond1, cond2, cond3], [a1, a2, a3], b)
  <tf.Tensor: shape=(5,), ... numpy=array([ 11, 102,   3, 104,  10], ...)>

  Args:
    cond: tf.Tensor or list of tf.Tensor of type bool. Where True, yield `a`.
      When muliple corresponding `cond` elements are true, the first one yield
      based on the first one encountered.
    a: tf.Tensor or list of tf.Tensor, each with the same type and shape as `b`.
    b: tf.Tensor or list of tf.Tensor with the same type and shape as `a`. Yield
      `b` if all corresponding `cond` values is False.
    name: An optional name for the op.

  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  if not isinstance(cond, (list, tuple)):
    # Support "old" use of multiplex where `cond` and `a` are tensors,
    # not lists of tensors.
    return gen_multiplex_4_op.examples_multiplex_dense(
        cond=[cond], a_values=[a], b_values=b, name=name)
  return gen_multiplex_4_op.examples_multiplex_dense(
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
error types and messages.

```
@test_util.with_eager_op_as_function
class MultiplexOpTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_select(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a = [a1, a2, a3]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond = [cond1, cond2, cond3]
    expect = np.select([self.evaluate(i) for i in cond],
                       [self.evaluate(i) for i in a], self.evaluate(b))
    # expected result is [11, 102, 3, 104, 10]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  def test_multiplex_saved_model(self):
    path = os.path.join(self.create_tempdir(), 'model')
    model_using_multiplex.save(multiplex_4_op.multiplex, path)
    result = model_using_multiplex.load_and_use(path)
    self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))

  # One tf.function that uses both multiplex with single tensors for `cond`
  # and `a` and with lists of tensors for `cond` and `a`, i.e. a graph
  # with two example_multiplex_dense kernels that have different numbers
  # of inputs.
  @tf.function
  def _both(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a_123 = [a1, a2, a3]
    b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond_123 = [cond1, cond2, cond3]
    mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
    b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
    cond4 = tf.constant([True, True, True, False, False], dtype=bool)
    result = multiplex_4_op.multiplex(cond4, mux_123, b4)
    return result

  def test_both_single_and_list(self):
    result = self._both()
    self.assertAllEqual(result,
                        tf.constant([11, 102, 3, 204, 205], dtype=tf.int64))

  @test_util.run_in_graph_and_eager_modes
  def test_inconsistent_inputs_error(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a = [a1, a2]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond = tf.constant([False, False, True, False, False], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(a_values\[0\] and b_values must have the same shape'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Shapes must be equal rank, but are 2 and 1)'):
      self.evaluate(multiplex_4_op.multiplex(cond, a, b))
```

The following `tf.function` in muliplex_4_test.py has two multiplex custom ops:
one that takes lists for its `cond` and `a` inputs, and another that takes
single tensors.

```
  @tf.function
  def _both(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a_123 = [a1, a2, a3]
    b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond_123 = [cond1, cond2, cond3]
    mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
    b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
    cond4 = tf.constant([True, True, True, False, False], dtype=bool)
    result = multiplex_4_op.multiplex(cond4, mux_123, b4)
    return result
```

The model_using_multiplex.py file has functions for creating and using a saved
custom op model `SavedModel`. In this test, the `multiplex_4` op is used to both
save and use models.

```
  def test_multiplex_saved_model(self):
    path = os.path.join(self.create_tempdir(), 'model')
    model_using_multiplex.save(multiplex_4_op.multiplex, path)
    result = model_using_multiplex.load_and_use(path)
    self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))
```

Test the op with the following:

<!-- test_snippets_in_readme skip -->
```shell
bazel test //third_party/tensorflow/google/g3doc/example/multiplex_4:multiplex_4_test
```

Reuse the `BUILD` file to add build rules for the Python API wrapper and the op
test.

```
py_strict_library(
    name = "multiplex_4_op",
    srcs = ["multiplex_4_op.py"],
    data = ["multiplex_4_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)
```
<!-- test_snippets_in_readme skip -->
```
tf_py_test(
    name = "multiplex_4_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["multiplex_4_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",
        "no_pip",
    ],
    deps = [
        ":model_using_multiplex",
        ":multiplex_4_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```

### Use the op

Build the op with the following:

<!-- test_snippets_in_readme skip -->
```shell
bazel build //third_party/tensorflow/examples/custom_ops_doc/multiplex_4:multiplex_4_op
```

Import the op and call it using the following example:

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_4 import multiplex_4_op

a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
a = [a1, a2, a3]
b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
cond1 = tf.constant([False, False, True, False, False], dtype=bool)
cond2 = tf.constant([False, False, False, False, True], dtype=bool)
cond3 = tf.constant([True, False, True, False, True], dtype=bool)
cond = [cond1, cond2, cond3]
# expected result is [11, 102, 3, 104, 10]
result = multiplex_4_op.multiplex(cond, a, b)
```

The `multiplex_4_load_use.py` file uses `load_and_use` from
`model_using_muliplex.py` to load a saved model from a `multiplex_2` op. The
saved model can be executed using the new kernel, (`multiplex_4`), which
supports both lists of tensors and single tensors for `cond` and `a` inputs.

Since `Examples>MultiplexDense` can only be defined once in a binary, there must
be two separate binaries. A binary can either depend on `multiplex_2_op` or
`multiplex_4_op`, but not both. The custom ops are backward compatible, so we
can use `save` on `multiplex_2` and `load_and_use` on `multiplex_4`.

### Summary

In this example, you learned how to implement a new multiplexer kernel that is
backwards compatible with an existing multiplexer kernel. This custom op handles
inputs that are lists of tensors, while continuing to handle inputs of single
tensors.

The tables below summarize the build rules and targets for building and testing
the `multiplex_4` op.

#### Kernel components

Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | `tf_custom_op_library` | `multiplex_4_kernel` | `multiplex_4_kernel.cc`, `multiplex_4_op.cc`
Wrapper (automatically generated)       | N/A                    | `gen_multiplex_4_op` | N/A
Wrapper (with public API and docstring) | `py_strict_library`    | `multiplex_4_op`     | `multiplex_4_op.py`
Tests                                   | `tf_py_test`           | `multiplex_4_test`   | `multiplex_4_test.py`

##### Usage example

Op components            | Build rule          | Build target               | Source
------------------------ | ------------------- | -------------------------- | ------
Common library           | `py_strict_library` | `model_using_multiplex`    | `model_using_multiplex.py`
Old op (with SavedModel) | `py_strict_binary`  | `multiplex_2_save`         | `multiplex_2_save.py`
New op (with SavedModel) | `py_strict_binary`  | `multiplex_4_load_and_use` | `multiplex_4_load_and_use.py`

## Resources

*   [OSS custom ops guide](https://www.tensorflow.org/guide/create_op)
*   [SavedModel](https://www.tensorflow.org/guide/saved_model)
*   [Numpy Select](https://numpy.org/doc/stable/reference/generated/numpy.select.html)
<!-- LINT.ThenChange(multiplex_4.md) -->
