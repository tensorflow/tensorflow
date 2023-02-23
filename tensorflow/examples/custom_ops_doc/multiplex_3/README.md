# Create a custom multiplexer op with dispatch to special case kernels

This guide provides an end-to-end example of handling special cases with a new
C++ kernel. The custom op includes a Python wrapper that uses
[dispatch decorators](https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch)
to override the default behavior of TensorFlow operations when applied to
tensor-like types. For more information, refer to
[extension types](https://www.tensorflow.org/guide/extension_type).

Special case kernels can add new functionality to an existing op without any
required changes to existing kernels that have already been registered. For
example, a special case kernel can enable an existing op to handle a different
type of input.

Optional Python wrappers can enable a variety of non-breaking future changes,
though it is important to avoid any non-TensorFlow Python code in the
implementation. This is because any non-Tensorflow Python code will only be used
in eager execution and not in
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)
execution.

Python wrappers can serve the following purposes:

*   **Handling special cases**: The default C++ kernel handles normal cases,
    while another C++ kernel handles special cases. For example,
    [`tf.add`](https://www.tensorflow.org/api_docs/python/tf/math/add) calls one
    of two different C++ kernels, depending on whether the inputs are strings or
    not.

*   **Decomposing operations into multiple kernels**: Some operations at the
    Python level are decomposed into multiple C++ kernels, rather than
    implemented through a single kernel. For example, there is no
    `ReduceVariance` kernel for the
    [`tf.reduce_variance`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance)
    op. Instead, the Python `reduce_variance` function computes the variance
    based on squared deviations from the mean.

*   **Adding an argument to an existing op**: Since `name` is always the last
    argument wrapper, adding a new, optional argument requires a new wrapper.
    This prevents the op from mistaking the new argument for the `name`
    argument.

*   **Changing the order of arguments**: Similar to the point above, a wrapper
    can be used to change the order of arguments.

The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## Dispatch to special case kernels using sparse tensors

This example demonstrates how you can create a Python custom multiplexer,
`multiplex_3_op`, to register a new kernel. If an existing op already handles
certain kinds of inputs, a special case kernel can extend the op to handle a
different kind of input without changing the existing op.

The special kernel (`multiplex_3`) in this example extends an existing op
(`multiplex_2`) so that it can handle
[sparse tensors](https://www.tensorflow.org/guide/sparse_tensor) as inputs. This
provides the custom op with the following two kernels:

*   **Default kernel**: registers the multiplex op with dense tensors
    (`MultiplexDense`).
*   **Special case kernel**: registers the multiplex op with sparse tensors
    (`MultiplexSparse`).

The sparse tensor object (`tf.SparseTensor`) is appropriate for tensors that
contain missing values. Storing sparse values in sparse tensors is more
memory-efficient than storing in a dense tensor.

In this example, the default kernel is the
`multiplex_2` (multiplex_2_kernel.cc) kernel, and the new kernel
(multiplex_3_kernel.cc) implements the multiplex with sparse tensors.

Like other multiplex custom ops, `multiplex_3` is similar to
[`tf.where`](https://tensorflow.org/api_docs/python/tf/where?version=nightly).
It returns elements chosen from either of the two input tensors (`x` or `y`),
depending on the `condition`. You can call the op with the following:

<!-- test_snippets_in_readme skip -->
```python
multiplex_3_op.multiplex(condition, x, y)
```

This simplified `multiplex_3` op has the following limitations that are not
present in `tf.where`:

*   Support only for CPU computations
*   No broadcasting capabilities
*   No extensibility through optional parameters

This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.

### Step 1 - Define the op interface

Define the op interface and register it using the `REGISTER_OP` macro.

```
REGISTER_OP("Examples>MultiplexSparse")
    .Input("cond_indices: int64")
    .Input("cond_values: bool")
    .Input("cond_shape: int64")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // cond_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // cond_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));  // cond_shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));  // a_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused));  // a_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &unused));  // a_shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &unused));  // b_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &unused));  // b_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &unused));  // b_shape
      const auto num_rows = c->UnknownDim();
      const auto dense_rank = c->UnknownDim();
      c->set_output(0, c->Matrix(num_rows, dense_rank));
      c->set_output(1, c->Vector(num_rows));
      c->set_output(2, c->Vector(dense_rank));
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Return elements chosen from `a` or `b` depending on `cond`.

This is similar to `np.where` and `tf.where`, but simplified to only handle
the case of sparse tensors that are vectors, no optional parameters,
no broadcasting, etc.. Elements for `a` are chosen if there is a `true` `cond`
value at the same position. Elements for `b` are chosen if there is not a `true`
`cond` value at the same position, i.e., if either there is a `false` `cond`
value or the `cond` value is not specified.

Indices must be ordered as described by tf.sparse_reorder.

cond_indices: a rank-2 tensor of sparse indices.
cond_values: a rank-1 tensor of sparse values.
cond_shape: a rank-1 tensor representing the dense shape.
a_indices: a rank-2 tensor of sparse indices.
a_values: a rank-1 tensor of sparse values.
a_shape: a rank-1 tensor representing the dense shape.
b_indices: a rank-2 tensor of sparse indices.
b_values: a rank-1 tensor of sparse values.
b_shape: a rank-1 tensor representing the dense shape.
output_indices: a rank-2 tensor of sparse indices.
output_values: a rank-1 tensor of sparse values.
output_shape: a rank-1 tensor representing the dense shape.
)doc");
```

#### Inputs and outputs

This op contains a total of nine input tensors. This is made up of three sparse
tensors (`a`, `b`, and `cond`), where each sparse tensor is encoded using the
coordinate list (COO) format:

*   `values`: 1D tensor with shape `[N]` containing all non-zero values.
*   `indices`: 2D tensor with shape `[N, rank]`, containing the indices of the
    non-zero values
*   `dense_shape`: 1D tensor with shape `[rank]`, specifying the shape of the
    tensor.

The `cond` tensor accepts a boolean value to select between `a` and `b`, and the
`a` and `b` tensors accept a value of type `T`. The output tensor also contains
a value of type `T`.

#### Shape function

Unlike dense tensors, which have a fixed shape, the shape of sparse tensors
depend on the number of non-missing values in the output. Since this can not be
determined by the shape of the inputs, the shape function (`SetShapeFn`) uses
`UnknownDim()`.

<!-- test_snippets_in_readme skip -->
```c++
  .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      // Error checking omitted, see source file.
      const auto num_rows = c->UnknownDim();
      const auto dense_rank = c->UnknownDim();
      c->set_output(0, c->Matrix(num_rows, dense_rank));
      c->set_output(1, c->Vector(num_rows));
      c->set_output(2, c->Vector(dense_rank));
      return tensorflow::OkStatus();
    })
```

#### Attributes and docstrings

The `Attr` for this op is defined as `.Attr("T: type")`, which specifies `T` as
an `Attr` of type `type`. In the subsequent steps, you will use `T` with a
template class to define the type of the contents of tensors.

The docstring for this op is specified by passing a string to `.Doc()`.

### Step 2 - Register the op implementation (kernel)

The C++ kernel in `multiplex_3_kernel.cc` implements a multiplex for sparse
tensors. For simplicity, this example only supports rank 1 sparse tensors
(sparse vectors).

Register the kernel by calling the `REGISTER_KERNEL_BUILDER` macro.

```
#define REGISTER_KERNELS_CPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexSparse")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexSparseOp<type>)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
```

### Step 3 - Implement the op kernel(s)

In the `multiplex_3_kernel.cc` op kernel, create a class derived from
[`OpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel)
that implements a `Compute` method. This method retrieves and validates input
tensors, performs computation, and creates output tensors.

```
template <typename T>
class MultiplexSparseOp : public OpKernel {
 public:
  explicit MultiplexSparseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexSparseOp(const MultiplexSparseOp& other) = delete;
  MultiplexSparseOp& operator=(const MultiplexSparseOp& other) = delete;
  ~MultiplexSparseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    const auto& cond_indices_tensor = ctx->input(0);
    const auto& cond_values_tensor = ctx->input(1);
    const auto& cond_shape_tensor = ctx->input(2);
    const auto& a_indices_tensor = ctx->input(3);
    const auto& a_values_tensor = ctx->input(4);
    const auto& a_shape_tensor = ctx->input(5);
    const auto& b_indices_tensor = ctx->input(6);
    const auto& b_values_tensor = ctx->input(7);
    const auto& b_shape_tensor = ctx->input(8);
    OP_REQUIRES_OK(ctx,
                   ValidateSparseTensor(cond_indices_tensor, cond_values_tensor,
                                        cond_shape_tensor, "cond"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(a_indices_tensor, a_values_tensor,
                                             a_shape_tensor, "a"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(b_indices_tensor, b_values_tensor,
                                             b_shape_tensor, "b"));
    OP_REQUIRES(
        ctx, cond_shape_tensor.shape() == a_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. cond_shape: ",
                        cond_shape_tensor.shape().DebugString(),
                        " vs a_shape: ", a_shape_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, a_shape_tensor.shape() == b_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. a_shape: ",
                        a_shape_tensor.shape().DebugString(),
                        " vs b_shape: ", b_shape_tensor.shape().DebugString()));
    const int rank = a_shape_tensor.dim_size(0);
    OP_REQUIRES(
        ctx, rank == 1,
        InvalidArgument("Sorry, multiplex for sparse tensors only "
                        "supports rank 1 tensors to simplify this example."));
    const int cond_elements = cond_indices_tensor.dim_size(0);
    const int a_elements = a_indices_tensor.dim_size(0);
    const int b_elements = b_indices_tensor.dim_size(0);
    const auto cond_indices = cond_indices_tensor.matrix<int64_t>();
    const auto cond_values = cond_values_tensor.flat<bool>();
    const auto cond_shape = cond_shape_tensor.flat<int64_t>();
    const auto a_indices = a_indices_tensor.matrix<int64_t>();
    const auto a_values = a_values_tensor.flat<T>();
    const auto a_shape = a_shape_tensor.flat<int64_t>();
    const auto b_indices = b_indices_tensor.matrix<int64_t>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto b_shape = b_shape_tensor.flat<int64_t>();
    int cond_index = 0;
    int a_index = 0;
    int b_index = 0;
    // This vector is a list of source tensors (a = true, b = false) and source
    // indices.
    std::vector<std::pair<bool, int>> merged_output;
    merged_output.reserve(std::min(cond_elements, a_elements) + b_elements);
    while (a_index < a_elements || b_index < b_elements) {
      // Determine the whether the current location with values has a value
      // for `a`, for `b` or for both `a` and `b`.
      int64_t cur_row;
      bool is_a_at_cur = false;
      bool is_b_at_cur = false;
      if (a_index < a_elements && b_index < b_elements) {
        const int64_t a_row = a_indices(a_index, 0);
        const int64_t b_row = b_indices(b_index, 0);
        cur_row = std::min(a_row, b_row);
        if (a_row == cur_row) {
          is_a_at_cur = true;
        }
        if (b_row == cur_row) {
          is_b_at_cur = true;
        }
      } else if (a_index < a_elements) {
        cur_row = a_indices(a_index, 0);
        is_a_at_cur = true;
      } else {  // b_index < b_elements
        cur_row = b_indices(b_index, 0);
        is_b_at_cur = true;
      }
      // Deterimine if `cond` has a value at the current location
      bool cond_flag = false;
      while (cond_index < cond_elements) {
        const int64_t cond_row = cond_indices(cond_index, 0);
        if (cond_row > cur_row) {
          break;
        }
        if (cond_row == cur_row) {
          cond_flag = cond_values(cond_index);
          break;
        }
        ++cond_index;
      }
      // Add `a` or `b` to the merged output based on the condition
      if (is_a_at_cur) {
        if (cond_flag) {
          merged_output.emplace_back(true, a_index);
        }
        ++a_index;
      }
      if (is_b_at_cur) {
        if (!cond_flag) {
          merged_output.emplace_back(false, b_index);
        }
        ++b_index;
      }
    }

    // Allocate output tensors.
    Tensor* output_indices_tensor;
    Tensor* output_values_tensor;
    Tensor* output_dense_shape_tensor;
    const int num_values = merged_output.size();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_values, rank}),
                                             &output_indices_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({num_values}),
                                             &output_values_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({rank}),
                                             &output_dense_shape_tensor));
    auto output_indices = output_indices_tensor->matrix<int64_t>();
    auto output_values = output_values_tensor->flat<T>();
    auto output_shape = output_dense_shape_tensor->flat<int64_t>();
    for (int row = 0; row < num_values; ++row) {
      const auto& source_flag = merged_output[row].first;
      const auto& source_row = merged_output[row].second;
      const auto& indices = source_flag ? a_indices : b_indices;
      const auto& values = source_flag ? a_values : b_values;
      for (int column = 0; column < rank; ++column) {
        output_indices(row, column) = indices(source_row, column);
      }
      output_values(row) = values(source_row);
    }
    // Expand the shape of the output sparse tensor so that it is as large
    // as the shape of the largest input in each dimension.
    // An alternative behavoir would be to require that the shapes be the
    // same and implement error checking that all the corresponding values
    // in the shape tensors are the same (e.g.
    // `cond_shape(i) == a_shape(i)` and `a_shape(i) == b_shape(i)` in
    // OP_REQUIRES above and `output_shape(i) = a_shape(i)` here).
    for (int i = 0; i < rank; ++i) {
      output_shape(i) =
          std::max(cond_shape(i), std::max(a_shape(i), b_shape(i)));
    }
  }

 private:
  Status ValidateSparseTensor(const ::tensorflow::Tensor& indices_tensor,
                              const ::tensorflow::Tensor& values_tensor,
                              const ::tensorflow::Tensor& shape_tensor,
                              const string label) {
    if (!TensorShapeUtils::IsMatrix(indices_tensor.shape())) {
      return InvalidArgument(
          "Sparse indices for ", label,
          " must be rank 2, not shape: ", indices_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(values_tensor.shape())) {
      return InvalidArgument("Sparse values for ", label,
                             " must be a vector, not shape: ",
                             values_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(shape_tensor.shape())) {
      return InvalidArgument(
          "Sparse shape for ", label,
          " must be a vector, not shape: ", shape_tensor.shape().DebugString());
    }
    if (indices_tensor.dim_size(0) != values_tensor.dim_size(0)) {
      return InvalidArgument("Sparse indices and values for " + label +
                                 " must have the same "
                                 "number of rows. indices: ",
                             indices_tensor.shape().DebugString(),
                             " values: ", values_tensor.shape().DebugString());
    }
    return OkStatus();
  }
};
```

The following snippet shows how the kernel accesses one of the inputs.

<!-- test_snippets_in_readme skip -->
```c++
  void Compute(OpKernelContext* ctx) override {
    const auto& cond_indices_tensor = ctx->input(0);
    const auto& cond_values_tensor = ctx->input(1);
    const auto& cond_shape_tensor = ctx->input(2);
    // Error checking omitted, see source file.
    const int cond_elements = cond_indices_tensor.dim_size(0);
    const auto cond_indices = cond_indices_tensor.matrix<int64_t>();
    const auto cond_values = cond_values_tensor.flat<bool>();
    const auto cond_shape = cond_shape_tensor.flat<int64_t>();
  }
```

The kernel implements the multiplex operation through the following steps:

1.  **Create an empty list**: The list contains elements that refer to values of
    `a` or `b` in merged order. The elements are pairs of a `bool` to indicate
    the source tensor (a = true, b = false) and an `int` to indicate the source
    index.

2.  **Append values in `a` and `b` to the list**: Looping through all
    non-missing values in `a` and `b`, determine whether each position has a
    non-missing value in `a`, `b`, or both.

    If `cond` is true and `a` has a non-missing value at that position, append
    this element to the list. If `cond` is false or missing and `b` has a
    non-missing value at that position, append this element to the list.

    Note: Assume that indices are sorted in canonical row-major order (e.g.
    using
    [`tf.sparse.reorder`](https://www.tensorflow.org/api_docs/python/tf/sparse/reorder)).

3.  **Allocate the output**: The size of the output is based on the length of
    the list.

4.  **Add indices and values of `a` and `b` to the output**: Iterate through the
    elements in the list and copy the indices and values of `a` and `b` to the
    output.

5.  **Set the shape of the output**: Set the (dense) shape of the output to
    match the largest of the inputs. Shaping the output is accomplished with
    following snippet:

    <!-- test_snippets_in_readme skip -->
    ```c++
    for (int i = 0; i < rank; ++i) {
      output_shape(i) =
          std::max(cond_shape(i), std::max(a_shape(i), b_shape(i)));
    }
    ```

##### Considerations when setting the output shape

This implementation is specific to sparse tensors. The expansion is simple
because the concept of "missing values" for sparse tensors is well-defined.
There is no exact equivalent for dense tensors.

In many cases, sparse tensors are just sparse encodings of a dense tensor. In
these cases, all inputs should have the same dense shape, and the output shape
would be identical to the shape of the inputs.

In these cases, you can replace the `std::max calculation` with
`output_shape(i) = a_shape(i)` after verifying the following conditions in
`OP_REQUIRES`:

<!-- test_snippets_in_readme skip -->
```c++
cond_shape(i) == a_shape(i)
a_shape(i) == b_shape(i)
```

### Step 4 - Create the Python wrapper

To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.

When the inputs are
[`tf.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor),
the
[dispatch decorator](https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch)
in the snippet below prompts the previous
`gen_multiplex_2_op.examples_multiplex_dense` op to use the new C++ kernel
wrapped by `gen_multiplex_3_op.examples_multiplex_sparse`.

Note: This optional Python wrapper depends on the `multiplex_2` op in addition
to the `multiplex_3` dependencies. See `deps` for `multiplex_3_op` in the BUILD
file.

```
@tf.experimental.dispatch_for_api(gen_multiplex_2_op.examples_multiplex_dense)
def multiplex_sparse(cond: tf.SparseTensor,
                     a: tf.SparseTensor,
                     b: tf.SparseTensor,
                     name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.


  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of rank 1 sparse tensors, no optional parameters, no broadcasting,
  etc..

  >>> cond = tf.SparseTensor(
  ...     indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
  >>> a = tf.sparse.from_dense(['', 'a0', '', 'a1', '', 'a2', ''])
  >>> b = tf.sparse.from_dense(['b0', '', 'b1', 'b2', '', '', 'b3'])
  >>> multiplex_3_op.multiplex_sparse(cond, a, b)
  SparseTensorValue(indices=array([[0],
    [1],
    [2],
    [3]]), values=array([b'b0', b'a0', b'b1', b'b2'], dtype=object),
    dense_shape=array([7]))
  Args:
    cond: tf.SparseTensor of type bool. Where True, yield `a`, otherwise yield
      `b`.
    a: tf.SparseTensor with the same type and shape as `b`.
    b: tf.SparseTensor with the same type and shape as `a`.
    name: An optional name for the op.

  Returns:
    A tf.SparseTensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  (indices, values, shape) = examples_multiplex_sparse(
      cond_indices=cond.indices,
      cond_values=cond.values,
      cond_shape=cond.dense_shape,
      a_indices=a.indices,
      a_values=a.values,
      a_shape=a.dense_shape,
      b_indices=b.indices,
      b_values=b.values,
      b_shape=b.dense_shape,
      name=name)
  return tf.SparseTensor(indices, values, shape)
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

The following tests use the `multiplex_2_op.multiplex` custom op from the
previous `multiplex_2` example, which now supports sparse tensors (while
continuing to support dense tensors). The `test_sparse_op_different` test inputs
are sparse tensors, so it uses the new `multiplex_3_kernel` C++ kernel.

```
  @test_util.run_in_graph_and_eager_modes
  def test_sparse_op_different(self):
    cond = tf.SparseTensor(
        indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
    a = tf.SparseTensor(
        indices=[[1], [3], [5]], values=['a0', 'a1', 'a2'], dense_shape=[6])
    b = tf.SparseTensor(
        indices=[[0], [2], [3], [6]],
        values=['b0', 'b1', 'b2', 'b3'],
        dense_shape=[7])
    result = self.evaluate(multiplex_2_op.multiplex(cond, a, b))
    self.assertAllEqual([7], result.dense_shape)
    self.assertAllEqual([[0], [1], [2], [3]], result.indices)
    self.assertAllEqual([b'b0', b'a0', b'b1', b'b2'], result.values)
```

The `test_multiplex_int` test inputs are dense tensors, so it uses the old
`multiplex_2_kernel` C++ kernel.

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

Refer to `multiplex_3_test.py` for the full source code which contains all the
test cases.

Reuse the `BUILD` file to add build rules for the Python API wrapper and the op
test.

```
tf_custom_op_library(
    name = "multiplex_3_kernel.so",
    srcs = [
        "multiplex_3_kernel.cc",
        "multiplex_3_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_3_op",
    srcs = ["multiplex_3_op.py"],
    data = [":multiplex_3_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
    ],
)
```

Test the op with the following:

<!-- test_snippets_in_readme skip -->
```shell
bazel test //third_party/tensorflow/google/g3doc/example/multiplex_3:multiplex_3_test
```

### Use the op

Import the op and call it using the following example:

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_3 import multiplex_3_op

cond = tf.SparseTensor(indices=[[1], [3], [6]],
                    values=[True, False, True], dense_shape=[7])
a = tf.SparseTensor(indices=[[1], [3], [5]],
                    values=['a0', 'a1', 'a2'], dense_shape=[6])
b = tf.SparseTensor(indices=[[0], [2], [3], [6]],
                    values=['b0', 'b1', 'b2', 'b3'], dense_shape=[7])
result = multiplex_2_op.multiplex(cond, a, b)
```

Here, `multiplex_2_op` is the name of the Python wrapper that was created in the
multiplex_2 example. Importing the `multiplex_3_op` Python wrapper created in
this example extends `multiplex_2_op.multiplex` to handle sparse tensors.

Build the op with the following:

<!-- test_snippets_in_readme skip -->
```shell
bazel build //third_party/tensorflow/examples/custom_ops_doc/multiplex_3:multiplex_3_op
```

### Summary

In this example, you learned how implement a new multiplexer kernel to handle
special cases. With a Python wrapper that uses
[dispatch decorators](https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch)
to override the default kernel, this custom op uses a new kernel to handle
sparse tensors.

The table below summarizes the build rules and targets for building and testing
the `multiplex_3` op.

Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | `tf_kernel_library`    | `multiplex_3_kernel` | `multiplex_3_kernel.cc`, `multiplex_3_op.cc`
Wrapper (automatically generated)       | `tf_gen_op_wrapper.py` | `gen_multiplex_3_op` | N/A
Wrapper (with public API and docstring) | `py_strict_library`    | `multiplex_3_op`     | `multiplex_3_op.py`
Tests                                   | `tf_py_test`           | `multiplex_3_test`   | `multiplex_3_test.py`

## Resources

*   [OSS custom ops guide](https://www.tensorflow.org/guide/create_op)
*   [Extension types and dispatch decorators](https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch)
*   [Working with sparse tensors](https://www.tensorflow.org/guide/sparse_tensor)
