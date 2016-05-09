/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/mirror_pad_mode.h"

namespace tensorflow {

REGISTER_OP("Pack")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .Doc(R"doc(
Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
`output[i, ...] = values[i][...]`.

This is the opposite of `unpack`.

values: Must be of same shape and type.
output: The packed tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Unpack")
    .Input("value: T")
    .Output("output: num * T")
    .Attr("num: int >= 0")
    .Attr("T: type")
    .Doc(R"doc(
Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the first dimension.
The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
`output` has shape `value.shape[1:]`.

This is the opposite of `pack`.

value: 1-D or higher, with first dimension `num`.
output: The list of tensors unpacked from `value`.
)doc");

// --------------------------------------------------------------------------
// TODO(josh11b): Remove the >= 2 constraint, once we can rewrite the graph
// in the N == 1 case to remove the node.
REGISTER_OP("Concat")
    .Input("concat_dim: int32")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Doc(R"doc(
Concatenates tensors along one dimension.

concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
  range [0, rank(values)).
values: The `N` Tensors to concatenate. Their ranks and types must match,
  and their sizes must match in all dimensions except `concat_dim`.
output: A `Tensor` with the concatenation of values stacked along the
  `concat_dim` dimension.  This tensor's shape matches that of `values` except
  in `concat_dim` where it has the sum of the sizes.
)doc");

REGISTER_OP("ConcatOffset")
    .Input("concat_dim: int32")
    .Input("shape: N * int32")
    .Output("offset: N * int32")
    .Attr("N: int >= 2")
    .Doc(R"doc(
Computes offsets of concat inputs within its output.

For example:

```prettyprint
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```

concat_dim: The dimension along which to concatenate.
shape: The `N` int32 vectors representing shape of tensors being concatenated.
output: The `N` int32 vectors representing the starting offset
        of input tensors within the concatenated output.

This is typically used by gradient computations for a concat operation.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Split")
    .Input("split_dim: int32")
    .Input("value: T")
    .Output("output: num_split * T")
    .Attr("num_split: int >= 1")
    .Attr("T: type")
    .Doc(R"doc(
Splits a tensor into `num_split` tensors along one dimension.

split_dim: 0-D.  The dimension along which to split.  Must be in the range
  `[0, rank(value))`.
num_split: The number of ways to split.  Must evenly divide
  `value.shape[split_dim]`.
value: The tensor to split.
output: They are identically shaped tensors, whose shape matches that of `value`
  except along `split_dim`, where their sizes are
  `values.shape[split_dim] / num_split`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Const")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type")
    .Doc(R"doc(
Returns a constant tensor.

value: Attr `value` is the tensor to return.
)doc");

// --------------------------------------------------------------------------
// TODO(mgubin): Update the doc when the freeze_graph script supports converting
// into memmapped format.
REGISTER_OP("ImmutableConst")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("memory_region_name: string")
    .Output("tensor: dtype")
    .Doc(R"doc(
Returns immutable tensor from memory region.

The current implementation memmaps the tensor from a file.

dtype: Type of the returned tensor.
shape: Shape of the returned tensor.
memory_region_name: Name of readonly memory region used by the tensor, see
  NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ZerosLike")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: type")
    .Doc(R"doc(
Returns a tensor of zeros with the same shape and type as x.

x: a tensor of type T.
y: a tensor of the same shape and type as x but filled with zeros.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Diag")
    .Input("diagonal: T")
    .Output("output: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns a diagonal tensor with a given diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

For example:

```prettyprint
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```

diagonal: Rank k tensor where k is at most 3.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DiagPart")
    .Input("input: T")
    .Output("diagonal: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns the diagonal part of the tensor.

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

```prettyprint
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
```

input: Rank k tensor where k is 2, 4, or 6.
diagonal: The extracted diagonal.

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixDiag")
    .Input("diagonal: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Returns a batched diagonal tensor with a given batched diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

For example:

```prettyprint
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

and diagonal.shape = (2, 4)

tf.batch_matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]

which has shape (2, 4, 4)
```

diagonal: Rank `k`, where `k >= 1`.
output: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixDiagPart")
    .Input("input: T")
    .Output("diagonal: T")
    .Attr("T: type")
    .Doc(R"doc(
Returns the batched diagonal part of a batched tensor.

This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:

Assume `input` has `k` dimensions `[I, J, K, ..., N, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., N]` where:

`diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

The input must be at least a matrix.

For example:

```prettyprint
# 'input' is [[[1, 0, 0, 0]
               [0, 2, 0, 0]
               [0, 0, 3, 0]
               [0, 0, 0, 4]],
              [[5, 0, 0, 0]
               [0, 6, 0, 0]
               [0, 0, 7, 0]
               [0, 0, 0, 8]]]

and input.shape = (2, 4, 4)

tf.batch_matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

which has shape (2, 4)
```

input: Rank `k` tensor where `k >= 2` and the last two dimensions are equal.
diagonal: The extracted diagonal(s) having shape
  `diagonal.shape = input.shape[:-1]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixBandPart")
    .Input("input: T")
    .Input("num_lower: int64")
    .Input("num_upper: int64")
    .Output("band: T")
    .Attr("T: type")
    .Doc(R"doc(
Copy a tensor setting everything outside a central band in each innermost matrix
to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function 'in_band(m, n)` is one if
`(num_lower < 0 || (m-n) <= num_lower)) &&
(num_upper < 0 || (n-m) <= num_upper)`, and zero otherwise.

For example:

```prettyprint
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.batch_matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                             [-1,  0,  1, 2]
                                             [ 0, -1,  0, 1]
                                             [ 0,  0, -1, 0]],

tf.batch_matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                            [-1,  0,  1, 0]
                                            [-2, -1,  0, 1]
                                            [ 0, -2, -1, 0]]
```

Useful special cases:

```prettyprint
 tf.batch_matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.batch_matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.batch_matrix_band_part(input, 0, 0) ==> Diagonal.
```

input: Rank `k` tensor.
num_lower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
           lower triangle.
num_upper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
           entire upper triangle.
band: Rank `k` tensor of the same shape as input. The extracted banded tensor.

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Reverse")
    .Input("tensor: T")
    .Input("dims: bool")
    .Output("output: T")
    .Attr("T: {uint8, int8, int32, bool, float, double}")
    .Doc(R"Doc(
Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

tensor: Up to 8-D.
dims: 1-D. The dimensions to reverse.
output: The same shape as `tensor`.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("EditDistance")
    .Input("hypothesis_indices: int64")
    .Input("hypothesis_values: T")
    .Input("hypothesis_shape: int64")
    .Input("truth_indices: int64")
    .Input("truth_values: T")
    .Input("truth_shape: int64")
    .Attr("normalize: bool = True")
    .Attr("T: type")
    .Output("output: float")
    .Doc(R"doc(
Computes the (possibly normalized) Levenshtein Edit Distance.

The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).

The inputs are:

hypothesis_indices: The indices of the hypothesis list SparseTensor.
  This is an N x R int64 matrix.
hypothesis_values: The values of the hypothesis list SparseTensor.
  This is an N-length vector.
hypothesis_shape: The shape of the hypothesis list SparseTensor.
  This is an R-length vector.
truth_indices: The indices of the truth list SparseTensor.
  This is an M x R int64 matrix.
truth_values: The values of the truth list SparseTensor.
  This is an M-length vector.
truth_shape: The shape of the truth list SparseTensor.
  This is an R-length vector.
truth_shape: truth indices, vector.
normalize: boolean (if true, edit distances are normalized by length of truth).

The output is:

output: A dense float tensor with rank R - 1.

For the example input:

    // hypothesis represents a 2x1 matrix with variable-length values:
    //   (0,0) = ["a"]
    //   (1,0) = ["b"]
    hypothesis_indices = [[0, 0, 0],
                          [1, 0, 0]]
    hypothesis_values = ["a", "b"]
    hypothesis_shape = [2, 1, 1]

    // truth represents a 2x2 matrix with variable-length values:
    //   (0,0) = []
    //   (0,1) = ["a"]
    //   (1,0) = ["b", "c"]
    //   (1,1) = ["a"]
    truth_indices = [[0, 1, 0],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0]]
    truth_values = ["a", "b", "c", "a"]
    truth_shape = [2, 2, 2]
    normalize = true

The output will be:

    // output is a 2x2 matrix with edit distances normalized by truth lengths.
    output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
              [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Fill")
    .Input("dims: int32")
    .Input("value: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

dims: 1-D. Represents the shape of the output tensor.
value: 0-D (scalar). Value to fill the returned tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Gather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Gather slices from `params` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/Gather.png" alt>
</div>
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("GatherNd")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Gather values from `params` according to `indices`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_N, R]` where `R` is the rank of `params`.
The innermost dimension of `indices` (with length `R`) corresponds to the
indices of `params`.

Produces an output tensor with shape `[d_0, ..., d_{n-1}]` where:

    output[i, j, k, ...] = params[indices[i, j, k, ..., :]]

e.g. for `indices` a matrix:

    output[i] = params[indices[i, :]]

params: R-D.  The tensor from which to gather values.
indices: (N+1)-D.  Index tensor having shape `[d_0, ..., d_N, R]`.
output: N-D.  Values from `params` gathered from indices given by `indices`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Identity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"Doc(
Return a tensor with the same shape and contents as the input tensor or value.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("RefIdentity")
    .Input("input: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Doc(R"Doc(
Return the same ref tensor as the input ref tensor.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("StopGradient")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"Doc(
Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:

*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("CheckNumerics")
    .Input("tensor: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("message: string")
    .Doc(R"doc(
Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

message: Prefix of the error message.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Reshape")
    .Input("tensor: T")
    .Input("shape: int32")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"Doc(
Reshapes a tensor.

Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of `shape` is the special value -1, the size of that dimension
is computed so that the total size remains constant.  In particular, a `shape`
of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

If `shape` is 1-D or higher, then the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

For example:

```prettyprint
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

shape: Defines the shape of the output tensor.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("InvertPermutation")
    .Input("x: int32")
    .Output("y: int32")
    .Doc(R"doc(
Computes the inverse permutation of a tensor.

This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:

`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

The values must include 0. There can be no duplicate values or negative values.

For example:

```prettyprint
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
```

x: 1-D.
y: 1-D.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Transpose")
    .Input("x: T")
    .Input("perm: int32")
    .Output("y: T")
    .Attr("T: type")
    .Doc(R"doc(
Shuffle dimensions of x according to a permutation.

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Unique")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: int32")
    .Attr("T: type")
    .Doc(R"doc(
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

x: 1-D.
y: 1-D.
idx: 1-D.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("UniqueWithCounts")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: int32")
    .Output("count: int32")
    .Attr("T: type")
    .Doc(R"doc(
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. Finally, it returns a third tensor `count` that
contains the count of each element of `y` in `x`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

x: 1-D.
y: 1-D.
idx: 1-D.
count: 1-D.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Shape")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .Doc(R"doc(
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

)doc");

REGISTER_OP("ShapeN")
    .Input("input: N * T")
    .Output("output: N * int32")
    .Attr("N: int32")
    .Attr("T: type")
    .Doc(R"doc(
Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ReverseSequence")
    .Input("input: T")
    .Input("seq_lengths: int64")
    .Output("output: T")
    .Attr("seq_dim: int")
    .Attr("batch_dim: int = 0")
    .Attr("T: type")
    .Doc(R"doc(
Reverses variable length slices.

This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.

For example:

```prettyprint
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```

In contrast, if:

```prettyprint
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
```

input: The input to reverse.
seq_lengths: 1-D with length `input.dims(batch_dim)` and
  `max(seq_lengths) < input.dims(seq_dim)`
seq_dim: The dimension which is partially reversed.
batch_dim: The dimension along which reversal is performed.
output: The partially reversed input. It has the same shape as `input`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Rank")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .Doc(R"doc(
Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as "order", "degree", or "ndims."
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Size")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .Doc(R"doc(
Returns the size of a tensor.

This operation returns an integer representing the number of elements in
`input`.

For example:

```prettyprint
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Slice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("size: Index")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Index: {int32,int64}")
    .Doc(R"doc(
Return a slice from 'input'.

The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

begin: begin[i] specifies the offset into the 'i'th dimension of
  'input' to slice from.
size: size[i] specifies the number of elements of the 'i'th dimension
  of 'input' to slice. If size[i] is -1, all remaining elements in dimension
  i are included in the slice (i.e. this is equivalent to setting
  size[i] = input.dim_size(i) - begin[i]).
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Tile")
    .Input("input: T")
    .Input("multiples: int32")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

input: 1-D or higher.
multiples: 1-D. Length must be the same as the number of dimensions in `input`
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("TileGrad")
    .Input("input: T")
    .Input("multiples: int32")
    .Output("output: T")
    .Attr("T: type")
    .Deprecated(3, "TileGrad has been replaced with reduce_sum")
    .Doc(R"doc(
Returns the gradient of `Tile`.

Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Where")
    .Input("input: bool")
    .Output("index: int64")
    .Doc(R"doc(
Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`input`. Indices are output in row-major order.

For example:

```prettyprint
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BroadcastGradientArgs")
    .Input("s0: int32")
    .Input("s1: int32")
    .Output("r0: int32")
    .Output("r1: int32")
    .Doc(R"doc(
Return the reduction indices for computing gradients of s0 op s1 with broadcast.

This is typically used by gradient computations for a broadcasting operation.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Pad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Pads a tensor with zeros.

This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("MirrorPad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr(GetMirrorPadModeAttrString())
    .Doc(R"doc(
Pads a tensor with mirrored values.

This operation pads a `input` with mirrored values according to the `paddings`
you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many values to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of `input`
in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
(if false, respectively).

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
```

input: The input tensor to be padded.
paddings: A two-column matrix specifying the padding sizes. The number of
  rows must be the same as the rank of `input`.
mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
  do not include the borders, while in symmetric mode the padded regions
  do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
  is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
  it is `[1, 2, 3, 3, 2]` in symmetric mode.
output: The padded tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("MirrorPadGrad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr(GetMirrorPadModeAttrString())
    .Doc(R"doc(
Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.

The folded size of each dimension D of the output is:

`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
```

input: The input tensor to be folded.
paddings: A two-column matrix specifying the padding sizes. The number of
  rows must be the same as the rank of `input`.
mode: The mode used in the `MirrorPad` op.
output: The folded tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Placeholder")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape = {}")
    .Doc(R"doc(
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

output: A placeholder tensor that must be replaced using the feed mechanism.
dtype: The type of elements in the tensor.
shape: (Optional) The shape of the tensor. If the shape has 0 dimensions, the
  shape is unconstrained.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("PlaceholderWithDefault")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Doc(R"doc(
A placeholder op that passes though `input` when its output is not fed.

input: The default value to produce when `output` is not fed.
output: A placeholder tensor that defaults to `input` if it is not fed.
dtype: The type of elements in the tensor.
shape: The (possibly partial) shape of the tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ExpandDims")
    .Input("input: T")
    .Input("dim: int32")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Inserts a dimension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
zero; if you specify a negative number for `dim` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```prettyprint
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.dims() <= dim <= input.dims()`

This operation is related to `squeeze()`, which removes dimensions of
size 1.

dim: 0-D (scalar). Specifies the dimension index at which to
  expand the shape of `input`.
output: Contains the same data as `input`, but its shape has an additional
  dimension of size 1 added.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Squeeze")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("squeeze_dims: list(int) >= 0 = []")
    .Doc(R"doc(
Removes dimensions of size 1 from the shape of a tensor.

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.

For example:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```

input: The `input` to squeeze.
squeeze_dims: If specified, only squeezes the dimensions listed. The dimension
  index starts at 0. It is an error to squeeze a dimension that is not 1.
output: Contains the same data as `input`, but has one or more dimensions of
  size 1 removed.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ListDiff")
    .Input("x: T")
    .Input("y: T")
    .Output("out: T")
    .Output("idx: int32")
    .Attr("T: type")
    .Doc(R"doc(
Computes the difference between two lists of numbers or strings.

Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

```prettyprint
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

```prettyprint
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```

x: 1-D. Values to keep.
y: 1-D. Values to remove.
out: 1-D. Values present in `x` but not in `y`.
idx: 1-D. Positions of `x` values preserved in `out`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("SpaceToBatch")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int32 > 1")
    .Doc(R"doc(
SpaceToBatch for 4-D tensors of type T.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

input: 4-D with shape `[batch, height, width, depth]`.

paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
  the padding of the input with zeros across the spatial dimensions as follows:

      paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

  The effective spatial dimensions of the zero-padded input tensor will be:

      height_pad = pad_top + height + pad_bottom
      width_pad = pad_left + width + pad_right

The attr `block_size` must be greater than one. It indicates the block size.

  * Non-overlapping blocks of size `block_size x block size` in the height and
    width dimensions are rearranged into the batch dimension at each location.
  * The batch of the output tensor is `batch * block_size * block_size`.
  * Both height_pad and width_pad must be divisible by block_size.

The shape of the output will be:

    [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
     depth]

Examples:

(1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:

```prettyprint
x = [[[[1], [2]], [[3], [4]]]]
```

The output tensor has shape `[4, 1, 1, 1]` and value:

```prettyprint
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

(2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:

```prettyprint
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

The output tensor has shape `[4, 1, 1, 3]` and value:

```prettyprint
[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
```

(3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:

```prettyprint
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]],
      [[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```

The output tensor has shape `[4, 2, 2, 1]` and value:

```prettyprint
x = [[[[1], [3]], [[5], [7]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```

(4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:

```prettyprint
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]]],
     [[[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```

The output tensor has shape `[8, 1, 2, 1]` and value:

```prettyprint
x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
     [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
```

Among others, this operation is useful for reducing atrous convolution into
regular convolution.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchToSpace")
    .Input("input: T")
    .Input("crops: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int32 > 1")
    .Doc(R"doc(
BatchToSpace for 4-D tensors of type T.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

input: 4-D tensor with shape
 `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
   depth]`. Note that the batch size of the input tensor must be divisible by
 `block_size * block_size`.

crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
  how many elements to crop from the intermediate result across the spatial
  dimensions as follows:

      crops = [[crop_top, crop_bottom], [crop_left, crop_right]]

output: 4-D with shape `[batch, height, width, depth]`, where:

      height = height_pad - crop_top - crop_bottom
      width = width_pad - crop_left - crop_right

The attr `block_size` must be greater than one. It indicates the block size.

Examples:

(1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:

```prettyprint
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

The output tensor has shape `[1, 2, 2, 1]` and value:

```prettyprint
x = [[[[1], [2]], [[3], [4]]]]
```

(2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:

```prettyprint
[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
```

The output tensor has shape `[1, 2, 2, 3]` and value:

```prettyprint
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

(3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:

```prettyprint
x = [[[[1], [3]], [[5], [7]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```

The output tensor has shape `[1, 4, 4, 1]` and value:

```prettyprint
x = [[[1],   [2],  [3],  [4]],
     [[5],   [6],  [7],  [8]],
     [[9],  [10], [11],  [12]],
     [[13], [14], [15],  [16]]]
```

(4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:

```prettyprint
x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
     [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
```

The output tensor has shape `[2, 2, 4, 1]` and value:

```prettyprint
x = [[[[1], [3]], [[5], [7]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("SpaceToDepth")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int32 >= 1")
    .Doc(R"doc(
SpaceToDepth for tensors of type T.

Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size and how the data is moved.

  * Non-overlapping blocks of size `block_size x block size` are rearranged
    into depth at each location.
  * The depth of the output tensor is `input_depth * block_size * block_size`.
  * The input tensor's height and width must be divisible by block_size.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height/block_size, width/block_size, depth*block_size*block_size]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and a divisor of both the input `height` and `width`.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:

```prettyprint
x = [[[[1], [2]],
      [[3], [4]]]]
```

This operation will output a tensor of shape `[1, 1, 1, 4]`:

```prettyprint
[[[[1, 2, 3, 4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 * block_size * block_size).
The output element shape is `[1, 1, 4]`.

For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

```prettyprint
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

This operation, for block_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`

```prettyprint
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

```prettyprint
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```

the operator will return the following tensor of shape `[1 2 2 4]`:

```prettyprint
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

block_size: The size of the spatial block.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DepthToSpace")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int32 >= 1")
    .Doc(R"doc(
DepthToSpace for tensors of type T.

Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width the output tensor is `input_depth * block_size`, whereas the
    height is `input_height * block_size`.
  * The depth of the input tensor must be divisible by
    `block_size * block_size`.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and that `block_size * block_size` be a divisor of the
input depth.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:

```prettyprint
x = [[[[1, 2, 3, 4]]]]

```

This operation will output a tensor of shape `[1, 2, 2, 1]`:

```prettyprint
   [[[[1], [2]],
     [[3], [4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.

For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

```prettyprint
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`

```prettyprint
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]

```

Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

```prettyprint
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

the operator will return the following tensor of shape `[1 4 4 1]`:

```prettyprint
x = [[ [1],   [2],  [5],  [6]],
     [ [3],   [4],  [7],  [8]],
     [ [9],  [10], [13],  [14]],
     [ [11], [12], [15],  [16]]]

```

block_size: The size of the spatial block, same as in Space2Depth.
)doc");

REGISTER_OP("Bitcast")
    .Input("input: T")
    .Output("output: type")
    .Attr("T: numbertype")
    .Attr("type: numbertype")
    .Doc(R"doc(
Bitcasts a tensor from one type to another without copying data.

Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.

If the input datatype `T` is larger than the output datatype `type` then the
shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
[..., sizeof(`type`)/sizeof(`T`)] to [...].
)doc");

REGISTER_OP("OneHot")
    .Input("indices: int64")
    .Input("depth: int32")
    .Input("on_value: T")
    .Input("off_value: T")
    .Attr("axis: int = -1")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Returns a one-hot tensor.

The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.

If the input `indices` is rank `N`, the output will have rank `N+1`,
The new axis is created at dimension `axis` (default: the new axis is
appended at the end).

If `indices` is a scalar the output shape will be a vector of length `depth`.

If `indices` is a vector of length `features`, the output shape will be:
```
  features x depth if axis == -1
  depth x features if axis == 0
```

If `indices` is a matrix (batch) with shape `[batch, features]`,
the output shape will be:
```
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```


Examples
=========

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```

Then output is `[4 x 3]`:

    ```output =
      [5.0 0.0 0.0]  // one_hot(0)
      [0.0 0.0 5.0]  // one_hot(2)
      [0.0 0.0 0.0]  // one_hot(-1)
      [0.0 5.0 0.0]  // one_hot(1)
    ```

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 0.0
  off_value = 3.0
  axis = 0
```

Then output is `[3 x 4]`:

    ```output =
      [0.0 3.0 3.0 3.0]
      [3.0 3.0 3.0 0.0]
      [3.0 3.0 3.0 3.0]
      [3.0 0.0 3.0 3.0]
    //  ^                one_hot(0)
    //      ^            one_hot(2)
    //          ^        one_hot(-1)
    //              ^    one_hot(1)
    ```
Suppose that

```
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```

Then output is `[2 x 2 x 3]`:

    ```output =
      [
        [1.0, 0.0, 0.0]  // one_hot(0)
        [0.0, 0.0, 1.0]  // one_hot(2)
      ][
        [0.0, 1.0, 0.0]  // one_hot(1)
        [0.0, 0.0, 0.0]  // one_hot(-1)
      ]```

indices: A tensor of indices.
depth: A scalar defining the depth of the one hot dimension.
on_value: A scalar defining the value to fill in output when `indices[j] = i`.
off_value: A scalar defining the value to fill in output when `indices[j] != i`.
axis: The axis to fill (default: -1, a new inner-most axis).
output: The one-hot tensor.
)doc");

}  // namespace tensorflow
