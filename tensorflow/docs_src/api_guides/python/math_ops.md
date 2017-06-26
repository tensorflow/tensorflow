# Math

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

Note: Elementwise binary operations in TensorFlow follow [numpy-style
broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Arithmetic Operators

TensorFlow provides several operations that you can use to add basic arithmetic
operators to your graph.

*   @{tf.add}
*   @{tf.subtract}
*   @{tf.multiply}
*   @{tf.scalar_mul}
*   @{tf.div}
*   @{tf.divide}
*   @{tf.truediv}
*   @{tf.floordiv}
*   @{tf.realdiv}
*   @{tf.truncatediv}
*   @{tf.floor_div}
*   @{tf.truncatemod}
*   @{tf.floormod}
*   @{tf.mod}
*   @{tf.cross}

## Basic Math Functions

TensorFlow provides several operations that you can use to add basic
mathematical functions to your graph.

*   @{tf.add_n}
*   @{tf.abs}
*   @{tf.negative}
*   @{tf.sign}
*   @{tf.reciprocal}
*   @{tf.square}
*   @{tf.round}
*   @{tf.sqrt}
*   @{tf.rsqrt}
*   @{tf.pow}
*   @{tf.exp}
*   @{tf.expm1}
*   @{tf.log}
*   @{tf.log1p}
*   @{tf.ceil}
*   @{tf.floor}
*   @{tf.maximum}
*   @{tf.minimum}
*   @{tf.cos}
*   @{tf.sin}
*   @{tf.lbeta}
*   @{tf.tan}
*   @{tf.acos}
*   @{tf.asin}
*   @{tf.atan}
*   @{tf.lgamma}
*   @{tf.digamma}
*   @{tf.erf}
*   @{tf.erfc}
*   @{tf.squared_difference}
*   @{tf.igamma}
*   @{tf.igammac}
*   @{tf.zeta}
*   @{tf.polygamma}
*   @{tf.betainc}
*   @{tf.rint}

## Matrix Math Functions

TensorFlow provides several operations that you can use to add linear algebra
functions on matrices to your graph.

*   @{tf.diag}
*   @{tf.diag_part}
*   @{tf.trace}
*   @{tf.transpose}
*   @{tf.eye}
*   @{tf.matrix_diag}
*   @{tf.matrix_diag_part}
*   @{tf.matrix_band_part}
*   @{tf.matrix_set_diag}
*   @{tf.matrix_transpose}
*   @{tf.matmul}
*   @{tf.norm}
*   @{tf.matrix_determinant}
*   @{tf.matrix_inverse}
*   @{tf.cholesky}
*   @{tf.cholesky_solve}
*   @{tf.matrix_solve}
*   @{tf.matrix_triangular_solve}
*   @{tf.matrix_solve_ls}
*   @{tf.qr}
*   @{tf.self_adjoint_eig}
*   @{tf.self_adjoint_eigvals}
*   @{tf.svd}


## Tensor Math Function

TensorFlow provides operations that you can use to add tensor functions to your
graph.

*   @{tf.tensordot}


## Complex Number Functions

TensorFlow provides several operations that you can use to add complex number
functions to your graph.

*   @{tf.complex}
*   @{tf.conj}
*   @{tf.imag}
*   @{tf.real}


## Reduction

TensorFlow provides several operations that you can use to perform
common math computations that reduce various dimensions of a tensor.

*   @{tf.reduce_sum}
*   @{tf.reduce_prod}
*   @{tf.reduce_min}
*   @{tf.reduce_max}
*   @{tf.reduce_mean}
*   @{tf.reduce_all}
*   @{tf.reduce_any}
*   @{tf.reduce_logsumexp}
*   @{tf.count_nonzero}
*   @{tf.accumulate_n}
*   @{tf.einsum}

## Scan

TensorFlow provides several operations that you can use to perform scans
(running totals) across one axis of a tensor.

*   @{tf.cumsum}
*   @{tf.cumprod}

## Segmentation

TensorFlow provides several operations that you can use to perform common
math computations on tensor segments.
Here a segmentation is a partitioning of a tensor along
the first dimension, i.e. it  defines a mapping from the first dimension onto
`segment_ids`. The `segment_ids` tensor should be the size of
the first dimension, `d0`, with consecutive IDs in the range `0` to `k`,
where `k<d0`.
In particular, a segmentation of a matrix tensor is a mapping of rows to
segments.

For example:

```python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
  ==>  [[0 0 0 0]
        [5 6 7 8]]
```

*   @{tf.segment_sum}
*   @{tf.segment_prod}
*   @{tf.segment_min}
*   @{tf.segment_max}
*   @{tf.segment_mean}
*   @{tf.unsorted_segment_sum}
*   @{tf.sparse_segment_sum}
*   @{tf.sparse_segment_mean}
*   @{tf.sparse_segment_sqrt_n}


## Sequence Comparison and Indexing

TensorFlow provides several operations that you can use to add sequence
comparison and index extraction to your graph. You can use these operations to
determine sequence differences and determine the indexes of specific values in
a tensor.

*   @{tf.argmin}
*   @{tf.argmax}
*   @{tf.setdiff1d}
*   @{tf.where}
*   @{tf.unique}
*   @{tf.edit_distance}
*   @{tf.invert_permutation}
