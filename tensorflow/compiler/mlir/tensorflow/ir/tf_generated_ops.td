/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for TensorFlow.
//
// This file contains TensorFlow ops whose definitions are programmatically
// generated from the TensorFlow codebase. The generated fields for an op
// includes name, summary, description, traits, arguments, results, derived
// attributes. Therefore, modifications to these fields will **not** be
// respected upon subsequent refreshes. However, additional fields after those
// fields will be retained.
//
// If you absolutely need to modify the generated fields of an op, move the
// definition to `tf_ops.td` and perform the modification there.
//
// Ops in this file are sorted alphabetically.

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

def TF_AbsOp : TF_Op<"Abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the absolute value of a tensor.";

  let description = [{
Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AddOp : TF_Op<"Add", [Broadcastable, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_NumberOrStrTensor:$x,
    TF_NumberOrStrTensor:$y
  );

  let results = (outs
    TF_NumberOrStrTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_AddNOp : TF_Op<"AddN", [Commutative, NoSideEffect]> {
  let summary = "Add all input tensors element wise.";

  let description = [{
Inputs must be of same size and shape.

  ```python
  x = [9, 7, 10]
  tf.math.add_n(x) ==> 26
  ```
  }];

  let arguments = (ins
    Variadic<TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>>:$inputs
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>:$sum
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let hasFolder = 1;
}

def TF_AddV2Op : TF_Op<"AddV2", [Broadcastable, Commutative, NoSideEffect]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_AllOp : TF_Op<"All", [NoSideEffect]> {
  let summary = [{
Computes the "logical and" of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    I1Tensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    I1Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{ return Verify(*this); }];
}

def TF_AnyOp : TF_Op<"Any", [NoSideEffect]> {
  let summary = [{
Computes the "logical or" of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    I1Tensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    I1Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{ return Verify(*this); }];
}

def TF_ArgMaxOp : TF_Op<"ArgMax", [NoSideEffect]> {
  let summary = [{
Returns the index with the largest value across dimensions of a tensor.
  }];

  let description = [{
Note that in case of ties the identity of the return value is not guaranteed.

Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmax(input = a)
  c = tf.keras.backend.eval(b)
  # c = 4
  # here a[4] = 166.32 which is the largest element of a across axis 0
  ```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$dimension
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedResultTypeAttr output_type = TF_DerivedResultTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_ArgMinOp : TF_Op<"ArgMin", [NoSideEffect]> {
  let summary = [{
Returns the index with the smallest value across dimensions of a tensor.
  }];

  let description = [{
Note that in case of ties the identity of the return value is not guaranteed.

Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmin(input = a)
  c = tf.keras.backend.eval(b)
  # c = 0
  # here a[0] = 1 which is the smallest element of a across axis 0
  ```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$dimension
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedResultTypeAttr output_type = TF_DerivedResultTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_AssertOp : TF_Op<"Assert", []> {
  let summary = "Asserts that the given condition is true.";

  let description = [{
If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.
  }];

  let arguments = (ins
    I1Tensor:$condition,
    Variadic<TF_Tensor>:$data,

    DefaultValuedAttr<I64Attr, "3">:$summarize
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<1>;

  let hasCanonicalizer = 1;
}

def TF_AssignAddVariableOp : TF_Op<"AssignAddVariableOp", []> {
  let summary = "Adds a value to the current value of a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to
see the incremented value or a subsequent newer one.
  }];

  let arguments = (ins
    TF_ResourceTensor:$resource,
    TF_Tensor:$value
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_AssignSubVariableOp : TF_Op<"AssignSubVariableOp", []> {
  let summary = "Subtracts a value from the current value of a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to
see the decremented value or a subsequent newer one.
  }];

  let arguments = (ins
    TF_ResourceTensor:$resource,
    TF_Tensor:$value
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_AssignVariableOp : TF_Op<"AssignVariableOp", []> {
  let summary = "Assigns a new value to a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.
  }];

  let arguments = (ins
    TF_ResourceTensor:$resource,
    TF_Tensor:$value
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BatchMatMulOp : TF_Op<"BatchMatMul", [NoSideEffect]> {
  let summary = "Multiplies slices of two tensors in batches.";

  let description = [{
Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y,

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_BatchMatMulV2Op : TF_Op<"BatchMatMulV2", [NoSideEffect]> {
  let summary = "Multiplies slices of two tensors in batches.";

  let description = [{
Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

*NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y,

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_BatchToSpaceNDOp : TF_Op<"BatchToSpaceND", [NoSideEffect]> {
  let summary = "BatchToSpace for N-D tensors of type T.";

  let description = [{
This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$block_shape,
    TF_I32OrI64Tensor:$crops
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tcrops = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tblock_shape = TF_DerivedOperandTypeAttr<1>;
}

def TF_BiasAddOp : TF_Op<"BiasAdd", [NoSideEffect]> {
  let summary = "Adds `bias` to `value`.";

  let description = [{
This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$value,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$bias,

    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_BiasAddGradOp : TF_Op<"BiasAddGrad", [NoSideEffect]> {
  let summary = [{
The backward operation for "BiasAdd" on the "bias" tensor.
  }];

  let description = [{
It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$out_backprop,

    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_BitcastOp : TF_Op<"Bitcast", [NoSideEffect]> {
  let summary = [{
Bitcasts a tensor from one type to another without copying data.
  }];

  let description = [{
Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.

If the input datatype `T` is larger than the output datatype `type` then the
shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
[..., sizeof(`type`)/sizeof(`T`)] to [...].

tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
(e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
gives module error.
For example,

Example 1:

>>> a = [1., 2., 3.]
>>> equality_bitcast = tf.bitcast(a, tf.complex128)
Traceback (most recent call last):
...
InvalidArgumentError: Cannot bitcast from 1 to 18 [Op:Bitcast]
>>> equality_cast = tf.cast(a, tf.complex128)
>>> print(equality_cast)
tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)

Example 2:

>>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
<tf.Tensor: shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>

Example 3:

>>> x = [1., 2., 3.]
>>> y = [0., 2., 3.]
>>> equality= tf.equal(x,y)
>>> equality_cast = tf.cast(equality,tf.float32)
>>> equality_bitcast = tf.bitcast(equality_cast,tf.uint8)
>>> print(equality)
tf.Tensor([False True True], shape=(3,), dtype=bool)
>>> print(equality_cast)
tf.Tensor([0. 1. 1.], shape=(3,), dtype=float32)
>>> print(equality_bitcast)
tf.Tensor(
    [[  0   0   0   0]
     [  0   0 128  63]
     [  0   0 128  63]], shape=(3, 4), dtype=uint8)

*NOTE*: Bitcast is implemented as a low-level cast, so machines with different
endian orderings will give different results.
  }];

  let arguments = (ins
    TF_NumberTensor:$input
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedResultTypeAttr type = TF_DerivedResultTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_BitwiseOrOp : TF_Op<"BitwiseOr", [Broadcastable, Commutative, NoSideEffect]>,
                     WithBroadcastableBinOpBuilder {
  let summary = "Elementwise computes the bitwise OR of `x` and `y`.";

  let description = [{
The result will have those bits set, that are set in `x`, `y` or both. The
computation is performed on the underlying representations of `x` and `y`.

For example:

```python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
              tf.uint8, tf.uint16, tf.uint32, tf.uint64]

for dtype in dtype_list:
  lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
  exp = tf.constant([5, 5, 7, 15], dtype=tf.float32)

  res = bitwise_ops.bitwise_or(lhs, rhs)
  tf.assert_equal(tf.cast(res,  tf.float32), exp)  # TRUE
```
  }];

  let arguments = (ins
    TF_IntTensor:$x,
    TF_IntTensor:$y
  );

  let results = (outs
    TF_IntTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BroadcastGradientArgsOp : TF_Op<"BroadcastGradientArgs", [NoSideEffect]> {
  let summary = [{
Return the reduction indices for computing gradients of s0 op s1 with broadcast.
  }];

  let description = [{
This is typically used by gradient computations for a broadcasting operation.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$s0,
    TF_I32OrI64Tensor:$s1
  );

  let results = (outs
    TF_I32OrI64Tensor:$r0,
    TF_I32OrI64Tensor:$r1
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BroadcastToOp : TF_Op<"BroadcastTo", [NoSideEffect]> {
  let summary = "Broadcast an array for a compatible shape.";

  let description = [{
Broadcasting is the process of making arrays to have compatible shapes
for arithmetic operations. Two shapes are compatible if for each
dimension pair they are either equal or one of them is one. When trying
to broadcast a Tensor to a shape, it starts with the trailing dimensions,
and works its way forward.

For example,

>>> x = tf.constant([1, 2, 3])
>>> y = tf.broadcast_to(x, [3, 3])
>>> print(y)
tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)

In the above example, the input Tensor with the shape of `[1, 3]`
is broadcasted to output Tensor with shape of `[3, 3]`.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_CastOp : TF_Op<"Cast", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Cast x of type SrcT to y of DstT.";

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$x,

    DefaultValuedAttr<BoolAttr, "false">:$Truncate
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr SrcT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr DstT = TF_DerivedResultTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_CeilOp : TF_Op<"Ceil", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns element-wise smallest integer not less than x.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$x
  );

  let results = (outs
    TF_FpTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CheckNumericsOp : TF_Op<"CheckNumerics", [SameOperandsAndResultType]> {
  let summary = "Checks a tensor for NaN and Inf values.";

  let description = [{
When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
  }];

  let arguments = (ins
    TF_FpTensor:$tensor,

    StrAttr:$message
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ComplexOp : TF_Op<"Complex", [Broadcastable, NoSideEffect]> {
  let summary = "Converts two real numbers to a complex number.";

  let description = [{
Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.

The input tensors `real` and `imag` must have the same shape.

For example:

```
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$real,
    TF_F32OrF64Tensor:$imag
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64]>:$out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_ComplexAbsOp : TF_Op<"ComplexAbs", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Computes the complex absolute value of a tensor.";

  let description = [{
Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TF_F32OrF64Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_ConcatOp : TF_Op<"Concat", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    I32Tensor:$concat_dim,
    Variadic<TF_Tensor>:$values
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<1>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_ConcatV2Op : TF_Op<"ConcatV2", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$values,
    TF_I32OrI64Tensor:$axis
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_ConjOp : TF_Op<"Conj", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns the complex conjugate of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `input`. The
complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
real part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```
  }];

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64, TF_Variant]>:$input
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64, TF_Variant]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_Conv2DOp : TF_Op<"Conv2D", [NoSideEffect]> {
  let summary = [{
Computes a 2-D convolution given 4-D `input` and `filter` tensors.
  }];

  let description = [{
Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32]>:$input,
    TensorOf<[BF16, F16, F32, F64, I32]>:$filter,

    I64ArrayAttr:$strides,
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_Conv2DBackpropFilterOp : TF_Op<"Conv2DBackpropFilter", [NoSideEffect]> {
  let summary = [{
Computes the gradients of convolution with respect to the filter.
  }];

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$input,
    I32Tensor:$filter_sizes,
    TF_FpTensor:$out_backprop,

    I64ArrayAttr:$strides,
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_Conv2DBackpropInputOp : TF_Op<"Conv2DBackpropInput", [NoSideEffect]> {
  let summary = [{
Computes the gradients of convolution with respect to the input.
  }];

  let description = [{
  }];

  let arguments = (ins
    I32Tensor:$input_sizes,
    TensorOf<[BF16, F16, F32, F64, I32]>:$filter,
    TensorOf<[BF16, F16, F32, F64, I32]>:$out_backprop,

    I64ArrayAttr:$strides,
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_Conv3DOp : TF_Op<"Conv3D", [NoSideEffect]> {
  let summary = [{
Computes a 3-D convolution given 5-D `input` and `filter` tensors.
  }];

  let description = [{
In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation.
  }];

  let arguments = (ins
    TF_FpTensor:$input,
    TF_FpTensor:$filter,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "NDHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_CosOp : TF_Op<"Cos", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes cos of x element-wise.";

  let description = [{
Given an input tensor, this function computes cosine of every
  element in the tensor. Input range is `(-inf, inf)` and
  output range is `[-1,1]`. If input lies outside the boundary, `nan`
  is returned.

  ```python
  x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
  tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
  ```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CrossReplicaSumOp : TF_Op<"CrossReplicaSum", [AllTypesMatch<["input", "output"]>, NoSideEffect]> {
  let summary = "An Op to sum inputs across replicated TPU instances.";

  let description = [{
Each instance supplies its own input.

For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
and `B, D, F, H` as group 1. Thus we get the outputs:
`[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.
  }];

  let arguments = (ins
    TensorOf<[BF16, F32, I32, TF_Uint32]>:$input,
    I32Tensor:$group_assignment
  );

  let results = (outs
    TensorOf<[BF16, F32, I32, TF_Uint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DepthToSpaceOp : TF_Op<"DepthToSpace", [NoSideEffect]> {
  let summary = "DepthToSpace for tensors of type T.";

  let description = [{
Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width the output tensor is `input_depth * block_size`, whereas the
    height is `input_height * block_size`.
  * The Y, X coordinates within each block of the output image are determined
    by the high order component of the input channel index.
  * The depth of the input tensor must be divisible by
    `block_size * block_size`.

The `data_format` attr specifies the layout of the input and output tensors
with the following options:
  "NHWC": `[ batch, height, width, channels ]`
  "NCHW": `[ batch, channels, height, width ]`
  "NCHW_VECT_C":
      `qint8 [ batch, channels / 4, height, width, 4 ]`

It is useful to consider the operation as transforming a 6-D Tensor.
e.g. for data_format = NHWC,
     Each element in the input tensor can be specified via 6 coordinates,
     ordered by decreasing memory layout significance as:
     n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
                        within the input image, bX, bY means coordinates
                        within the output block, oC means output channels).
     The output would be the input transposed to the following layout:
     n,iY,bY,iX,bX,oC

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
block_size = 2:

```
x = [[[[1, 2, 3, 4]]]]

```

This operation will output a tensor of shape `[1, 2, 2, 1]`:

```
   [[[[1], [2]],
     [[3], [4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.

For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

```
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`

```
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]

```

Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

```
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

the operator will return the following tensor of shape `[1 4 4 1]`:

```
x = [[[ [1],   [2],  [5],  [6]],
      [ [3],   [4],  [7],  [8]],
      [ [9],  [10], [13],  [14]],
      [ [11], [12], [15],  [16]]]]

```
  }];

  let arguments = (ins
    TF_Tensor:$input,

    Confined<I64Attr, [IntMinValue<2>]>:$block_size,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "NHWC">:$data_format
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DepthwiseConv2dNativeOp : TF_Op<"DepthwiseConv2dNative", [NoSideEffect]> {
  let summary = [{
Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
  }];

  let description = [{
Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

```
for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]
```

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
  }];

  let arguments = (ins
    TF_FpTensor:$input,
    TF_FpTensor:$filter,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DivOp : TF_Op<"Div", [Broadcastable, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise.";

  let description = [{
*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_DivNoNanOp : TF_Op<"DivNoNan", [Broadcastable, NoSideEffect]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if the denominator is zero.";

  let description = [{
*NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$y
  );

  let results = (outs
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DynamicStitchOp : TF_Op<"DynamicStitch", [NoSideEffect, SameVariadicOperandSize]> {
  let summary = [{
Interleave the values from the `data` tensors into a single tensor.
  }];

  let description = [{
Builds a merged tensor such that

```python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

```python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]

    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result. If you do not need this guarantee, ParallelDynamicStitch might
perform better on some devices.

For example:

```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

This method can be used to merge partitions created by `dynamic_partition`
as illustrated on the following example:

```python
    # Apply function (increments x_i) on elements for which a certain condition
    # apply (x_i != -1 in this example).
    x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
    condition_mask=tf.not_equal(x,tf.constant(-1.))
    partitioned_data = tf.dynamic_partition(
        x, tf.cast(condition_mask, tf.int32) , 2)
    partitioned_data[1] = partitioned_data[1] + 1.0
    condition_indices = tf.dynamic_partition(
        tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
    x = tf.dynamic_stitch(condition_indices, partitioned_data)
    # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
    # unchanged.
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
</div>
  }];

  let arguments = (ins
    Variadic<I32Tensor>:$indices,
    Variadic<TF_Tensor>:$data
  );

  let results = (outs
    TF_Tensor:$merged
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_EinsumOp : TF_Op<"Einsum", [NoSideEffect]> {
  let summary = [{
Tensor contraction according to Einstein summation convention.
  }];

  let description = [{
Implements generalized Tensor contraction and reduction. Each input Tensor must
have a corresponding input subscript appearing in the comma-separated left-hand
side of the equation. The right-hand side of the equation consists of the
output subscript. The input subscripts and the output subscript should consist
of zero or more named axis labels and at most one ellipsis (`...`).

The named axis labels may be any single character other than those having
special meaning, namely `,.->`. The behavior of this Op is undefined if it
receives an ill-formatted equation; since the validation is done at
graph-building time, we omit format validation checks at runtime.

Note: This Op is *not* intended to be called by the user; instead users should
call `tf.einsum` directly. It is a hidden Op used by `tf.einsum`.

Operations are applied to the input(s) according to the following rules:

 (a) Generalized Diagonals: For input dimensions corresponding to axis labels
     appearing more than once in the same input subscript, we take the
     generalized (`k`-dimensional) diagonal.
     For example, in the equation `iii->i` with input shape `[3, 3, 3]`, the
     generalized diagonal would consist of `3` elements at indices `(0, 0, 0)`,
     `(1, 1, 1)` and `(2, 2, 2)` to create a Tensor of shape `[3]`.

 (b) Reduction: Axes corresponding to labels appearing only in one input
     subscript but not in the output subscript are summed over prior to Tensor
     contraction.
     For example, in the equation `ab,bc->b`, the axis labels `a` and `c` are
     the reduction axis labels.

 (c) Batch Dimensions: Axes corresponding to labels appearing in each of the
     input subscripts and also in the output subscript make up the batch
     dimensions in Tensor contraction. Unnamed axis labels corresponding to
     ellipsis (`...`) also correspond to batch dimensions.
     For example, for the equation denoting batch matrix multiplication,
     `bij,bjk->bik`, the axis label `b` corresponds to a batch dimension.

 (d) Contraction: In case of binary einsum, axes corresponding to labels
     appearing in two different inputs (and not in the output) are contracted
     against each other.
     Considering the batch matrix multiplication equation again
     (`bij,bjk->bik`), the contracted axis label is `j`.

 (e) Expand Diagonal: If the output subscripts contain repeated (explicit) axis
     labels, the opposite operation of (a) is applied. For example, in the
     equation `i->iii`, and input shape `[3]`, the output of shape `[3, 3, 3]`
     are all zeros, except for the (generalized) diagonal which is populated
     with values from the input.
     Note: This operation is not supported by `np.einsum` or `tf.einsum`; it is
     provided to enable computing the symbolic gradient of `tf.einsum`.

The output subscripts must contain only labels appearing in at least one of the
input subscripts. Furthermore, all dimensions mapping to the same axis label
must be equal.

Any of the input and output subscripts may contain at most a single ellipsis
(`...`). These ellipsis are mapped against dimensions not corresponding to any
named axis label. If two inputs contain ellipsis, then they are broadcasted
according to standard NumPy broadcasting
[rules](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

The broadcasted dimensions are placed in the corresponding location of the
ellipsis in the output subscript. If the broadcasted dimensions are non-empty
and the output subscripts do not contain ellipsis, then an InvalidArgument error
is raised.

@compatibility(numpy)
Similar to [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

Comparison with `numpy.einsum`:

 * This Op only supports unary and binary forms of `numpy.einsum`.
 * This Op does not support implicit form. (i.e. equations without `->`).
 * This Op also supports repeated indices in the output subscript, which is not
   supported by `numpy.einsum`.
@end_compatibility
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    StrAttr:$equation
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_EluOp : TF_Op<"Elu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
  }];

  let description = [{
See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
  }];

  let arguments = (ins
    TF_FpTensor:$features
  );

  let results = (outs
    TF_FpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_EqualOp : TF_Op<"Equal", [Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of (x == y) element-wise.";

  let description = [{
*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

```python
x = tf.constant([2, 4])
y = tf.constant(2)
tf.math.equal(x, y) ==> array([True, False])

x = tf.constant([2, 4])
y = tf.constant([2, 4])
tf.math.equal(x, y) ==> array([True,  True])
```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Str, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Str, TF_Uint8]>:$y,

    DefaultValuedAttr<BoolAttr, "true">:$incompatible_shape_error
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, Value* x, "
              "Value* y, BoolAttr incompatible_shape_error">
  ];

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_ExpOp : TF_Op<"Exp", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Computes exponential of x element-wise.  \\(y = e^x\\).
  }];

  let description = [{
This function computes the exponential of every element in the input tensor.
  i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
  `e` denotes Euler's number and is approximately equal to 2.718281.
  Output is positive for any real input.

  ```python
  x = tf.constant(2.0)
  tf.math.exp(x) ==> 7.389056

  x = tf.constant([2.0, 8.0])
  tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
  ```

  For complex numbers, the exponential value is calculated as follows:

  ```
  e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
  ```

  Let's consider complex number 1+1j as an example.
  e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)

  ```python
  x = tf.constant(1 + 1j)
  tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
  ```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ExpandDimsOp : TF_Op<"ExpandDims", [NoSideEffect]> {
  let summary = "Inserts a dimension of 1 into a tensor's shape.";

  let description = [{
Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
zero; if you specify a negative number for `axis` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```
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
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$dim
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tdim = TF_DerivedOperandTypeAttr<1>;
}

def TF_FakeQuantWithMinMaxArgsOp : TF_Op<"FakeQuantWithMinMaxArgs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
  }];

  let description = [{
Attributes `[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

Quantization is called fake since the output is still in floating point.
  }];

  let arguments = (ins
    F32Tensor:$inputs,

    DefaultValuedAttr<F32Attr, "-6.0f">:$min,
    DefaultValuedAttr<F32Attr, "6.0f">:$max,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    F32Tensor:$outputs
  );

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_FakeQuantWithMinMaxVarsOp : TF_Op<"FakeQuantWithMinMaxVars", [NoSideEffect]> {
  let summary = [{
Fake-quantize the 'inputs' tensor of type float via global float scalars `min`
  }];

  let description = [{
and `max` to 'outputs' tensor of same shape as `inputs`.

`[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

This operation has a gradient and thus allows for training `min` and `max`
values.
  }];

  let arguments = (ins
    F32Tensor:$inputs,
    F32Tensor:$min,
    F32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    F32Tensor:$outputs
  );

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_FakeQuantWithMinMaxVarsPerChannelOp : TF_Op<"FakeQuantWithMinMaxVarsPerChannel", [NoSideEffect]> {
  let summary = [{
Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,
  }];

  let description = [{
`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.

`[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

This operation has a gradient and thus allows for training `min` and `max`
values.
  }];

  let arguments = (ins
    F32Tensor:$inputs,
    F32Tensor:$min,
    F32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    F32Tensor:$outputs
  );

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_FillOp : TF_Op<"Fill", [NoSideEffect]> {
  let summary = "Creates a tensor filled with a scalar value.";

  let description = [{
This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

`tf.fill` differs from `tf.constant` in a few ways:

*   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
    Tensor values.
*   `tf.fill` creates an Op in the computation graph that constructs the actual
    Tensor value at runtime. This is in contrast to `tf.constant` which embeds
    the entire Tensor into the graph with a `Const` node.
*   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
    based on other runtime Tensors, unlike `tf.constant`.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$dims,
    TF_Tensor:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr index_type = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_FloorOp : TF_Op<"Floor", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns element-wise largest integer not greater than x.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$x
  );

  let results = (outs
    TF_FpTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FloorDivOp : TF_Op<"FloorDiv", [Broadcastable, NoSideEffect]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns x // y element-wise.";

  let description = [{
*NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FloorModOp : TF_Op<"FloorMod", [Broadcastable, NoSideEffect]>,
                    WithBroadcastableBinOpBuilder {
  let summary = [{
Returns element-wise remainder of division. When `x < 0` xor `y < 0` is
  }];

  let description = [{
true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$x,
    TF_FpOrI32OrI64Tensor:$y
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FusedBatchNormOp : TF_Op<"FusedBatchNorm", [NoSideEffect]> {
  let summary = "Batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    F32Tensor:$x,
    F32Tensor:$scale,
    F32Tensor:$offset,
    F32Tensor:$mean,
    F32Tensor:$variance,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    F32Tensor:$y,
    F32Tensor:$batch_mean,
    F32Tensor:$batch_variance,
    F32Tensor:$reserve_space_1,
    F32Tensor:$reserve_space_2
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_FusedBatchNormV3Op : TF_Op<"FusedBatchNormV3", [NoSideEffect]> {
  let summary = "Batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32]>:$x,
    F32Tensor:$scale,
    F32Tensor:$offset,
    F32Tensor:$mean,
    F32Tensor:$variance,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    TensorOf<[BF16, F16, F32]>:$y,
    F32Tensor:$batch_mean,
    F32Tensor:$batch_variance,
    F32Tensor:$reserve_space_1,
    F32Tensor:$reserve_space_2,
    F32Tensor:$reserve_space_3
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;
}

def TF_GatherOp : TF_Op<"Gather", [NoSideEffect]> {
  let summary = "Gather slices from `params` according to `indices`.";

  let description = [{
`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

`validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
`indices` are always validated to be within range. If assigned to GPU,
out-of-bound indices result in safe but unspecified behavior, which may include
raising an error.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
</div>
  }];

  let arguments = (ins
    TF_Tensor:$params,
    TF_I32OrI64Tensor:$indices,

    DefaultValuedAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;
}

def TF_GatherNdOp : TF_Op<"GatherNd", [NoSideEffect]> {
  let summary = [{
Gather slices from `params` into a Tensor with shape specified by `indices`.
  }];

  let description = [{
`indices` is a K-dimensional integer tensor, best thought of as a
(K-1)-dimensional tensor of indices into `params`, where each element defines a
slice of `params`:

    output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]

Whereas in `tf.gather` `indices` defines slices into the `axis`
dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
first `N` dimensions of `params`, where `N = indices.shape[-1]`.

The last dimension of `indices` can be at most the rank of
`params`:

    indices.shape[-1] <= params.rank

The last dimension of `indices` corresponds to elements
(if `indices.shape[-1] == params.rank`) or slices
(if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
of `params`.  The output tensor has shape

    indices.shape[:-1] + params.shape[indices.shape[-1]:]

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, a 0 is stored in the
corresponding output value.

Some examples below.

Simple indexing into a matrix:

```python
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']
```

Slice indexing into a matrix:

```python
    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]
```

Indexing into a 3-tensor:

```python
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]


    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]


    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']
```

Batched indexing into a matrix:

```python
    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]
```

Batched slice indexing into a matrix:

```python
    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]
```

Batched indexing into a 3-tensor:

```python
    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]

    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]


    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]
```

See also `tf.gather` and `tf.batch_gather`.
  }];

  let arguments = (ins
    TF_Tensor:$params,
    TF_I32OrI64Tensor:$indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;
}

def TF_GatherV2Op : TF_Op<"GatherV2", [NoSideEffect]> {
  let summary = [{
Gather slices from `params` axis `axis` according to `indices`.
  }];

  let description = [{
`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `params.shape[:axis] + indices.shape +
params.shape[axis + 1:]` where:

```python
    # Scalar indices (output is rank(params) - 1).
    output[a_0, ..., a_n, b_0, ..., b_n] =
      params[a_0, ..., a_n, indices, b_0, ..., b_n]

    # Vector indices (output is rank(params)).
    output[a_0, ..., a_n, i, b_0, ..., b_n] =
      params[a_0, ..., a_n, indices[i], b_0, ..., b_n]

    # Higher rank indices (output is rank(params) + rank(indices) - 1).
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
</div>

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, a 0 is stored in the
corresponding output value.

See also `tf.batch_gather` and `tf.gather_nd`.
  }];

  let arguments = (ins
    TF_Tensor:$params,
    TF_I32OrI64Tensor:$indices,
    TF_I32OrI64Tensor:$axis,

    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Taxis = TF_DerivedOperandTypeAttr<2>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_GreaterOp : TF_Op<"Greater", [Broadcastable, NoSideEffect]>,
                   WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x > y) element-wise.";

  let description = [{
*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

```python
x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
tf.math.greater(x, y) ==> [False, True, True]

x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.greater(x, y) ==> [False, False, True]
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_GreaterEqualOp : TF_Op<"GreaterEqual", [Broadcastable, NoSideEffect]>,
                        WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x >= y) element-wise.";

  let description = [{
*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

```python
x = tf.constant([5, 4, 6, 7])
y = tf.constant([5, 2, 5, 10])
tf.math.greater_equal(x, y) ==> [True, True, True, False]

x = tf.constant([5, 4, 6, 7])
y = tf.constant([5])
tf.math.greater_equal(x, y) ==> [True, False, True, True]
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_HashTableV2Op : TF_Op<"HashTableV2", []> {
  let summary = "Creates a non-initialized hash table.";

  let description = [{
This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.
  }];

  let arguments = (ins
    StrAttr:$container,
    StrAttr:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype
  );

  let results = (outs
    TF_ResourceTensor:$table_handle
  );
}

def TF_IdentityNOp : TF_Op<"IdentityN", [NoSideEffect]> {
  let summary = [{
Returns a list of tensors with the same shapes and contents as the input
  }];

  let description = [{
tensors.

This op can be used to override the gradient for complicated functions. For
example, suppose y = f(x) and we wish to apply a custom function g for backprop
such that dx = g(dy). In Python,

```python
with tf.get_default_graph().gradient_override_map(
    {'IdentityN': 'OverrideGradientWithG'}):
  y, _ = identity_n([f(x), x])

@tf.RegisterGradient('OverrideGradientWithG')
def ApplyG(op, dy, _):
  return [None, g(dy)]  # Do not backprop to f(x).
```
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$input
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

def TF_ImagOp : TF_Op<"Imag", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns the imaginary part of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
```
  }];

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64]>:$input
  );

  let results = (outs
    TF_F32OrF64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_InvertOp : TF_Op<"Invert", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Invert (flip) each bit of supported types; for example, type `uint8` value 01010101 becomes 10101010.
  }];

  let description = [{
Flip each bit of supported types.  For example, type `int8` (decimal 2) binary 00000010 becomes (decimal -3) binary 11111101.
This operation is performed on each element of the tensor argument `x`.

Example:
```python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops

# flip 2 (00000010) to -3 (11111101)
tf.assert_equal(-3, bitwise_ops.invert(2))

dtype_list = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
              dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]

inputs = [0, 5, 3, 14]
for dtype in dtype_list:
  # Because of issues with negative numbers, let's test this indirectly.
  # 1. invert(a) and a = 0
  # 2. invert(a) or a = invert(0)
  input_tensor = tf.constant([0, 5, 3, 14], dtype=dtype)
  not_a_and_a, not_a_or_a, not_0 = [bitwise_ops.bitwise_and(
                                      input_tensor, bitwise_ops.invert(input_tensor)),
                                    bitwise_ops.bitwise_or(
                                      input_tensor, bitwise_ops.invert(input_tensor)),
                                    bitwise_ops.invert(
                                      tf.constant(0, dtype=dtype))]

  expected = tf.constant([0, 0, 0, 0], dtype=tf.float32)
  tf.assert_equal(tf.cast(not_a_and_a, tf.float32), expected)

  expected = tf.cast([not_0] * 4, tf.float32)
  tf.assert_equal(tf.cast(not_a_or_a, tf.float32), expected)

  # For unsigned dtypes let's also check the result directly.
  if dtype.is_unsigned:
    inverted = bitwise_ops.invert(input_tensor)
    expected = tf.constant([dtype.max - x for x in inputs], dtype=tf.float32)
    tf.assert_equal(tf.cast(inverted, tf.float32), tf.cast(expected, tf.float32))
```
  }];

  let arguments = (ins
    TF_IntTensor:$x
  );

  let results = (outs
    TF_IntTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_IsFiniteOp : TF_Op<"IsFinite", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns which elements of x are finite.";

  let description = [{
@compatibility(numpy)
Equivalent to np.isfinite
@end_compatibility

Example:

```python
x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
tf.math.is_finite(x) ==> [True, True, True, False, False]
```
  }];

  let arguments = (ins
    TF_FpTensor:$x
  );

  let results = (outs
    I1Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_L2LossOp : TF_Op<"L2Loss", [NoSideEffect]> {
  let summary = "L2 Loss.";

  let description = [{
Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2
  }];

  let arguments = (ins
    TF_FpTensor:$t
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LRNOp : TF_Op<"LRN", [NoSideEffect]> {
  let summary = "Local Response Normalization.";

  let description = [{
The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32]>:$input,

    DefaultValuedAttr<I64Attr, "5">:$depth_radius,
    DefaultValuedAttr<F32Attr, "1.0f">:$bias,
    DefaultValuedAttr<F32Attr, "1.0f">:$alpha,
    DefaultValuedAttr<F32Attr, "0.5f">:$beta
  );

  let results = (outs
    TensorOf<[BF16, F16, F32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LeakyReluOp : TF_Op<"LeakyRelu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear: `max(features, features * alpha)`.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$features,

    DefaultValuedAttr<F32Attr, "0.2f">:$alpha
  );

  let results = (outs
    TF_FpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_LeftShiftOp : TF_Op<"LeftShift", [Broadcastable, NoSideEffect]>,
                     WithBroadcastableBinOpBuilder {
  let summary = "Elementwise computes the bitwise left-shift of `x` and `y`.";

  let description = [{
If `y` is negative, or greater than or equal to the width of `x` in bits the
result is implementation defined.

Example:

```python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
import numpy as np
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]

for dtype in dtype_list:
  lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

  left_shift_result = bitwise_ops.left_shift(lhs, rhs)

  print(left_shift_result)

# This will print:
# tf.Tensor([ -32   -5 -128    0], shape=(4,), dtype=int8)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int16)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int32)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int64)

lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
bitwise_ops.left_shift(lhs, rhs)
# <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
```
  }];

  let arguments = (ins
    TF_IntTensor:$x,
    TF_IntTensor:$y
  );

  let results = (outs
    TF_IntTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LessOp : TF_Op<"Less", [Broadcastable, NoSideEffect]>,
                WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x < y) element-wise.";

  let description = [{
*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

```python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less(x, y) ==> [False, True, False]

x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 7])
tf.math.less(x, y) ==> [False, True, True]
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LessEqualOp : TF_Op<"LessEqual", [Broadcastable, NoSideEffect]>,
                     WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x <= y) element-wise.";

  let description = [{
*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

```python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less_equal(x, y) ==> [True, True, False]

x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 6])
tf.math.less_equal(x, y) ==> [True, True, True]
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LogOp : TF_Op<"Log", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes natural logarithm of x element-wise.";

  let description = [{
I.e., \\(y = \log_e x\\).

Example:

```python
x = tf.constant([0, 0.5, 1, 5])
tf.math.log(x) ==> [-inf, -0.6931472,  0. ,  1.609438]
```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_LogSoftmaxOp : TF_Op<"LogSoftmax", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes log softmax activations.";

  let description = [{
For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  }];

  let arguments = (ins
    TF_FpTensor:$logits
  );

  let results = (outs
    TF_FpTensor:$logsoftmax
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LogicalAndOp : TF_Op<"LogicalAnd", [Broadcastable, Commutative, NoSideEffect]>,
                      WithBroadcastableBinOpBuilder {
  let summary = "Returns the truth value of x AND y element-wise.";

  let description = [{
*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    I1Tensor:$x,
    I1Tensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );
}

def TF_LogicalNotOp : TF_Op<"LogicalNot", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns the truth value of `NOT x` element-wise.";

  let description = [{
  }];

  let arguments = (ins
    I1Tensor:$x
  );

  let results = (outs
    I1Tensor:$y
  );

  let hasCanonicalizer = 1;
}

def TF_LogicalOrOp : TF_Op<"LogicalOr", [Broadcastable, Commutative, NoSideEffect]>,
                     WithBroadcastableBinOpBuilder {
  let summary = "Returns the truth value of x OR y element-wise.";

  let description = [{
*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    I1Tensor:$x,
    I1Tensor:$y
  );

  let results = (outs
    I1Tensor:$z
  );
}

def TF_LookupTableFindV2Op : TF_Op<"LookupTableFindV2", []> {
  let summary = "Looks up keys in a table, outputs the corresponding values.";

  let description = [{
The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.
  }];

  let arguments = (ins
    TF_ResourceTensor:$table_handle,
    TF_Tensor:$keys,
    TF_Tensor:$default_value
  );

  let results = (outs
    TF_Tensor:$values
  );

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<2>;
}

def TF_LookupTableImportV2Op : TF_Op<"LookupTableImportV2", []> {
  let summary = [{
Replaces the contents of the table with the specified keys and values.
  }];

  let description = [{
The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.
  }];

  let arguments = (ins
    TF_ResourceTensor:$table_handle,
    TF_Tensor:$keys,
    TF_Tensor:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<2>;
}

def TF_LookupTableSizeV2Op : TF_Op<"LookupTableSizeV2", []> {
  let summary = "Computes the number of elements in the given table.";

  let description = [{
  }];

  let arguments = (ins
    TF_ResourceTensor:$table_handle
  );

  let results = (outs
    I64Tensor:$size
  );
}

def TF_MatMulOp : TF_Op<"MatMul", [NoSideEffect]> {
  let summary = [{
Multiply the matrix "a" by the matrix "b".
  }];

  let description = [{
The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$a,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$b,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixDiagOp : TF_Op<"MatrixDiag", [NoSideEffect]> {
  let summary = [{
Returns a batched diagonal tensor with a given batched diagonal values.
  }];

  let description = [{
Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

For example:

```
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

and diagonal.shape = (2, 4)

tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]

which has shape (2, 4, 4)
```
  }];

  let arguments = (ins
    TF_Tensor:$diagonal
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixDiagV2Op : TF_Op<"MatrixDiagV2", [NoSideEffect]> {
  let summary = [{
Returns a batched diagonal tensor with given batched diagonal values.
  }];

  let description = [{
Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
diagonals of a matrix, with everything else padded with `padding`. `num_rows`
and `num_cols` specify the dimension of the innermost matrix of the output. If
both are not specified, the op assumes the innermost matrix is square and infers
its size from `k` and the innermost dimension of `diagonal`. If only one of them
is specified, the op assumes the unspecified value is the smallest possible
based on other criteria.

Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
`r` with shape `[I, J, ..., L, num_rows, num_cols]`.

The second innermost dimension of `diagonal` has double meaning.
When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
[I, J, ..., M], and the output tensor is:

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
    padding_value                             ; otherwise
```

Otherwise, `M` is treated as the number of diagonals for the matrix in the
same batch (`M = k[1]-k[0]+1`), and the output tensor is:

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    padding_value                                     ; otherwise
```
where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

For example:

```
# The main diagonal.
diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                     [5, 6, 7, 8]])
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                               [0, 2, 0, 0],
                               [0, 0, 3, 0],
                               [0, 0, 0, 4]],
                              [[5, 0, 0, 0],
                               [0, 6, 0, 0],
                               [0, 0, 7, 0],
                               [0, 0, 0, 8]]]

# A superdiagonal (per batch).
diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                     [4, 5, 6]])
tf.matrix_diag(diagonal, k = 1)
  ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0]],
       [[0, 4, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 6],
        [0, 0, 0, 0]]]

# A band of diagonals.
diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
                       [4, 5, 0]],
                      [[6, 7, 9],
                       [9, 1, 0]]])
tf.matrix_diag(diagonals, k = (-1, 0))
  ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
        [4, 2, 0],
        [0, 5, 3]],
       [[6, 0, 0],
        [9, 7, 0],
        [0, 1, 9]]]

# Rectangular matrix.
diagonal = np.array([1, 2])  # Input shape: (2)
tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
  ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
       [1, 0, 0, 0],
       [0, 2, 0, 0]]

# Rectangular matrix with inferred num_cols and padding_value = 9.
tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
  ==> [[9, 9],  # Output shape: (3, 2)
       [1, 9],
       [9, 2]]
```
  }];

  let arguments = (ins
    TF_Tensor:$diagonal,
    I32Tensor:$k,
    I32Tensor:$num_rows,
    I32Tensor:$num_cols,
    TF_Tensor:$padding_value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixDiagV3Op : TF_Op<"MatrixDiagV3", [NoSideEffect]> {
  let summary = [{
Returns a batched diagonal tensor with given batched diagonal values.
  }];

  let description = [{
Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
diagonals of a matrix, with everything else padded with `padding`. `num_rows`
and `num_cols` specify the dimension of the innermost matrix of the output. If
both are not specified, the op assumes the innermost matrix is square and infers
its size from `k` and the innermost dimension of `diagonal`. If only one of them
is specified, the op assumes the unspecified value is the smallest possible
based on other criteria.

Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
`r` with shape `[I, J, ..., L, num_rows, num_cols]`.

The second innermost dimension of `diagonal` has double meaning.
When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
[I, J, ..., M], and the output tensor is:

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
    padding_value                             ; otherwise
```

Otherwise, `M` is treated as the number of diagonals for the matrix in the
same batch (`M = k[1]-k[0]+1`), and the output tensor is:

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    padding_value                                     ; otherwise
```
where `d = n - m`, `diag_index = [k] - d`, and
`index_in_diag = n - max(d, 0) + offset`.

`offset` is zero except when the alignment of the diagonal is to the right.
```
offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                           and `d >= 0`) or
                                         (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                           and `d <= 0`)
         0                          ; otherwise
```
where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

For example:

```
# The main diagonal.
diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                     [5, 6, 7, 8]])
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                               [0, 2, 0, 0],
                               [0, 0, 3, 0],
                               [0, 0, 0, 4]],
                              [[5, 0, 0, 0],
                               [0, 6, 0, 0],
                               [0, 0, 7, 0],
                               [0, 0, 0, 8]]]

# A superdiagonal (per batch).
diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                     [4, 5, 6]])
tf.matrix_diag(diagonal, k = 1)
  ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0]],
       [[0, 4, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 6],
        [0, 0, 0, 0]]]

# A tridiagonal band (per batch).
diagonals = np.array([[[0, 8, 9],  # Input shape: (2, 2, 3)
                       [1, 2, 3],
                       [4, 5, 0]],
                      [[0, 2, 3],
                       [6, 7, 9],
                       [9, 1, 0]]])
tf.matrix_diag(diagonals, k = (-1, 1))
  ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
        [4, 2, 9],
        [0, 5, 3]],
       [[6, 2, 0],
        [9, 7, 3],
        [0, 1, 9]]]

# LEFT_RIGHT alignment.
diagonals = np.array([[[8, 9, 0],  # Input shape: (2, 2, 3)
                       [1, 2, 3],
                       [0, 4, 5]],
                      [[2, 3, 0],
                       [6, 7, 9],
                       [0, 9, 1]]])
tf.matrix_diag(diagonals, k = (-1, 1), align="LEFT_RIGHT")
  ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
        [4, 2, 9],
        [0, 5, 3]],
       [[6, 2, 0],
        [9, 7, 3],
        [0, 1, 9]]]

# Rectangular matrix.
diagonal = np.array([1, 2])  # Input shape: (2)
tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
  ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
       [1, 0, 0, 0],
       [0, 2, 0, 0]]

# Rectangular matrix with inferred num_cols and padding_value = 9.
tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
  ==> [[9, 9],  # Output shape: (3, 2)
       [1, 9],
       [9, 2]]

```
  }];

  let arguments = (ins
    TF_Tensor:$diagonal,
    I32Tensor:$k,
    I32Tensor:$num_rows,
    I32Tensor:$num_cols,
    TF_Tensor:$padding_value,

    DefaultValuedAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "RIGHT_LEFT">:$align
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixSetDiagV3Op : TF_Op<"MatrixSetDiagV3", [NoSideEffect]> {
  let summary = [{
Returns a batched matrix tensor with new batched diagonal values.
  }];

  let description = [{
Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the specified diagonals of the
innermost matrices. These will be overwritten by the values in `diagonal`.

`input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
`k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
`num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
`max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
`max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
If `k` is scalar or `k[0] == k[1]`:

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
    input[i, j, ..., l, m, n]              ; otherwise
```

Otherwise,

```
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    input[i, j, ..., l, m, n]                         ; otherwise
```
where `d = n - m`, `diag_index = k[1] - d`, and
`index_in_diag = n - max(d, 0) + offset`.

`offset` is zero except when the alignment of the diagonal is to the right.
```
offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                           and `d >= 0`) or
                                         (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                           and `d <= 0`)
         0                          ; otherwise
```
where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

For example:

```
# The main diagonal.
input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                   [7, 7, 7, 7],
                   [7, 7, 7, 7]],
                  [[7, 7, 7, 7],
                   [7, 7, 7, 7],
                   [7, 7, 7, 7]]])
diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                     [4, 5, 6]])
tf.matrix_set_diag(input, diagonal)
  ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
        [7, 2, 7, 7],
        [7, 7, 3, 7]],
       [[4, 7, 7, 7],
        [7, 5, 7, 7],
        [7, 7, 6, 7]]]

# A superdiagonal (per batch).
tf.matrix_set_diag(input, diagonal, k = 1)
  ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
        [7, 7, 2, 7],
        [7, 7, 7, 3]],
       [[7, 4, 7, 7],
        [7, 7, 5, 7],
        [7, 7, 7, 6]]]

# A band of diagonals.
diagonals = np.array([[[0, 9, 1],  # Diagonal shape: (2, 4, 3)
                       [6, 5, 8],
                       [1, 2, 3],
                       [4, 5, 0]],
                      [[0, 1, 2],
                       [5, 6, 4],
                       [6, 1, 2],
                       [3, 4, 0]]])
tf.matrix_set_diag(input, diagonals, k = (-1, 2))
  ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
        [4, 2, 5, 1],
        [7, 5, 3, 8]],
       [[6, 5, 1, 7],
        [3, 1, 6, 2],
        [7, 4, 2, 4]]]

# LEFT_RIGHT alignment.
diagonals = np.array([[[9, 1, 0],  # Diagonal shape: (2, 4, 3)
                       [6, 5, 8],
                       [1, 2, 3],
                       [0, 4, 5]],
                      [[1, 2, 0],
                       [5, 6, 4],
                       [6, 1, 2],
                       [0, 3, 4]]])
tf.matrix_set_diag(input, diagonals, k = (-1, 2), align="LEFT_RIGHT")
  ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
        [4, 2, 5, 1],
        [7, 5, 3, 8]],
       [[6, 5, 1, 7],
        [3, 1, 6, 2],
        [7, 4, 2, 4]]]

```
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_Tensor:$diagonal,
    I32Tensor:$k,

    DefaultValuedAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "RIGHT_LEFT">:$align
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxOp : TF_Op<"Max", [NoSideEffect]> {
  let summary = [{
Computes the maximum of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *input, "
    "Value *reduction_indices, BoolAttr keep_dims"
  >];
}

def TF_MaxPoolOp : TF_Op<"MaxPool", [NoSideEffect]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Qint8, TF_Uint16, TF_Uint8]>:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "NHWC">:$data_format
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Qint8, TF_Uint16, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPoolGradOp : TF_Op<"MaxPoolGrad", [NoSideEffect]> {
  let summary = "Computes gradients of the maxpooling function.";

  let description = [{
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$orig_input,
    TF_IntOrFpTensor:$orig_output,
    TF_IntOrFpTensor:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_IntOrFpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_MaximumOp : TF_Op<"Maximum", [Broadcastable, NoSideEffect]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns the max of x and y (i.e. x > y ? x : y) element-wise.";

  let description = [{
*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$x,
    TF_FpOrI32OrI64Tensor:$y
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MinOp : TF_Op<"Min", [NoSideEffect]> {
  let summary = [{
Computes the minimum of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_MinimumOp : TF_Op<"Minimum", [Broadcastable, NoSideEffect]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns the min of x and y (i.e. x < y ? x : y) element-wise.";

  let description = [{
*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$x,
    TF_FpOrI32OrI64Tensor:$y
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MirrorPadOp : TF_Op<"MirrorPad", [NoSideEffect]> {
  let summary = "Pads a tensor with mirrored values.";

  let description = [{
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

```
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$paddings,

    TF_AnyStrAttrOf<["REFLECT", "SYMMETRIC"]>:$mode
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
}

def TF_MlirPassthroughOp : TF_Op<"MlirPassthroughOp", [NoSideEffect]> {
  let summary = [{
Wraps an arbitrary MLIR computation expressed as a module with a main() function.
  }];

  let description = [{
This operation does not have an associated kernel and is not intended to be
executed in a regular TensorFlow session. Instead it is intended to be used for
testing or for special case where a user intends to pass custom MLIR computation
through a TensorFlow graph with the intent of having custom tooling processing
it downstream (when targeting a different environment, like TensorFlow lite for
example).
The MLIR module is expected to have a main() function that will be used as an
entry point. The inputs to the operations will be passed as argument to the
main() function and the returned values of the main function mapped to the
outputs.
Example usage:

```
import tensorflow as tf
from tensorflow.compiler.mlir.tensorflow.gen_mlir_passthrough_op import mlir_passthrough_op

mlir_module = '''python
func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {
   %add = "magic.op"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
   return %ret : tensor<10x10xf32>
}
'''

@tf.function
def foo(x, y):
  return = mlir_passthrough_op([x, y], mlir_module, Toutputs=[tf.float32])

graph_def = foo.get_concrete_function(tf.TensorSpec([10], tf.float32), tf.TensorSpec([10], tf.float32)).graph.as_graph_def()
```
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    StrAttr:$mlir_module
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedOperandTypeListAttr Tinputs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF_MulOp : TF_Op<"Mul", [Broadcastable, Commutative, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x * y element-wise.";

  let description = [{
*NOTE*: `Multiply` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MulNoNanOp : TF_Op<"MulNoNan", [Broadcastable, NoSideEffect]>,
                    WithBroadcastableBinOpBuilder {
  let summary = [{
Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.
  }];

  let description = [{
*NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$y
  );

  let results = (outs
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NegOp : TF_Op<"Neg", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes numerical negative value element-wise.";

  let description = [{
I.e., \\(y = -x\\).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_NoOp : TF_Op<"NoOp", [NoSideEffect]> {
  let summary = "Does nothing. Only useful as a placeholder for control edges.";

  let description = [{
  }];

  let arguments = (ins);

  let results = (outs);
}

def TF_NonMaxSuppressionV4Op : TF_Op<"NonMaxSuppressionV4", [NoSideEffect]> {
  let summary = [{
Greedily selects a subset of bounding boxes in descending order of score,
  }];

  let description = [{
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold, score_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)
  }];

  let arguments = (ins
    TensorOf<[F16, F32]>:$boxes,
    TensorOf<[F16, F32]>:$scores,
    I32Tensor:$max_output_size,
    TensorOf<[F16, F32]>:$iou_threshold,
    TensorOf<[F16, F32]>:$score_threshold,

    DefaultValuedAttr<BoolAttr, "false">:$pad_to_max_output_size
  );

  let results = (outs
    I32Tensor:$selected_indices,
    I32Tensor:$valid_outputs
  );

  TF_DerivedOperandTypeAttr T_threshold = TF_DerivedOperandTypeAttr<3>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NonMaxSuppressionV5Op : TF_Op<"NonMaxSuppressionV5", [NoSideEffect]> {
  let summary = [{
Greedily selects a subset of bounding boxes in descending order of score,
  }];

  let description = [{
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold, score_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)
This op also supports a Soft-NMS (with Gaussian weighting) mode (c.f.
Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
of other overlapping boxes instead of directly causing them to be pruned.
To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
larger than 0.
  }];

  let arguments = (ins
    TensorOf<[F16, F32]>:$boxes,
    TensorOf<[F16, F32]>:$scores,
    I32Tensor:$max_output_size,
    TensorOf<[F16, F32]>:$iou_threshold,
    TensorOf<[F16, F32]>:$score_threshold,
    TensorOf<[F16, F32]>:$soft_nms_sigma,

    DefaultValuedAttr<BoolAttr, "false">:$pad_to_max_output_size
  );

  let results = (outs
    I32Tensor:$selected_indices,
    TensorOf<[F16, F32]>:$selected_scores,
    I32Tensor:$valid_outputs
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NotEqualOp : TF_Op<"NotEqual", [Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of (x != y) element-wise.";

  let description = [{
*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Str, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Str, TF_Uint8]>:$y,

    DefaultValuedAttr<BoolAttr, "true">:$incompatible_shape_error
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, Value* x, "
              "Value* y, BoolAttr incompatible_shape_error">
  ];

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_OneHotOp : TF_Op<"OneHot", [NoSideEffect]> {
  let summary = "Returns a one-hot tensor.";

  let description = [{
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
```
output =
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
```
output =
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
```
output =
  [
    [1.0, 0.0, 0.0]  // one_hot(0)
    [0.0, 0.0, 1.0]  // one_hot(2)
  ][
    [0.0, 1.0, 0.0]  // one_hot(1)
    [0.0, 0.0, 0.0]  // one_hot(-1)
  ]
```
  }];

  let arguments = (ins
    TensorOf<[I32, I64, TF_Uint8]>:$indices,
    I32Tensor:$depth,
    TF_Tensor:$on_value,
    TF_Tensor:$off_value,

    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr TI = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_PackOp : TF_Op<"Pack", [NoSideEffect]> {
  let summary = [{
Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
  }];

  let description = [{
Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values`, by packing them along the `axis` dimension.
Given a list of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of `unpack`.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$values,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_PadOp : TF_Op<"Pad", [NoSideEffect]> {
  let summary = "Pads a tensor with zeros.";

  let description = [{
This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$paddings
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
}

def TF_PadV2Op : TF_Op<"PadV2", [NoSideEffect]> {
  let summary = "Pads a tensor.";

  let description = [{
This operation pads `input` according to the `paddings` and `constant_values`
you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many padding values to add before the contents of `input` in that dimension,
and `paddings[D, 1]` indicates how many padding values to add after the contents
of `input` in that dimension. `constant_values` is a scalar tensor of the same
type as `input` that indicates the value to use for padding `input`.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# 'constant_values' is 0
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$paddings,
    TF_Tensor:$constant_values
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
}

def TF_PowOp : TF_Op<"Pow", [Broadcastable, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Computes the power of one value to another.";

  let description = [{
Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_PreventGradientOp : TF_Op<"PreventGradient", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
An identity op that triggers an error if a gradient is requested.
  }];

  let description = [{
When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, the TensorFlow gradient system
will return an error when trying to lookup the gradient of this op,
because no gradient must ever be registered for this function.  This
op exists to prevent subtle bugs from silently returning unimplemented
gradients in some corner cases.
  }];

  let arguments = (ins
    TF_Tensor:$input,

    StrAttr:$message
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ProdOp : TF_Op<"Prod", [NoSideEffect]> {
  let summary = [{
Computes the product of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_QuantizeAndDequantizeOp : TF_Op<"QuantizeAndDequantize", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Use QuantizeAndDequantizeV2 instead.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$input,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$range_given,
    DefaultValuedAttr<F32Attr, "0.0f">:$input_min,
    DefaultValuedAttr<F32Attr, "0.0f">:$input_max
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_QuantizeAndDequantizeV2Op : TF_Op<"QuantizeAndDequantizeV2", [NoSideEffect]> {
  let summary = "Quantizes then dequantizes a tensor.";

  let description = [{
This op simulates the precision loss from the quantized forward pass by:

1. Quantizing the tensor to fixed point numbers, which should match the target
   quantization method when it is used in inference.
2. Dequantizing it back to floating point numbers for the following ops, most
   likely matmul.

There are different ways to quantize. This version uses only scaling, so 0.0
maps to 0.

From the specified 'num_bits' in the quantized output type, it determines
minimum and maximum representable quantized values.

e.g.

*   [-128, 127] for signed, num_bits = 8, or
*   [0, 255] for unsigned, num_bits = 8.

If range_given == False, the initial input_min, input_max will be determined
automatically as the minimum and maximum values in the input tensor, otherwise
the specified values of input_min, input_max are used.

Note: If the input_min, input_max are specified, they do not need to equal the
actual minimum and maximum values in the tensor. e.g. in some cases it may be
beneficial to specify these values such that the low probability extremes of the
input distribution are clipped.

This op determines the maximum scale_factor that would map the initial
[input_min, input_max] range to a range that lies within the representable
quantized range.

It determines the scale from one of input_min and input_max, then updates the
other one to maximize the representable range.

e.g.

*   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
    5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
    would update input_max to be 127 / 12.8 = 9.921875
*   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
    10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
    would update input_min to be 128.0 / 12.7 = -10.07874
*   if the output is unsigned, input_min is forced to be 0, and only the
    specified input_max is used.

After determining the scale_factor and updating the input range, it applies the
following to each value in the 'input' tensor.

output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.

The above round function rounds the value based on the given round_mode.
  }];

  let arguments = (ins
    TF_FpTensor:$input,
    TF_FpTensor:$input_min,
    TF_FpTensor:$input_max,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$range_given,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HALF_TO_EVEN", "HALF_UP"]>, "HALF_TO_EVEN">:$round_mode,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_QuantizeAndDequantizeV3Op : TF_Op<"QuantizeAndDequantizeV3", [NoSideEffect]> {
  let summary = "Quantizes then dequantizes a tensor.";

  let description = [{
This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
tensor, so its value can change during training.
  }];

  let arguments = (ins
    TF_FpTensor:$input,
    TF_FpTensor:$input_min,
    TF_FpTensor:$input_max,
    I32Tensor:$num_bits,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<BoolAttr, "true">:$range_given,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RFFTOp : TF_Op<"RFFT", [NoSideEffect]> {
  let summary = "Real-valued fast Fourier transform.";

  let description = [{
Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
`fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
followed by the `fft_length / 2` positive-frequency terms.

Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$input,
    I32Tensor:$fft_length
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64]>:$output
  );

  TF_DerivedOperandTypeAttr Treal = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tcomplex = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomUniformOp : TF_Op<"RandomUniform", []> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$shape,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_RangeOp : TF_Op<"Range", [NoSideEffect]> {
  let summary = "Creates a sequence of numbers.";

  let description = [{
This operation creates a sequence of numbers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.

For example:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
```
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$start,
    TF_FpOrI32OrI64Tensor:$limit,
    TF_FpOrI32OrI64Tensor:$delta
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, Value* start, "
              "Value* limit, Value* delta">
  ];
}

def TF_RankOp : TF_Op<"Rank", [NoSideEffect]> {
  let summary = "Returns the rank of a tensor.";

  let description = [{
This operation returns an integer representing the rank of `input`.

For example:

```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as "order", "degree", or "ndims."
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    I32Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, Value* input">
  ];
}

def TF_ReadVariableOp : TF_Op<"ReadVariableOp", []> {
  let summary = "Reads the value of a variable.";

  let description = [{
The tensor returned by this operation is immutable.

The value returned by this operation is guaranteed to be influenced by all the
writes on which this operation depends directly or indirectly, and to not be
influenced by any of the writes which depend directly or indirectly on this
operation.
  }];

  let arguments = (ins
    TF_ResourceTensor:$resource
  );

  let results = (outs
    TF_Tensor:$value
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_RealOp : TF_Op<"Real", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns the real part of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
 part returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
```
  }];

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64]>:$input
  );

  let results = (outs
    TF_F32OrF64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_RealDivOp : TF_Op<"RealDiv", [Broadcastable, NoSideEffect]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise for real types.";

  let description = [{
If `x` and `y` are reals, this will return the floating-point division.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_ReciprocalOp : TF_Op<"Reciprocal", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the reciprocal of x element-wise.";

  let description = [{
I.e., \\(y = 1 / x\\).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_ReluOp : TF_Op<"Relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear: `max(features, 0)`.";

  let description = [{
See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
Example usage:
>>> tf.nn.relu([-2., 0., -0., 3.]).numpy()
array([ 0.,  0., -0.,  3.], dtype=float32)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Qint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$features
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Qint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_Relu6Op : TF_Op<"Relu6", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear 6: `min(max(features, 0), 6)`.";

  let description = [{
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$features
  );

  let results = (outs
    TF_IntOrFpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReluGradOp : TF_Op<"ReluGrad", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear gradients for a Relu operation.";

  let description = [{
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$gradients,
    TF_IntOrFpTensor:$features
  );

  let results = (outs
    TF_IntOrFpTensor:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReshapeOp : TF_Op<"Reshape", [NoSideEffect]> {
  let summary = "Reshapes a tensor.";

  let description = [{
Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of 1-D tensor `shape` is the special value -1, the size of that
dimension is computed so that the total size remains constant.  In particular, a
`shape` of `[-1]` flattens into 1-D.  At most one component of `shape` may be
unknown.

The `shape` must be 1-D and the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

It is an error if `shape` is not 1-D.

For example:

```
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
  }];

  let arguments = (ins
    TF_Tensor:$tensor,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<
      "Builder* builder, OperationState& result, Value* tensor, Value* shape">
  ];

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_ResizeBilinearOp : TF_Op<"ResizeBilinear", [NoSideEffect]> {
  let summary = "Resize `images` to `size` using bilinear interpolation.";

  let description = [{
Input images can be of different types but output images are always float.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Uint16, TF_Uint8]>:$images,
    I32Tensor:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    F32Tensor:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResizeNearestNeighborOp : TF_Op<"ResizeNearestNeighbor", [NoSideEffect]> {
  let summary = [{
Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let description = [{
  }];

  let arguments = (ins
    TensorOf<[F16, F32, F64, I16, I32, I64, I8, TF_Uint16, TF_Uint8]>:$images,
    I32Tensor:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    TensorOf<[F16, F32, F64, I16, I32, I64, I8, TF_Uint16, TF_Uint8]>:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResourceApplyAdamOp : TF_Op<"ResourceApplyAdam", []> {
  let summary = "Update '*var' according to the Adam algorithm.";

  let description = [{
$$\text{lr}_t := \mathrm{learning_rate} * \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
$$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
$$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
$$\text{variable} := \text{variable} - \text{lr}_t * m_t / (\sqrt{v_t} + \epsilon)$$
  }];

  let arguments = (ins
    TF_ResourceTensor:$var,
    TF_ResourceTensor:$m,
    TF_ResourceTensor:$v,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$beta1_power,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$beta2_power,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$lr,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$beta1,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$beta2,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$epsilon,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$use_nesterov
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyGradientDescentOp : TF_Op<"ResourceApplyGradientDescent", []> {
  let summary = "Update '*var' by subtracting 'alpha' * 'delta' from it.";

  let description = [{
  }];

  let arguments = (ins
    TF_ResourceTensor:$var,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$alpha,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$delta,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResourceApplyKerasMomentumOp : TF_Op<"ResourceApplyKerasMomentum", []> {
  let summary = "Update '*var' according to the momentum scheme.";

  let description = [{
Set use_nesterov = True if you want to use Nesterov momentum.

accum = accum * momentum - lr * grad
var += accum
  }];

  let arguments = (ins
    TF_ResourceTensor:$var,
    TF_ResourceTensor:$accum,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$lr,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$grad,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$momentum,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$use_nesterov
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ReverseSequenceOp : TF_Op<"ReverseSequence", [NoSideEffect]> {
  let summary = "Reverses variable length slices.";

  let description = [{
This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.

For example:

```
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

```
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
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$seq_lengths,

    I64Attr:$seq_dim,
    DefaultValuedAttr<I64Attr, "0">:$batch_dim
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReverseV2Op : TF_Op<"ReverseV2", [NoSideEffect]> {
  let summary = "Reverses specific dimensions of a tensor.";

  let description = [{
NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

```
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is [-1]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str, TF_Uint16, TF_Uint8]>:$tensor,
    TF_I32OrI64Tensor:$axis
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str, TF_Uint16, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_RightShiftOp : TF_Op<"RightShift", [Broadcastable, NoSideEffect]>,
                      WithBroadcastableBinOpBuilder {
  let summary = "Elementwise computes the bitwise right-shift of `x` and `y`.";

  let description = [{
Performs a logical shift for unsigned integer types, and an arithmetic shift
for signed integer types.

If `y` is negative, or greater than or equal to than the width of `x` in bits
the result is implementation defined.

Example:

```python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
import numpy as np
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]

for dtype in dtype_list:
  lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

  right_shift_result = bitwise_ops.right_shift(lhs, rhs)

  print(right_shift_result)

# This will print:
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int8)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int16)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int32)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int64)

lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
bitwise_ops.right_shift(lhs, rhs)
# <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
```
  }];

  let arguments = (ins
    TF_IntTensor:$x,
    TF_IntTensor:$y
  );

  let results = (outs
    TF_IntTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RoundOp : TF_Op<"Round", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Rounds the values of a tensor to the nearest integer, element-wise.
  }];

  let description = [{
Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RsqrtOp : TF_Op<"Rsqrt", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes reciprocal of square root of x element-wise.";

  let description = [{
I.e., \\(y = 1 / \sqrt{x}\\).
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RsqrtGradOp : TF_Op<"RsqrtGrad", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the gradient for the rsqrt of `x` wrt its input.";

  let description = [{
Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
is the corresponding input gradient.
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$y,
    TF_FpOrComplexTensor:$dy
  );

  let results = (outs
    TF_FpOrComplexTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SelectOp : TF_Op<"Select", [NoSideEffect]> {
  let summary = "Selects elements from `x` or `y`, depending on `condition`.";

  let description = [{
The `x`, and `y` tensors must all have the same shape, and the
output will also have that shape.

The `condition` tensor must be a scalar if `x` and `y` are scalars.
If `x` and `y` are vectors or higher rank, then `condition` must be either a
scalar, a vector with size matching the first dimension of `x`, or must have
the same shape as `x`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `x` (if true) or `y` (if false).

If `condition` is a vector and `x` and `y` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `x` and `y`.
If `condition` has the same shape as `x` and `y`, then it chooses which
element to copy from `x` and `y`.

For example:

```python
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e)  # => [[1, 6], [7, 4]]


# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]

```
  }];

  let arguments = (ins
    I1Tensor:$condition,
    TF_Tensor:$t,
    TF_Tensor:$e
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_SelectV2Op : TF_Op<"SelectV2", [NoSideEffect]> {
  let summary = "";

  let description = [{
  }];

  let arguments = (ins
    I1Tensor:$condition,
    TF_Tensor:$t,
    TF_Tensor:$e
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ShapeOp : TF_Op<"Shape", [NoSideEffect]> {
  let summary = "Returns the shape of a tensor.";

  let description = [{
This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];

  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, Value* input, BoolAttr use32Bit">
  ];

  let hasFolder = 1;
}

def TF_ShapeNOp : TF_Op<"ShapeN", [NoSideEffect]> {
  let summary = "Returns shape of tensors.";

  let description = [{
This operation returns N 1-D integer tensors representing shape of `input[i]s`.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$input
  );

  let results = (outs
    Variadic<TF_I32OrI64Tensor>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];

  let hasFolder = 1;
}

def TF_SigmoidOp : TF_Op<"Sigmoid", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes sigmoid of `x` element-wise.";

  let description = [{
Specifically, `y = 1 / (1 + exp(-x))`.
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SignOp : TF_Op<"Sign", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns an element-wise indication of the sign of a number.";

  let description = [{
`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

Example usage:
>>> tf.math.sign([0., 2., -3.])
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SinOp : TF_Op<"Sin", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes sine of x element-wise.";

  let description = [{
Given an input tensor, this function computes sine of every
  element in the tensor. Input range is `(-inf, inf)` and
  output range is `[-1,1]`.

  ```python
  x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
  tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
  ```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SizeOp : TF_Op<"Size", [NoSideEffect]> {
  let summary = "Returns the size of a tensor.";

  let description = [{
This operation returns an integer representing the number of elements in
`input`.

For example:

```
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_SliceOp : TF_Op<"Slice", [NoSideEffect]> {
  let summary = "Return a slice from 'input'.";

  let description = [{
The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$begin,
    TF_I32OrI64Tensor:$size
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_SnapshotOp : TF_Op<"Snapshot", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns a copy of the input tensor.";

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SoftmaxOp : TF_Op<"Softmax", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes softmax activations.";

  let description = [{
For each batch `i` and class `j` we have

    $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
  }];

  let arguments = (ins
    TF_FpTensor:$logits
  );

  let results = (outs
    TF_FpTensor:$softmax
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_SoftmaxCrossEntropyWithLogitsOp : TF_Op<"SoftmaxCrossEntropyWithLogits", [NoSideEffect]> {
  let summary = [{
Computes softmax cross entropy cost and gradients to backpropagate.
  }];

  let description = [{
Inputs are the logits, not probabilities.
  }];

  let arguments = (ins
    TF_FpTensor:$features,
    TF_FpTensor:$labels
  );

  let results = (outs
    TF_FpTensor:$loss,
    TF_FpTensor:$backprop
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_SoftplusOp : TF_Op<"Softplus", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes softplus: `log(exp(features) + 1)`.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpTensor:$features
  );

  let results = (outs
    TF_FpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SpaceToBatchNDOp : TF_Op<"SpaceToBatchND", [NoSideEffect]> {
  let summary = "SpaceToBatch for N-D tensors of type T.";

  let description = [{
This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
"batch" dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`.  See below for a
precise description.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$block_shape,
    TF_I32OrI64Tensor:$paddings
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tblock_shape = TF_DerivedOperandTypeAttr<1>;
}

def TF_SpaceToDepthOp : TF_Op<"SpaceToDepth", [NoSideEffect]> {
  let summary = "SpaceToDepth for tensors of type T.";

  let description = [{
Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size.

  * Non-overlapping blocks of size `block_size x block size` are rearranged
    into depth at each location.
  * The depth of the output tensor is `block_size * block_size * input_depth`.
  * The Y, X coordinates within each block of the input become the high order
    component of the output channel index.
  * The input tensor's height and width must be divisible by block_size.

The `data_format` attr specifies the layout of the input and output tensors
with the following options:
  "NHWC": `[ batch, height, width, channels ]`
  "NCHW": `[ batch, channels, height, width ]`
  "NCHW_VECT_C":
      `qint8 [ batch, channels / 4, height, width, 4 ]`

It is useful to consider the operation as transforming a 6-D Tensor.
e.g. for data_format = NHWC,
     Each element in the input tensor can be specified via 6 coordinates,
     ordered by decreasing memory layout significance as:
     n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
                        within the output image, bX, bY means coordinates
                        within the input block, iC means input channels).
     The output would be a transpose to the following layout:
     n,oY,oX,bY,bX,iC

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
block_size = 2:

```
x = [[[[1], [2]],
      [[3], [4]]]]
```

This operation will output a tensor of shape `[1, 1, 1, 4]`:

```
[[[[1, 2, 3, 4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 * block_size * block_size).
The output element shape is `[1, 1, 4]`.

For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

This operation, for block_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`

```
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

```
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```

the operator will return the following tensor of shape `[1 2 2 4]`:

```
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,

    Confined<I64Attr, [IntMinValue<2>]>:$block_size,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "NHWC">:$data_format
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SparseSoftmaxCrossEntropyWithLogitsOp : TF_Op<"SparseSoftmaxCrossEntropyWithLogits", [NoSideEffect]> {
  let summary = [{
Computes softmax cross entropy cost and gradients to backpropagate.
  }];

  let description = [{
Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.

Inputs are the logits, not probabilities.
  }];

  let arguments = (ins
    TF_FpTensor:$features,
    TF_I32OrI64Tensor:$labels
  );

  let results = (outs
    TF_FpTensor:$loss,
    TF_FpTensor:$backprop
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tlabels = TF_DerivedOperandTypeAttr<1>;
}

def TF_SparseToDenseOp : TF_Op<"SparseToDense", [NoSideEffect]> {
  let summary = "Converts a sparse representation into a dense tensor.";

  let description = [{
Builds an array `dense` with shape `output_shape` such that

```
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)

# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]

# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```

All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.

Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$sparse_indices,
    TF_I32OrI64Tensor:$output_shape,
    TF_Tensor:$sparse_values,
    TF_Tensor:$default_value,

    DefaultValuedAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    TF_Tensor:$dense
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_SplitOp : TF_Op<"Split", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    I32Tensor:$split_dim,
    TF_Tensor:$value
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultSizeAttr num_split = TF_DerivedResultSizeAttr<0>;

  let verifier = [{ return Verify(*this); }];
}

def TF_SplitVOp : TF_Op<"SplitV", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$value,
    TF_I32OrI64Tensor:$size_splits,
    I32Tensor:$split_dim
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultSizeAttr num_split = TF_DerivedResultSizeAttr<0>;

  let verifier = [{ return Verify(*this); }];
}

def TF_SqrtOp : TF_Op<"Sqrt", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes square root of x element-wise.";

  let description = [{
I.e., \\(y = \sqrt{x} = x^{1/2}\\).
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SquareOp : TF_Op<"Square", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes square of x element-wise.";

  let description = [{
I.e., \\(y = x * x = x^2\\).
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_SquaredDifferenceOp : TF_Op<"SquaredDifference", [Broadcastable, Commutative, NoSideEffect]>,
                             WithBroadcastableBinOpBuilder {
  let summary = "Returns (x - y)(x - y) element-wise.";

  let description = [{
*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SqueezeOp : TF_Op<"Squeeze", [NoSideEffect]> {
  let summary = "Removes dimensions of size 1 from the shape of a tensor.";

  let description = [{
Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.

For example:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,

    DefaultValuedAttr<I64ArrayAttr, "{}">:$squeeze_dims
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_StopGradientOp : TF_Op<"StopGradient", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Stops gradient computation.";

  let description = [{
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
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_StridedSliceOp : TF_Op<"StridedSlice", [NoSideEffect]> {
  let summary = "Return a strided slice from `input`.";

  let description = [{
Note, most python users will want to use the Python `Tensor.__getitem__`
or `Variable.__getitem__` rather than this op directly.

The goal of this op is to produce a new tensor with a subset of
the elements from the `n` dimensional `input` tensor. The subset is chosen using
a sequence of `m` sparse range specifications encoded into the arguments
of this function. Note, in some cases
`m` could be equal to `n`, but this need not be the case. Each
range specification entry can be one of the following:

- An ellipsis (...). Ellipses are used to imply zero or more
  dimensions of full-dimension selection and are produced using
  `ellipsis_mask`. For example, `foo[...]` is the identity slice.

- A new axis. This is used to insert a new shape=1 dimension and is
  produced using `new_axis_mask`. For example, `foo[:, ...]` where
  `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.


- A range `begin:end:stride`. This is used to specify how much to choose from
  a given dimension. `stride` can be any integer but 0.  `begin` is an integer
  which represents the index of the first value to select while `end` represents
  the index of the last value to select. The number of values selected in each
  dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
  `begin` and `end` can be negative where `-1` is the last element, `-2` is
  the second to last. `begin_mask` controls whether to replace the explicitly
  given `begin` with an implicit effective value of `0` if `stride > 0` and
  `-1` if `stride < 0`. `end_mask` is analogous but produces the number
  required to create the largest open interval. For example, given a shape
  `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
  not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
  and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
  first dimension of a tensor while dropping the last two (in the original
  order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.

- A single index. This is used to keep only elements that have a given
  index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
  shape `(6,)` tensor. This is encoded in `begin` and `end` and
  `shrink_axis_mask`.

Each conceptual range specification is encoded in the op's argument. This
encoding is best understand by considering a non-trivial example. In
particular,
`foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as

```
begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
end = [2, 4, x, x, -3, x]
strides = [1, 1, x, x, -1, 1]
begin_mask = 1<<4 | 1 << 5 = 48
end_mask = 1<<5 = 32
ellipsis_mask = 1<<3 = 8
new_axis_mask = 1<<2 4
shrink_axis_mask = 1<<0
```

In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
the slice becomes (2, 1, 5, 5, 2, 5).
Let us walk step by step through each argument specification.

1.  The first argument in the example slice is turned into `begin = 1` and
`end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
also set the appropriate bit in `shrink_axis_mask`.

2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
zero bits contributed.

3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
dimension in the final shape. Dummy values are contributed to begin,
end and stride, while the new_axis_mask bit is set.

4. `...` grab the full ranges from as many dimensions as needed to
fully specify a slice for every dimension of the input shape.

5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
with a dimension that has shape `s` is converted to a positive index
`s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
is done internally so begin, end and strides receive x, -3, and -1.
The appropriate begin_mask bit is set to indicate the start range is the
full range (ignoring the x).

6. `:` indicates that the entire contents of the corresponding dimension
is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
`end_mask` are also set.

*Requirements*:
  `0 != strides[i] for i in [0, m)`
  `ellipsis_mask must be a power of two (only one ellipsis)`
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$begin,
    TF_I32OrI64Tensor:$end,
    TF_I32OrI64Tensor:$strides,

    DefaultValuedAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedAttr<I64Attr, "0">:$end_mask,
    DefaultValuedAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedAttr<I64Attr, "0">:$shrink_axis_mask
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;

  let verifier = [{ return VerifyStridedSliceBase(*this); }];
}

def TF_StridedSliceGradOp : TF_Op<"StridedSliceGrad", [NoSideEffect]> {
  let summary = "Returns the gradient of `StridedSlice`.";

  let description = [{
Since `StridedSlice` cuts out pieces of its `input` which is size
`shape`, its gradient will have the same shape (which is passed here
as `shape`). The gradient will be zero in any element that the slice
does not select.

Arguments are the same as StridedSliceGrad with the exception that
`dy` is the input gradient to be propagated and `shape` is the
shape of `StridedSlice`'s `input`.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$shape,
    TF_I32OrI64Tensor:$begin,
    TF_I32OrI64Tensor:$end,
    TF_I32OrI64Tensor:$strides,
    TF_Tensor:$dy,

    DefaultValuedAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedAttr<I64Attr, "0">:$end_mask,
    DefaultValuedAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedAttr<I64Attr, "0">:$shrink_axis_mask
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<4>;
  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{ return Verify(*this); }];

  let extraClassDeclaration = [{
    // If sliced shape is able to be deduced, returns true, updates `shape`
    // with the final shape after performing StridedSlice, and updates
    // `begin_indices`, `end_indices`, and `strides` with their canonical
    // values, respectively.
    bool GetSlicedShapeAndBoundRanges(
      ::llvm::SmallVectorImpl<int64_t> *shape,
      ::llvm::SmallVectorImpl<int64_t> *begin_indices,
      ::llvm::SmallVectorImpl<int64_t> *end_indices,
      ::llvm::SmallVectorImpl<int64_t> *strides);
  }];
}

def TF_SubOp : TF_Op<"Sub", [Broadcastable, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x - y element-wise.";

  let description = [{
*NOTE*: `Subtract` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_SumOp : TF_Op<"Sum", [NoSideEffect]> {
  let summary = "Computes the sum of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *input, "
    "Value *reduction_indices, BoolAttr keep_dims"
  >];
}

def TF_TPUCompilationResultOp : TF_Op<"TPUCompilationResult", [NoSideEffect]> {
  let summary = "Returns the result of a TPU compilation.";

  let description = [{
This operation returns the result of a TPU compilation as a serialized
CompilationResultProto, which holds a status and an error message if an error
occurred during compilation.
  }];

  let arguments = (ins);

  let results = (outs
    TF_StrTensor:$output
  );
}

def TF_TPUExecuteOp : TF_Op<"TPUExecute", []> {
  let summary = "Op that loads and executes a TPU program on a TPU device.";

  let description = [{
For the internal use of the distributed TPU compiler.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,
    TF_StrTensor:$key
  );

  let results = (outs
    Variadic<TF_Tensor>:$results
  );

  TF_DerivedOperandTypeListAttr Targs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tresults = TF_DerivedResultTypeListAttr<0>;
}

def TF_TPUExecuteAndUpdateVariablesOp : TF_Op<"TPUExecuteAndUpdateVariables", []> {
  let summary = [{
Op that executes a program with optional in-place variable updates.
  }];

  let description = [{
It (optionally) reads device variables, loads and executes a TPU program on a
TPU device, and then (optionally) in-place updates variables using the program
outputs, as specified in attributes device_var_reads_indices (program input
indices from directly reading variables) and device_var_updates_indices (program
output indices used to update variables, -1 means no-update/read-only). Such
program outputs are consumed by these variables will not appear in the op
output. For the internal use of the distributed TPU compiler.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,
    TF_StrTensor:$key,

    I64ArrayAttr:$device_var_reads_indices,
    I64ArrayAttr:$device_var_updates_indices
  );

  let results = (outs
    Variadic<TF_Tensor>:$results
  );

  TF_DerivedOperandTypeListAttr Targs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tresults = TF_DerivedResultTypeListAttr<0>;
}

def TF_TPUReplicatedInputOp : TF_Op<"TPUReplicatedInput", [NoSideEffect]> {
  let summary = "Connects N inputs to an N-way replicated TPU computation.";

  let description = [{
This operation holds a replicated input to a `tpu.replicate()` computation subgraph.
Each replicated input has the same shape and type alongside the output.

For example:
```
%a = "tf.opA"()
%b = "tf.opB"()
%replicated_input = "tf.TPUReplicatedInput"(%a, %b)
%computation = "tf.Computation"(%replicated_input)
```
The above computation has a replicated input of two replicas.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    DefaultValuedAttr<BoolAttr, "false">:$is_mirrored_variable,
    DefaultValuedAttr<I64Attr, "-1">:$index
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_TPUReplicatedOutputOp : TF_Op<"TPUReplicatedOutput", [NoSideEffect]> {
  let summary = "Connects N outputs from an N-way replicated TPU computation.";

  let description = [{
This operation holds a replicated output from a `tpu.replicate()` computation subgraph.
Each replicated output has the same shape and type alongside the input.

For example:
```
%computation = "tf.Computation"()
%replicated_output:2 = "tf.TPUReplicatedOutput"(%computation)
```
The above computation has a replicated output of two replicas.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedResultSizeAttr num_replicas = TF_DerivedResultSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TanhOp : TF_Op<"Tanh", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes hyperbolic tangent of `x` element-wise.";

  let description = [{
Given an input tensor, this function computes hyperbolic tangent of every
  element in the tensor. Input range is `[-inf, inf]` and
  output range is `[-1,1]`.

  ```python
  x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
  tf.math.tanh(x) ==> [-1. -0.99990916 -0.46211717 0.7615942 0.8336547 0.9640276 0.9950547 1.]
  ```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TanhGradOp : TF_Op<"TanhGrad", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the gradient for the tanh of `x` wrt its input.";

  let description = [{
Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient.
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$y,
    TF_FpOrComplexTensor:$dy
  );

  let results = (outs
    TF_FpOrComplexTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TensorListFromTensorOp : TF_Op<"TensorListFromTensor", [NoSideEffect]> {
  let summary = [{
Creates a TensorList which, when stacked, has the value of `tensor`.
  }];

  let description = [{
Each tensor in the result list corresponds to one row of the input tensor.

tensor: The input tensor.
output_handle: The list.
  }];

  let arguments = (ins
    TF_Tensor:$tensor,
    TF_I32OrI64Tensor:$element_shape
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<0>;
}

def TF_TensorListGetItemOp : TF_Op<"TensorListGetItem", [NoSideEffect]> {
  let summary = "";

  let description = [{
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    I32Tensor:$index,
    I32Tensor:$element_shape
  );

  let results = (outs
    TF_Tensor:$item
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorListLengthOp : TF_Op<"TensorListLength", [NoSideEffect]> {
  let summary = "Returns the number of tensors in the input tensor list.";

  let description = [{
input_handle: the input list
length: the number of tensors in the list
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle
  );

  let results = (outs
    I32Tensor:$length
  );
}

def TF_TensorListPushBackOp : TF_Op<"TensorListPushBack", [NoSideEffect]> {
  let summary = [{
Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.
  }];

  let description = [{
tensor: The tensor to put on the list.
input_handle: The old list.
output_handle: A list with the elements of the old list followed by tensor.
element_dtype: the type of elements in the list.
element_shape: a shape compatible with that of elements in the list.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Tensor:$tensor
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorListResizeOp : TF_Op<"TensorListResize", [NoSideEffect]> {
  let summary = "Resizes the list.";

  let description = [{
input_handle: the input list
size: size of the output list
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    I32Tensor:$size
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );
}

def TF_TensorListSetItemOp : TF_Op<"TensorListSetItem", [NoSideEffect]> {
  let summary = "";

  let description = [{
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    I32Tensor:$index,
    TF_Tensor:$item
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_TensorListStackOp : TF_Op<"TensorListStack", [NoSideEffect]> {
  let summary = "Stacks all tensors in the list.";

  let description = [{
Requires that all tensors have the same shape.

input_handle: the input list
tensor: the gathered result
num_elements: optional. If not -1, the number of elements in the list.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    I32Tensor:$element_shape,

    DefaultValuedAttr<I64Attr, "-1">:$num_elements
  );

  let results = (outs
    TF_Tensor:$tensor
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_TileOp : TF_Op<"Tile", [NoSideEffect]> {
  let summary = "Constructs a tensor by tiling a given tensor.";

  let description = [{
This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

>>> a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
>>> b = tf.constant([1,2], tf.int32)
>>> tf.tile(a, b)
<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]], dtype=int32)>
>>> c = tf.constant([2,1], tf.int32)
>>> tf.tile(a, c)
<tf.Tensor: shape=(4, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]], dtype=int32)>
>>> d = tf.constant([2,2], tf.int32)
>>> tf.tile(a, d)
<tf.Tensor: shape=(4, 6), dtype=int32, numpy=
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6],
       [1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]], dtype=int32)>
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$multiples
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tmultiples = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  // TODO(parkers): Add folds for multiples = [1,...].
  // TODO(parkers): Add errors for negative multiples and multiples.size() !=
  // input.rank()
}

def TF_TopKV2Op : TF_Op<"TopKV2", [NoSideEffect]> {
  let summary = [{
Finds values and indices of the `k` largest elements for the last dimension.
  }];

  let description = [{
If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$input,
    I32Tensor:$k,

    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );

  let results = (outs
    TF_IntOrFpTensor:$values,
    I32Tensor:$indices
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{ return Verify(*this); }];
}

def TF_TransposeOp : TF_Op<"Transpose", [NoSideEffect]> {
  let summary = "Shuffle dimensions of x according to a permutation.";

  let description = [{
The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
  }];

  let arguments = (ins
    TF_Tensor:$x,
    TF_I32OrI64Tensor:$perm
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tperm = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<
      "Builder* builder, OperationState& result, Value* x, Value* perm">
  ];

  let verifier = [{
    return Verify(*this);
  }];

  let hasFolder = 1;
}

def TF_TruncateDivOp : TF_Op<"TruncateDiv", [Broadcastable, NoSideEffect]>,
                       WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise for integer types.";

  let description = [{
Truncation designates that negative numbers will round fractional quantities
toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
than Python semantics. See `FloorDiv` for a division function that matches
Python Semantics.

*NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$x,
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Uint16, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_UniqueOp : TF_Op<"Unique", [NoSideEffect]> {
  let summary = "Finds unique elements in a 1-D tensor.";

  let description = [{
This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`; `x` does not need to be sorted.
This operation also returns a tensor `idx` the same size as `x` that contains
the index of each value of `x` in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

Examples:

```
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

```
# tensor 'x' is [4, 5, 1, 2, 3, 3, 4, 5]
y, idx = unique(x)
y ==> [4, 5, 1, 2, 3]
idx ==> [0, 1, 2, 3, 4, 4, 0, 1]
```
  }];

  let arguments = (ins
    TF_Tensor:$x
  );

  let results = (outs
    TF_Tensor:$y,
    TF_I32OrI64Tensor:$idx
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_idx = TF_DerivedResultTypeAttr<1>;
}

def TF_UnpackOp : TF_Op<"Unpack", [NoSideEffect]> {
  let summary = [{
Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
  }];

  let description = [{
Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
  and each tensor in `output` will have shape `(B, C, D)`. (Note that the
  dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
  and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of `pack`.
  }];

  let arguments = (ins
    TF_Tensor:$value,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedResultSizeAttr num = TF_DerivedResultSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{ return Verify(*this); }];
}

def TF_UnsortedSegmentMaxOp : TF_Op<"UnsortedSegmentMax", [NoSideEffect]> {
  let summary = "Computes the maximum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the maximum such that:

\\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
that `segment_ids[j...] == i`.

If the maximum is empty for a given segment ID `i`, it outputs the smallest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::lowest()`.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
</div>

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 4,  3, 3, 4],
#       [5,  6, 7, 8]]
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    TF_I32OrI64Tensor:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    TF_IntOrFpTensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let verifier = [{ return VerifyUnsortedSegmentReduction(*this); }];
}

def TF_UnsortedSegmentMinOp : TF_Op<"UnsortedSegmentMin", [NoSideEffect]> {
  let summary = "Computes the minimum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the minimum such that:

\\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
that `segment_ids[j...] == i`.

If the minimum is empty for a given segment ID `i`, it outputs the largest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::max()`.

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 1,  2, 2, 1],
#       [5,  6, 7, 8]]
```

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    TF_I32OrI64Tensor:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    TF_IntOrFpTensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let verifier = [{ return VerifyUnsortedSegmentReduction(*this); }];
}

def TF_UnsortedSegmentProdOp : TF_Op<"UnsortedSegmentProd", [NoSideEffect]> {
  let summary = "Computes the product along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the product of all
entries belonging to a segment such that:

\\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
`j...` such that `segment_ids[j...] == i`.

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 4,  6, 6, 4],
#       [5,  6, 7, 8]]
```

If there is no entry for a given segment ID `i`, it outputs 1.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    TF_I32OrI64Tensor:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let verifier = [{ return VerifyUnsortedSegmentReduction(*this); }];
}

def TF_UnsortedSegmentSumOp : TF_Op<"UnsortedSegmentSum", [NoSideEffect]> {
  let summary = "Computes the sum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.
If the given segment ID `i` is negative, the value is dropped and will not be
added to the sum of the segment.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
</div>

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 5,  5, 5, 5],
#       [5,  6, 7, 8]]
```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    TF_I32OrI64Tensor:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let verifier = [{ return VerifyUnsortedSegmentReduction(*this); }];
}

def TF_VariableShapeOp : TF_Op<"VariableShape", []> {
  let summary = "Returns the shape of the variable pointed to by `resource`.";

  let description = [{
This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```
  }];

  let arguments = (ins
    TF_ResourceTensor:$input
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_WhereOp : TF_Op<"Where", [NoSideEffect]> {
  let summary = "Returns locations of nonzero / true values in a tensor.";

  let description = [{
This operation returns the coordinates of true elements in `condition`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`condition`. Indices are output in row-major order.

For example:

```
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `condition` tensor is [[[True, False]
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

# `condition` tensor is [[[1.5,  0.0]
#                     [-0.5, 0.0]]
#                    [[0.0,  0.25]
#                     [0.0,  0.75]]
#                    [[0.0,  0.0]
#                     [0.0,  0.01]]]
# 'input' has 5 nonzero values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]

# `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
#                     [0.0 + 0.5j, 0.0  + 0.0j]]
#                    [[0.0 + 0.0j, 0.25 + 1.5j]
#                     [0.0 + 0.0j, 0.75 + 0.0j]]
#                    [[0.0 + 0.0j, 0.0  + 0.0j]
#                     [0.0 + 0.0j, 0.01 + 0.0j]]]
# 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input
  );

  let results = (outs
    I64Tensor:$index
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XdivyOp : TF_Op<"Xdivy", [Broadcastable, NoSideEffect]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if x == 0, and x / y otherwise, elementwise.";

  let description = [{
  }];

  let arguments = (ins
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$x,
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$y
  );

  let results = (outs
    TensorOf<[F16, F32, F64, TF_Complex128, TF_Complex64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_ZerosLikeOp : TF_Op<"ZerosLike", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns a tensor of zeros with the same shape and type as x.";

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$x
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
