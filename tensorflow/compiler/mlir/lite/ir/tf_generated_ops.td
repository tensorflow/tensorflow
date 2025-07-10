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

// This is the auto-generated operation definition file for TensorFlow.
//
// PLEASE DO NOT MANUALLY EDIT THIS FILE!
// ONLY EXCEPTION: FIELDS THAT CANNOT BE GENERATED
//
// If you absolutely need to modify the generated fields of an op, move the op
// definition to `tf_ops.td` and perform the modification there. Generated
// fields and the process to generate them are documented at:
// mlir/tensorflow/dialectgen/README.md
//
// This file contains TensorFlow ops whose definitions are programmatically
// generated from the TF op registration and the api-def-files in the following
// folder:
// tensorflow/core/api_def/base_api
// The generated fields for an op include name, summary, description, traits,
// arguments, results, derived attributes. Therefore, modifications to these
// fields will NOT be respected upon subsequent refreshes. However, additional
// fields after those fields will be retained.
//
// Ops in this file are sorted alphabetically.

include "tensorflow/compiler/mlir/lite/ir/tf_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def TF_AbsOp : TF_Op<"Abs", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the absolute value of a tensor.";

  let description = [{
Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_AddOp : TF_Op<"Add", [Pure, ResultsBroadcastableShape, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Given two input tensors, the `tf.add` operation computes the sum for every element in the tensor.

Both input and output have a range `(-inf, inf)`.
  }];

  let arguments = (ins
    TF_NumberNotQuantizedOrStrTensor:$x,
    TF_NumberNotQuantizedOrStrTensor:$y
  );

  let results = (outs
    TF_NumberNotQuantizedOrStrTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_AddNOp : TF_Op<"AddN", [Commutative, Pure]> {
  let summary = "Add all input tensors element wise.";

  let description = [{
Inputs must be of same size and shape.

  ```python
  x = [9, 7, 10]
  tf.math.add_n(x) ==> 26
  ```
  }];

  let arguments = (ins
    Variadic<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>>:$inputs
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>:$sum
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_AddV2Op : TF_Op<"AddV2", [Commutative, Pure, ResultsBroadcastableShape, TF_CwiseBinary, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_AllOp : TF_Op<"All", [Pure]> {
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
    Arg<TF_BoolTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_BoolTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_AnyOp : TF_Op<"Any", [Pure]> {
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
    Arg<TF_BoolTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_BoolTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_ApproxTopKOp : TF_Op<"ApproxTopK", [Pure]> {
  let summary = [{
Returns min/max k values and their indices of the input operand in an approximate manner.
  }];

  let description = [{
See https://arxiv.org/abs/2206.14286 for the algorithm details.
This op is only optimized on TPU currently.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{Array to search. Must be at least 1-D of the floating type}]>:$input,

    ConfinedAttr<I64Attr, [IntMinValue<0>]>:$k,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$reduction_dimension,
    DefaultValuedOptionalAttr<F32Attr, "0.95f">:$recall_target,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$is_max_k,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$reduction_input_size_override,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$aggregate_to_topk
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{The min/max k values along the `reduction_dimension` of the `input` operand.
The dimension are the same as the `input` operand except for the
`reduction_dimension`: when `aggregate_to_topk` is true, the reduction
dimension is `k`; otherwise, it is greater equals to `k` where the size is
implementation-defined.}]>:$values,
    Res<TF_Int32Tensor, [{The indices of `values` along the `reduction_dimension` of the `input` operand.}]>:$indices
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ArgMaxOp : TF_Op<"ArgMax", [Pure]> {
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{int16, int32 or int64, must be in the range `[-rank(input), rank(input))`.
Describes which dimension of the input Tensor to reduce across. For vectors,
use dimension = 0.}]>:$dimension
  );

  let results = (outs
    TensorOf<[TF_Int16, TF_Int32, TF_Int64, TF_Uint16]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr output_type = TF_DerivedResultTypeAttr<0>;
}

def TF_ArgMinOp : TF_Op<"ArgMin", [Pure]> {
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    Arg<TF_I32OrI64Tensor, [{int32 or int64, must be in the range `[-rank(input), rank(input))`.
Describes which dimension of the input Tensor to reduce across. For vectors,
use dimension = 0.}]>:$dimension
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr output_type = TF_DerivedResultTypeAttr<0>;
}

def TF_AssignVariableOp : TF_Op<"AssignVariableOp", []> {
  let summary = "Assigns a new value to a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{handle to the resource in which to store the variable.}], [TF_VariableWrite]>:$resource,
    Arg<TF_Tensor, [{the value to set the new tensor to use.}]>:$value,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$validate_shape
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_Atan2Op : TF_Op<"Atan2", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                 WithBroadcastableBinOpBuilder {
  let summary = [{
Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
  }];

  let description = [{
This is the angle \\( \theta \in [-\pi, \pi] \\) such that
\\[ x = r \cos(\theta) \\]
and
\\[ y = r \sin(\theta) \\]
where \\(r = \sqrt{x^2 + y^2} \\).

For example:

>>> x = [1., 1.]
>>> y = [1., -1.]
>>> print((tf.math.atan2(y,x) * (180 / np.pi)).numpy())
[ 45. -45.]
  }];

  let arguments = (ins
    TF_FloatTensor:$y,
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AvgPoolOp : TF_Op<"AvgPool", [Pure]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{4-D with shape `[batch, height, width, channels]`.}]>:$value,

    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedOptionalAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_FloatTensor, [{The average pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BatchMatMulOp : TF_Op<"BatchMatMul", [Pure, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{2-D or higher with shape `[..., r_x, c_x]`.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{2-D or higher with shape `[..., r_y, c_y]`.}]>:$y,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_y,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def TF_BatchMatMulV2Op : TF_Op<"BatchMatMulV2", [Pure, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{2-D or higher with shape `[..., r_x, c_x]`.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{2-D or higher with shape `[..., r_y, c_y]`.}]>:$y,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_y,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_BatchMatMulV3Op : TF_Op<"BatchMatMulV3", [Pure]> {
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

*NOTE*: `BatchMatMulV3` supports broadcasting in the batch dimensions. More
about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint8]>, [{2-D or higher with shape `[..., r_x, c_x]`.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint8]>, [{2-D or higher with shape `[..., r_y, c_y]`.}]>:$y,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_y,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr Ta = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tb = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_BatchToSpaceNDOp : TF_Op<"BatchToSpaceND", [Pure]> {
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
    Arg<TF_Tensor, [{N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
where spatial_shape has M dimensions.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D with shape `[M]`, all values must be >= 1.}]>:$block_shape,
    Arg<TF_I32OrI64Tensor, [{2-D with shape `[M, 2]`, all values must be >= 0.
  `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
  dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
  required that
  `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.

This operation is equivalent to the following steps:

1. Reshape `input` to `reshaped` of shape:
     [block_shape[0], ..., block_shape[M-1],
      batch / prod(block_shape),
      input_shape[1], ..., input_shape[N-1]]

2. Permute dimensions of `reshaped` to produce `permuted` of shape
     [batch / prod(block_shape),

      input_shape[1], block_shape[0],
      ...,
      input_shape[M], block_shape[M-1],

      input_shape[M+1], ..., input_shape[N-1]]

3. Reshape `permuted` to produce `reshaped_permuted` of shape
     [batch / prod(block_shape),

      input_shape[1] * block_shape[0],
      ...,
      input_shape[M] * block_shape[M-1],

      input_shape[M+1],
      ...,
      input_shape[N-1]]

4. Crop the start and end of dimensions `[1, ..., M]` of
   `reshaped_permuted` according to `crops` to produce the output of shape:
     [batch / prod(block_shape),

      input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
      ...,
      input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],

      input_shape[M+1], ..., input_shape[N-1]]

Some examples:

(1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:

```
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

The output tensor has shape `[1, 2, 2, 1]` and value:

```
x = [[[[1], [2]], [[3], [4]]]]
```

(2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:

```
[[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
```

The output tensor has shape `[1, 2, 2, 3]` and value:

```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

(3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:

```
x = [[[[1], [3]], [[9], [11]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```

The output tensor has shape `[1, 4, 4, 1]` and value:

```
x = [[[[1],   [2],  [3],  [4]],
     [[5],   [6],  [7],  [8]],
     [[9],  [10], [11],  [12]],
     [[13], [14], [15],  [16]]]]
```

(4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [2, 0]]`:

```
x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
     [[[0], [2], [4]]], [[[0], [10], [12]]],
     [[[0], [5], [7]]], [[[0], [13], [15]]],
     [[[0], [6], [8]]], [[[0], [14], [16]]]]
```

The output tensor has shape `[2, 2, 4, 1]` and value:

```
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]]],
     [[[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```}]>:$crops
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tblock_shape = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tcrops = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_BiasAddOp : TF_Op<"BiasAdd", [Pure, TF_LayoutSensitiveInterface]> {
  let summary = "Adds `bias` to `value`.";

  let description = [{
This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.
  }];

  let arguments = (ins
    Arg<TF_NumberTensor, [{Any number of dimensions.}]>:$value,
    Arg<TF_NumberTensor, [{1-D with size the last dimension of `value`.}]>:$bias,

    DefaultValuedOptionalAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_NumberTensor, [{Broadcasted sum of `value` and `bias`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    // TF_LayoutSensitiveInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
  }];

  let hasVerifier = 1;
}

def TF_BitcastOp : TF_Op<"Bitcast", [Pure]> {
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
endian orderings will give different results. A copy from input buffer to output
buffer is made on BE machines when types are of different sizes in order to get
the same casting results as on LE machines.
  }];

  let arguments = (ins
    TF_NumberTensor:$input
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr type = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_BitwiseAndOp : TF_Op<"BitwiseAnd", [Commutative, Pure, ResultsBroadcastableShape]>,
                      WithBroadcastableBinOpBuilder {
  let summary = "Elementwise computes the bitwise AND of `x` and `y`.";

  let description = [{
The result will have those bits set, that are set in both `x` and `y`. The
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
  exp = tf.constant([0, 0, 3, 10], dtype=tf.float32)

  res = bitwise_ops.bitwise_and(lhs, rhs)
  tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
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

def TF_BitwiseOrOp : TF_Op<"BitwiseOr", [Commutative, Pure, ResultsBroadcastableShape]>,
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

def TF_BitwiseXorOp : TF_Op<"BitwiseXor", [Commutative, Pure, ResultsBroadcastableShape]>,
                      WithBroadcastableBinOpBuilder {
  let summary = "Elementwise computes the bitwise XOR of `x` and `y`.";

  let description = [{
The result will have those bits set, that are different in `x` and `y`. The
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
  exp = tf.constant([5, 5, 4, 5],  dtype=tf.float32)

  res = bitwise_ops.bitwise_xor(lhs, rhs)
  tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
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

def TF_BroadcastArgsOp : TF_Op<"BroadcastArgs", [Pure]> {
  let summary = "Return the shape of s0 op s1 with broadcast.";

  let description = [{
Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$s0,
    TF_I32OrI64Tensor:$s1
  );

  let results = (outs
    TF_I32OrI64Tensor:$r0
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BroadcastToOp : TF_Op<"BroadcastTo", [Pure]> {
  let summary = "Broadcast an array for a compatible shape.";

  let description = [{
Broadcasting is the process of making arrays to have compatible shapes
for arithmetic operations. Two shapes are compatible if for each
dimension pair they are either equal or one of them is one.

For example:

>>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
>>> y = tf.broadcast_to(x, [2, 3])
>>> print(y)
tf.Tensor(
    [[1 2 3]
     [1 2 3]], shape=(2, 3), dtype=int32)

In the above example, the input Tensor with the shape of `[1, 3]`
is broadcasted to output Tensor with shape of `[2, 3]`.

When broadcasting, if a tensor has fewer axes than necessary its shape is
padded on the left with ones. So this gives the same result as the previous
example:

>>> x = tf.constant([1, 2, 3])   # Shape (3,)
>>> y = tf.broadcast_to(x, [2, 3])


When doing broadcasted operations such as multiplying a tensor
by a scalar, broadcasting (usually) confers some time or space
benefit, as the broadcasted tensor is never materialized.

However, `broadcast_to` does not carry with it any such benefits.
The newly-created tensor takes the full memory of the broadcasted
shape. (In a graph context, `broadcast_to` might be fused to
subsequent operation and then be optimized away, however.)
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A Tensor to broadcast.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{An 1-D `int` Tensor. The shape of the desired output.}]>:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{A Tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
  let hasFolder = 1;
}

def TF_BucketizeOp : TF_Op<"Bucketize", [Pure, SameOperandsAndResultShape]> {
  let summary = "Bucketizes 'input' based on 'boundaries'.";

  let description = [{
For example, if the inputs are
    boundaries = [0, 10, 100]
    input = [[-5, 10000]
             [150,   10]
             [5,    100]]

then the output will be
    output = [[0, 3]
              [3, 2]
              [1, 3]]
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{Any shape of Tensor contains with int or float type.}]>:$input,

    F32ArrayAttr:$boundaries
  );

  let results = (outs
    Res<TF_Int32Tensor, [{Same shape with 'input', each value of input replaced with bucket index.

@compatibility(numpy)
Equivalent to np.digitize.
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CastOp : TF_Op<"Cast", [Pure, SameOperandsAndResultShape]> {
  let summary = "Cast x of type SrcT to y of DstT.";

  let arguments = (ins
    TF_Tensor:$x,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$Truncate
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr SrcT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr DstT = TF_DerivedResultTypeAttr<0>;

  let hasFolder = 1;
}

def TF_CeilOp : TF_Op<"Ceil", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns element-wise smallest integer not less than x.";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_CheckNumericsOp : TF_Op<"CheckNumerics", [TF_MustExecute, TF_NoConstantFold, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Checks a tensor for NaN and Inf values.";

  let description = [{
When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
tensor.

Example usage:

``` python
a = tf.Variable(1.0)
tf.debugging.check_numerics(a, message='')

b = tf.Variable(np.nan)
try:
  tf.debugging.check_numerics(b, message='Checking b')
except Exception as e:
  assert "Checking b : Tensor had NaN values" in e.message

c = tf.Variable(np.inf)
try:
  tf.debugging.check_numerics(c, message='Checking c')
except Exception as e:
  assert "Checking c : Tensor had Inf values" in e.message
```
  }];

  let arguments = (ins
    TF_FloatTensor:$tensor,

    StrAttr:$message
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ComplexOp : TF_Op<"Complex", [Pure, ResultsBroadcastableShape]> {
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

def TF_ComplexAbsOp : TF_Op<"ComplexAbs", [Pure, SameOperandsAndResultShape]> {
  let summary = "Computes the complex absolute value of a tensor.";

  let description = [{
Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).

For example:

>>> x = tf.complex(3.0, 4.0)
>>> print((tf.raw_ops.ComplexAbs(x=x, Tout=tf.dtypes.float32, name=None)).numpy())
5.0
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

def TF_ConcatV2Op : TF_Op<"ConcatV2", [Pure]> {
  let summary = "Concatenates tensors along one dimension.";

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{List of `N` Tensors to concatenate. Their ranks and types must match,
and their sizes must match in all dimensions except `concat_dim`.}]>:$values,
    Arg<TF_I32OrI64Tensor, [{0-D.  The dimension along which to concatenate.  Must be in the
range [-rank(values), rank(values)).}]>:$axis
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` with the concatenation of values stacked along the
`concat_dim` dimension.  This tensor's shape matches that of `values` except
in `concat_dim` where it has the sum of the sizes.}]>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_Conv2DOp : TF_Op<"Conv2D", [InferTensorType, Pure, TF_LayoutSensitiveInterface]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{A 4-D tensor. The dimension order is interpreted according to the value
of `data_format`, see below for details.}]>:$input,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{A 4-D tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`}]>:$filter,

    I64ArrayAttr:$strides,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedOptionalAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{A 4-D tensor. The dimension order is determined by the value of
`data_format`, see below for details.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // TF_LayoutSensitiveInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];
}

def TF_Conv2DBackpropInputOp : TF_Op<"Conv2DBackpropInput", [Pure, TF_LayoutSensitiveInterface]> {
  let summary = [{
Computes the gradients of convolution with respect to the input.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{An integer vector representing the shape of `input`,
where `input` is a 4-D `[batch, height, width, channels]` tensor.}]>:$input_sizes,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{4-D with shape
`[filter_height, filter_width, in_channels, out_channels]`.}]>:$filter,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{4-D with shape `[batch, out_height, out_width, out_channels]`.
Gradients w.r.t. the output of the convolution.}]>:$out_backprop,

    I64ArrayAttr:$strides,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedOptionalAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32]>, [{4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
w.r.t. the input of the convolution.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // TF_LayoutSensitiveInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {2}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
  }];
}

def TF_CosOp : TF_Op<"Cos", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_CumsumOp : TF_Op<"Cumsum", [Pure, TF_AllTypesMatch<["x", "out"]>]> {
  let summary = "Compute the cumulative sum of the tensor `x` along `axis`.";

  let description = [{
By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:

```python
tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
performed instead:

```python
tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
```

By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:

```python
tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
```

This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:

```python
tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
```
  }];

  let arguments = (ins
    Arg<TF_NumberTensor, [{A `Tensor`. Must be one of the following types: `float32`, `float64`,
`int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
`complex128`, `qint8`, `quint8`, `qint32`, `half`.}]>:$x,
    Arg<TF_I32OrI64Tensor, [{A `Tensor` of type `int32` (default: 0). Must be in the range
`[-rank(x), rank(x))`.}]>:$axis,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$exclusive,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$reverse
  );

  let results = (outs
    TF_NumberTensor:$out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_DepthToSpaceOp : TF_Op<"DepthToSpace", [Pure]> {
  let summary = "DepthToSpace for tensors of type T.";

  let description = [{
Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width of the output tensor is `input_depth * block_size`, whereas the
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

    ConfinedAttr<I64Attr, [IntMinValue<2>]>:$block_size,
    DefaultValuedOptionalAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DivOp : TF_Op<"Div", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise.";

  let description = [{
*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_EluOp : TF_Op<"Elu", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the exponential linear function.";

  let description = [{
The ELU function is defined as:

 * $ e ^ x - 1 $ if $ x < 0 $
 * $ x $ if $ x >= 0 $

Examples:

>>> tf.nn.elu(1.0)
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
>>> tf.nn.elu(0.0)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
>>> tf.nn.elu(-1000.0)
<tf.Tensor: shape=(), dtype=float32, numpy=-1.0>

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
  }];

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_EqualOp : TF_Op<"Equal", [Commutative, Pure]> {
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
    TF_Tensor:$x,
    TF_Tensor:$y,

    DefaultValuedOptionalAttr<BoolAttr, "true">:$incompatible_shape_error
  );

  let results = (outs
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$x, "Value":$y,
      "BoolAttr":$incompatible_shape_error)>
  ];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_ErfOp : TF_Op<"Erf", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes the [Gauss error function](https://en.wikipedia.org/wiki/Error_function) of `x` element-wise. In statistics, for non-negative values of $x$, the error function has the following interpretation: for a random variable $Y$ that is normally distributed with mean 0 and variance $1/\sqrt{2}$, $erf(x)$ is the probability that $Y$ falls in the range $[−x, x]$.
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ErfcOp : TF_Op<"Erfc", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes the complementary error function of `x` element-wise.
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ExpOp : TF_Op<"Exp", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ExpandDimsOp : TF_Op<"ExpandDims", [Pure]> {
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
    Arg<TF_I32OrI64Tensor, [{0-D (scalar). Specifies the dimension index at which to
expand the shape of `input`. Must be in the range
`[-rank(input) - 1, rank(input)]`.}]>:$dim
  );

  let results = (outs
    Res<TF_Tensor, [{Contains the same data as `input`, but its shape has an additional
dimension of size 1 added.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tdim = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$dim)>
  ];
}

def TF_Expm1Op : TF_Op<"Expm1", [InferTensorType, Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes `exp(x) - 1` element-wise.";

  let description = [{
i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
  `e` denotes Euler's number and is approximately equal to 2.718281.

  ```python
  x = tf.constant(2.0)
  tf.math.expm1(x) ==> 6.389056

  x = tf.constant([2.0, 8.0])
  tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

  x = tf.constant(1 + 1j)
  tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
  ```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_FakeQuantWithMinMaxArgsOp : TF_Op<"FakeQuantWithMinMaxArgs", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same shape and type.
  }];

  let description = [{
Quantization is called fake since the output is still in floating point.
  The API converts inputs into values within the range [min and max] and returns
  as output.

Attributes

*   `[min; max]` define the clamping range for the `inputs` data.
*   `inputs` values are quantized into the quantization range (
`[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
when it is true) and then de-quantized and output as floats in `[min; max]`
interval.
*   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:

*   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
*   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
*   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.


Examples

```python

inp = tf.constant ([10.03, -10.23, 3])
out = tf.quantization.fake_quant_with_min_max_args(inp, min=-5, max=5,
                                                   num_bits=16)
print(out)

#  Output:
#  tf.Tensor([ 4.9999237 -5.0000763  3.0000763], shape=(3,), dtype=float32)
```

Raises:
  * InvalidArgumentError:
    - If num_bits are outside of range [2, 16].
    - If min >= max.
  * ValueError: If `inputs` are of any other type than float32.
  }];

  let arguments = (ins
    TF_Float32Tensor:$inputs,

    DefaultValuedOptionalAttr<F32Attr, "-6.0f">:$min,
    DefaultValuedOptionalAttr<F32Attr, "6.0f">:$max,
    DefaultValuedOptionalAttr<I64Attr, "8">:$num_bits,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_Float32Tensor:$outputs
  );

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_FakeQuantWithMinMaxVarsOp : TF_Op<"FakeQuantWithMinMaxVars", [Pure]> {
  let summary = [{
Fake-quantize the 'inputs' tensor of type float via global float scalars
  }];

  let description = [{
Fake-quantize the `inputs` tensor of type float via global float scalars
`min` and `max` to `outputs` tensor of same shape as `inputs`.

Attributes

*   `[min; max]` define the clamping range for the `inputs` data.
*   `inputs` values are quantized into the quantization range (
`[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
when it is true) and then de-quantized and output as floats in `[min; max]`
interval.
*   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:

*   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
*   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
*   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

This operation has a gradient and thus allows for training `min` and `max`
values.

>>> constant_input = tf.constant([[1.2, -0.3, 0.7], [2.1, 0.5, -1.0]], dtype=tf.float32)
>>>
>>> min_val = -0.5
>>> max_val = 0.8
>>> num_bits = 8
>>> narrow_range = False #False:for the quantization range [0; 2^num_bits - 1]
>>>
>>> quantized_data = tf.quantization.fake_quant_with_min_max_vars(
...   inputs=constant_input, min=min_val, max=max_val, num_bits=num_bits, narrow_range=narrow_range
... )
>>>
>>> print("Input:\n", constant_input.numpy())
Input:
[[ 1.2 -0.3  0.7]
[ 2.1  0.5 -1. ]]
>>> print("Output:\n", quantized_data.numpy())
Output:
[[ 0.8003921 -0.3007843  0.6984313]
[ 0.8003921  0.4996078 -0.4996078]]
  }];

  let arguments = (ins
    TF_Float32Tensor:$inputs,
    TF_Float32Tensor:$min,
    TF_Float32Tensor:$max,

    DefaultValuedOptionalAttr<I64Attr, "8">:$num_bits,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_Float32Tensor:$outputs
  );

  let hasVerifier = 1;
}

def TF_FillOp : TF_Op<"Fill", [Pure]> {
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
    Arg<TF_I32OrI64Tensor, [{1-D. Represents the shape of the output tensor.}]>:$dims,
    Arg<TF_Tensor, [{0-D (scalar). Value to fill the returned tensor.

@compatibility(numpy)
Equivalent to np.full
@end_compatibility}]>:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr index_type = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$dims, "Value":$value)>
  ];
}


def TF_FFTOp : TF_Op<"FFT", [Pure]> {
  let summary = "Fast Fourier transform.";

  let description = [{
Computes the 1-dimensional discrete Fourier transform over the inner-most
dimension of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fft
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_FloorOp : TF_Op<"Floor", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns element-wise largest integer not greater than x.";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_FloorDivOp : TF_Op<"FloorDiv", [Pure, ResultsBroadcastableShape]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns x // y element-wise.";

  let description = [{
*NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FloorModOp : TF_Op<"FloorMod", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns element-wise remainder of division.";

  let description = [{
This follows Python semantics in that the
result here is consistent with a flooring divide. E.g.
`floor(x / y) * y + floormod(x, y) = x`, regardless of the signs of x and y.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    TF_IntOrFpTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_GatherOp : TF_Op<"Gather", [Pure]> {
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

    DefaultValuedOptionalAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_GatherNdOp : TF_Op<"GatherNd", [Pure]> {
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

If `indices` contains any out-of-bound indices, depending on
`bad_indices_policy`, the op will either return an error or ignore the
out-of-bound indices. `bad_indices_policy` can be one of the following values:
1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
   historically on CPU and GPU we handle errors in different ways, and for
   backward compatibility we keep the default behavior.
2. "ERROR": raises error; GPU does not support this value.
3. "IGNORE": ignore error and set the corresponding output to 0;
   supported on both CPU and GPU.

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
    Arg<TF_Tensor, [{The tensor from which to gather values.}]>:$params,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Index tensor.}]>:$indices,

    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$bad_indices_policy
  );

  let results = (outs
    Res<TF_Tensor, [{Values from `params` gathered from indices given by `indices`, with
shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;
}

def TF_GatherV2Op : TF_Op<"GatherV2", [Pure]> {
  let summary = [{
Gather slices from `params` axis `axis` according to `indices`.
  }];

  let description = [{
`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `params.shape[:axis] +
indices.shape[batch_dims:] + params.shape[axis + 1:]` where:

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

Note that on TPU, if any dimension of `params` is of size 0 then the output will
be the expected shape filled with zeros. On CPU and GPU an error will be
returned.

See also `tf.batch_gather` and `tf.gather_nd`.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor from which to gather values. Must be at least rank
`axis + 1`.}]>:$params,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Index tensor. Must be in range `[0, params.shape[axis])`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{The axis in `params` to gather `indices` from. Defaults to the first
dimension. Supports negative indexes.}]>:$axis,

    DefaultValuedOptionalAttr<I64Attr, "0">:$batch_dims
  );

  let results = (outs
    Res<TF_Tensor, [{Values from `params` gathered from indices given by `indices`, with
shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr Taxis = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tparams = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_GreaterOp : TF_Op<"Greater", [Pure, ResultsBroadcastableShape]>,
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
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_GreaterEqualOp : TF_Op<"GreaterEqual", [Pure, ResultsBroadcastableShape]>,
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
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IdentityOp : TF_Op<"Identity", [Pure, TF_NoConstantFold, TF_OperandsSameAsResultsTypeOrRef]> {
  let summary = [{
Return a tensor with the same shape and contents as the input tensor or value.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IdentityNOp : TF_Op<"IdentityN", [Pure]> {
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

def TF_ImagOp : TF_Op<"Imag", [Pure, SameOperandsAndResultShape]> {
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

def TF_IsFiniteOp : TF_Op<"IsFinite", [Pure, SameOperandsAndResultShape]> {
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
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_BoolTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LRNOp : TF_Op<"LRN", [Pure]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{4-D.}]>:$input,

    DefaultValuedOptionalAttr<I64Attr, "5">:$depth_radius,
    DefaultValuedOptionalAttr<F32Attr, "1.0f">:$bias,
    DefaultValuedOptionalAttr<F32Attr, "1.0f">:$alpha,
    DefaultValuedOptionalAttr<F32Attr, "0.5f">:$beta
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LeakyReluOp : TF_Op<"LeakyRelu", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear: `max(features, features * alpha)`.";

  let arguments = (ins
    TF_FloatTensor:$features,

    DefaultValuedOptionalAttr<F32Attr, "0.2f">:$alpha
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_LeftShiftOp : TF_Op<"LeftShift", [Pure, ResultsBroadcastableShape]>,
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

def TF_LessOp : TF_Op<"Less", [Pure, ResultsBroadcastableShape]>,
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
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LessEqualOp : TF_Op<"LessEqual", [Pure, ResultsBroadcastableShape]>,
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
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LogOp : TF_Op<"Log", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_Log1pOp : TF_Op<"Log1p", [Pure, TF_CwiseUnary, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes natural logarithm of (1 + x) element-wise.";

  let description = [{
I.e., \\(y = \log_e (1 + x)\\).

Example:

```python
x = tf.constant([0, 0.5, 1, 5])
tf.math.log1p(x) ==> [0., 0.4054651, 0.6931472, 1.7917595]
```
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_LogSoftmaxOp : TF_Op<"LogSoftmax", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes log softmax activations.";

  let description = [{
For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{2-D with shape `[batch_size, num_classes]`.}]>:$logits
  );

  let results = (outs
    Res<TF_FloatTensor, [{Same shape as `logits`.}]>:$logsoftmax
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_LogicalAndOp : TF_Op<"LogicalAnd", [Commutative, Pure, ResultsBroadcastableShape]>,
                      WithBroadcastableBinOpBuilder {
  let summary = "Returns the truth value of x AND y element-wise.";

  let description = [{
*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_BoolTensor:$x,
    TF_BoolTensor:$y
  );

  let results = (outs
    TF_BoolTensor:$z
  );

  let hasFolder = 1;
}

def TF_LogicalNotOp : TF_Op<"LogicalNot", [Pure, TF_Involution, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns the truth value of `NOT x` element-wise.";

  let arguments = (ins
    Arg<TF_BoolTensor, [{A `Tensor` of type `bool`.}]>:$x
  );

  let results = (outs
    Res<TF_BoolTensor, [{A `Tensor` of type `bool` with the same shape as `x`. The logical negation of `x`.}]>:$y
  );

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_LogicalOrOp : TF_Op<"LogicalOr", [Commutative, Pure, ResultsBroadcastableShape]>,
                     WithBroadcastableBinOpBuilder {
  let summary = "Returns the truth value of x OR y element-wise.";

  let description = [{
*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_BoolTensor:$x,
    TF_BoolTensor:$y
  );

  let results = (outs
    TF_BoolTensor:$z
  );
}


def TF_MatMulOp : TF_Op<"MatMul", [Pure, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$a,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$b,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$grad_b
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixDiagOp : TF_Op<"MatrixDiag", [Pure]> {
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
    Arg<TF_Tensor, [{Rank `k`, where `k >= 1`.}]>:$diagonal
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MatrixSetDiagOp : TF_Op<"MatrixSetDiag", [Pure]> {
  let summary = [{
Returns a batched matrix tensor with new batched diagonal values.
  }];

  let description = [{
Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.

The output is computed as follows:

Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:

  * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
  * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Rank `k+1`, where `k >= 1`.}]>:$input,
    Arg<TF_Tensor, [{Rank `k`, where `k >= 1`.}]>:$diagonal
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `k+1`, with `output.shape = input.shape`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MatrixSetDiagV2Op : TF_Op<"MatrixSetDiagV2", [Pure]> {
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
where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

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
tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
                                   [7, 2, 7, 7],
                                   [7, 7, 3, 7]],
                                  [[4, 7, 7, 7],
                                   [7, 5, 7, 7],
                                   [7, 7, 6, 7]]]

# A superdiagonal (per batch).
tf.matrix_set_diag(diagonal, k = 1)
  ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
        [7, 7, 2, 7],
        [7, 7, 7, 3]],
       [[7, 4, 7, 7],
        [7, 7, 5, 7],
        [7, 7, 7, 6]]]

# A band of diagonals.
diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
                       [4, 5, 0]],
                      [[6, 1, 2],
                       [3, 4, 0]]])
tf.matrix_set_diag(diagonals, k = (-1, 0))
  ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
        [4, 2, 7, 7],
        [0, 5, 3, 7]],
       [[6, 7, 7, 7],
        [3, 1, 7, 7],
        [7, 4, 2, 7]]]

```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Rank `r+1`, where `r >= 1`.}]>:$input,
    Arg<TF_Tensor, [{Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
`k >= 1`.}]>:$diagonal,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `r+1`, with `output.shape = input.shape`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MatrixSetDiagV3Op : TF_Op<"MatrixSetDiagV3", [Pure]> {
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
    Arg<TF_Tensor, [{Rank `r+1`, where `r >= 1`.}]>:$input,
    Arg<TF_Tensor, [{Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
`k >= 1`.}]>:$diagonal,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k,

    DefaultValuedOptionalAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "\"RIGHT_LEFT\"">:$align
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `r+1`, with `output.shape = input.shape`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxOp : TF_Op<"Max", [Pure]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$reduction_indices,
      "BoolAttr":$keep_dims)>
  ];
}

def TF_MaxPoolOp : TF_Op<"MaxPool", [Pure, TF_FoldOperandsTransposeInterface, TF_LayoutSensitiveInterface]> {
  let summary = "Performs max pooling on the input.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint8]>, [{4-D input to pool over.}]>:$input,

    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedOptionalAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint8]>, [{The max pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    // TF_FoldOperandsTransposeInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
    // TF_LayoutSensitiveInterface:
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
  }];
}

def TF_MaximumOp : TF_Op<"Maximum", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns the max of x and y (i.e. x > y ? x : y) element-wise.";

  let description = [{
*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    TF_IntOrFpTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MeanOp : TF_Op<"Mean", [Pure, TF_FoldOperandsTransposeInterface]> {
  let summary = "Computes the mean of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    Arg<TF_NumberTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_NumberTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let extraClassDeclaration = [{
    // TF_FoldOperandsTransposeInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {}; }
    LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
  }];
}

def TF_MinOp : TF_Op<"Min", [Pure]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_MinimumOp : TF_Op<"Minimum", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns the min of x and y (i.e. x < y ? x : y) element-wise.";

  let description = [{
*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    TF_IntOrFpTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MirrorPadOp : TF_Op<"MirrorPad", [Pure, TF_OperandHasRank<1, 2>]> {
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
    Arg<TF_Tensor, [{The input tensor to be padded.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{A two-column matrix specifying the padding sizes. The number of
rows must be the same as the rank of `input`.}]>:$paddings,

    TF_AnyStrAttrOf<["REFLECT", "SYMMETRIC"]>:$mode
  );

  let results = (outs
    Res<TF_Tensor, [{The padded tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
}

def TF_ModOp : TF_Op<"Mod", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = [{
Returns element-wise remainder of division. This emulates C semantics in that
  }];

  let description = [{
the result here is consistent with a truncating divide. E.g.
`tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.

*NOTE*: `Mod` supports broadcasting. More about broadcasting
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

def TF_MulOp : TF_Op<"Mul", [Commutative, Pure, ResultsBroadcastableShape, TF_CwiseBinary, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x * y element-wise.";

  let description = [{
*NOTE*: `Multiply` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_MultinomialOp : TF_Op<"Multinomial", [TF_CannotDuplicate]> {
  let summary = "Draws samples from a multinomial distribution.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
represents the unnormalized log probabilities for all classes.}]>:$logits,
    Arg<TF_Int32Tensor, [{0-D.  Number of independent samples to draw for each row slice.}]>:$num_samples,

    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
contains the drawn class labels with range `[0, num_classes)`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr output_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_NegOp : TF_Op<"Neg", [Pure, TF_CwiseUnary, TF_Involution, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes numerical negative value element-wise.";

  let description = [{
I.e., \\(y = -x\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_NonMaxSuppressionV4Op : TF_Op<"NonMaxSuppressionV4", [Pure]> {
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
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 2-D float tensor of shape `[num_boxes, 4]`.}]>:$boxes,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 1-D float tensor of shape `[num_boxes]` representing a single
score corresponding to each box (each row of boxes).}]>:$scores,
    Arg<TF_Int32Tensor, [{A scalar integer tensor representing the maximum number of
boxes to be selected by non max suppression.}]>:$max_output_size,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding whether
boxes overlap too much with respect to IOU.}]>:$iou_threshold,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding when to remove
boxes based on score.}]>:$score_threshold,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$pad_to_max_output_size
  );

  let results = (outs
    Res<TF_Int32Tensor, [{A 1-D integer tensor of shape `[M]` representing the selected
indices from the boxes tensor, where `M <= max_output_size`.}]>:$selected_indices,
    Res<TF_Int32Tensor, [{A 0-D integer tensor representing the number of valid elements in
`selected_indices`, with the valid elements appearing first.}]>:$valid_outputs
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T_threshold = TF_DerivedOperandTypeAttr<3>;
}

def TF_NonMaxSuppressionV5Op : TF_Op<"NonMaxSuppressionV5", [Pure]> {
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
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 2-D float tensor of shape `[num_boxes, 4]`.}]>:$boxes,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 1-D float tensor of shape `[num_boxes]` representing a single
score corresponding to each box (each row of boxes).}]>:$scores,
    Arg<TF_Int32Tensor, [{A scalar integer tensor representing the maximum number of
boxes to be selected by non max suppression.}]>:$max_output_size,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding whether
boxes overlap too much with respect to IOU.}]>:$iou_threshold,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding when to remove
boxes based on score.}]>:$score_threshold,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the sigma parameter for Soft NMS; see Bodla et
al (c.f. https://arxiv.org/abs/1704.04503).  When `soft_nms_sigma=0.0` (which
is default), we fall back to standard (hard) NMS.}]>:$soft_nms_sigma,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$pad_to_max_output_size
  );

  let results = (outs
    Res<TF_Int32Tensor, [{A 1-D integer tensor of shape `[M]` representing the selected
indices from the boxes tensor, where `M <= max_output_size`.}]>:$selected_indices,
    Res<TensorOf<[TF_Float16, TF_Float32]>, [{A 1-D float tensor of shape `[M]` representing the corresponding
scores for each selected box, where `M <= max_output_size`.  Scores only differ
from corresponding input scores when using Soft NMS (i.e. when
`soft_nms_sigma>0`)}]>:$selected_scores,
    Res<TF_Int32Tensor, [{A 0-D integer tensor representing the number of valid elements in
`selected_indices`, with the valid elements appearing first.}]>:$valid_outputs
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NotEqualOp : TF_Op<"NotEqual", [Commutative, Pure]> {
  let summary = "Returns the truth value of (x != y) element-wise.";

  let description = [{
*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_Tensor:$x,
    TF_Tensor:$y,

    DefaultValuedOptionalAttr<BoolAttr, "true">:$incompatible_shape_error
  );

  let results = (outs
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$x, "Value":$y,
      "BoolAttr":$incompatible_shape_error)>
  ];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_OneHotOp : TF_Op<"OneHot", [Pure]> {
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
    Arg<TensorOf<[TF_Int32, TF_Int64, TF_Int8, TF_Uint8]>, [{A tensor of indices.}]>:$indices,
    Arg<TF_Int32Tensor, [{A scalar defining the depth of the one hot dimension.}]>:$depth,
    Arg<TF_Tensor, [{A scalar defining the value to fill in output when `indices[j] = i`.}]>:$on_value,
    Arg<TF_Tensor, [{A scalar defining the value to fill in output when `indices[j] != i`.}]>:$off_value,

    DefaultValuedOptionalAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    Res<TF_Tensor, [{The one-hot tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr TI = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$indices, "Value":$depth, "Value":$on_value,
      "Value":$off_value, "IntegerAttr":$axis)>
  ];

  let hasVerifier = 1;
}

def TF_PadOp : TF_Op<"Pad", [Pure, TF_FoldOperandsTransposeInterface, TF_OperandHasRank<1, 2>]> {
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

  let extraClassDeclaration = [{
    // TF_FoldOperandsTransposeInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
  }];
}

def TF_PadV2Op : TF_Op<"PadV2", [Pure, TF_OperandHasRank<1, 2>]> {
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

def TF_PowOp : TF_Op<"Pow", [Pure, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_PreventGradientOp : TF_Op<"PreventGradient", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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
    Arg<TF_Tensor, [{any tensor.}]>:$input,

    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$message
  );

  let results = (outs
    Res<TF_Tensor, [{the same input tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ProdOp : TF_Op<"Prod", [Pure]> {
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
    Arg<TF_NumberTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_NumberTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_QuantizeAndDequantizeV4Op : TF_Op<"QuantizeAndDequantizeV4", [Pure]> {
  let summary = "Quantizes then dequantizes a tensor.";

  let description = [{
This is almost identical to QuantizeAndDequantizeV2, except that it returns a
gradient of 1 for inputs that are within the quantization range, or 0 otherwise.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{Tensor to quantize and then dequantize.}]>:$input,
    Arg<TF_FloatTensor, [{If `range_given == True`, this specifies the minimum input value that needs to
be represented, otherwise it is determined from the min value of the `input`
tensor.}]>:$input_min,
    Arg<TF_FloatTensor, [{If `range_given == True`, this specifies the maximum input value that needs to
be represented, otherwise it is determined from the max value of the `input`
tensor.}]>:$input_max,

    DefaultValuedOptionalAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedOptionalAttr<I64Attr, "8">:$num_bits,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$range_given,
    DefaultValuedOptionalAttr<TF_AnyStrAttrOf<["HALF_TO_EVEN", "HALF_UP"]>, "\"HALF_TO_EVEN\"">:$round_mode,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RFFT2DOp : TF_Op<"RFFT2D", [Pure]> {
  let summary = "2D real-valued fast Fourier transform.";

  let description = [{
Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    Arg<TF_F32OrF64Tensor, [{A float32 tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [2]. The FFT length for each dimension.}]>:$fft_length
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex64 tensor of the same rank as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier transform. The
  inner-most dimension contains `fft_length / 2 + 1` unique frequency
  components.

@compatibility(numpy)
Equivalent to np.fft.rfft2
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Treal = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tcomplex = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomStandardNormalOp : TF_Op<"RandomStandardNormal", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Outputs random values from a normal distribution.";

  let description = [{
The generated values will have mean 0 and standard deviation 1.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,

    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with random normal values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomUniformOp : TF_Op<"RandomUniform", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,

    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with uniform random values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_RangeOp : TF_Op<"Range", [Pure, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64]>, [{0-D (scalar). First entry in the sequence.}]>:$start,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64]>, [{0-D (scalar). Upper limit of sequence, exclusive.}]>:$limit,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64]>, [{0-D (scalar). Optional. Default is 1. Number that increments `start`.}]>:$delta
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64]>, [{1-D.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$start, "Value":$limit, "Value":$delta)>
  ];

  let hasFolder = 1;

}

def TF_RankOp : TF_Op<"Rank", [Pure]> {
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
    TF_Int32Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasFolder = 1;
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
    Arg<TF_ResourceTensor, [{handle to the resource in which to store the variable.}], [TF_VariableRead]>:$resource
  );

  let results = (outs
    TF_Tensor:$value
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_RealOp : TF_Op<"Real", [Pure, SameOperandsAndResultShape]> {
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

def TF_RealDivOp : TF_Op<"RealDiv", [Pure, ResultsBroadcastableShape, TF_CwiseBinary]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise for real types.";

  let description = [{
If `x` and `y` are reals, this will return the floating-point division.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_ReluOp : TF_Op<"Relu", [Pure, TF_Idempotent, TF_LayoutAgnostic, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear: `max(features, 0)`.";

  let description = [{
See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
Example usage:
>>> tf.nn.relu([-2., 0., 3.]).numpy()
array([0., 0., 3.], dtype=float32)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$features
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_Relu6Op : TF_Op<"Relu6", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear 6: `min(max(features, 0), 6)`.";

  let arguments = (ins
    TF_IntOrFpTensor:$features
  );

  let results = (outs
    TF_IntOrFpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_ReshapeOp : TF_Op<"Reshape", [Pure]> {
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
    Arg<TF_I32OrI64Tensor, [{Defines the shape of the output tensor.}]>:$shape
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$tensor, "Value":$shape)>
  ];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def TF_ResizeBilinearOp : TF_Op<"ResizeBilinear", [Pure]> {
  let summary = "Resize `images` to `size` using bilinear interpolation.";

  let description = [{
Input images can be of different types but output images are always float.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$images,
    Arg<TF_Int32Tensor, [{= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
new size for the images.}]>:$size,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TF_Float32Tensor, [{4-D with shape
`[batch, new_height, new_width, channels]`.}]>:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResizeNearestNeighborOp : TF_Op<"ResizeNearestNeighbor", [Pure]> {
  let summary = [{
Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$images,
    Arg<TF_Int32Tensor, [{= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
new size for the images.}]>:$size,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape
`[batch, new_height, new_width, channels]`.}]>:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReverseSequenceOp : TF_Op<"ReverseSequence", [Pure]> {
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
    Arg<TF_Tensor, [{The input to reverse.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D with length `input.dims(batch_dim)` and
`max(seq_lengths) <= input.dims(seq_dim)`}]>:$seq_lengths,

    I64Attr:$seq_dim,
    DefaultValuedOptionalAttr<I64Attr, "0">:$batch_dim
  );

  let results = (outs
    Res<TF_Tensor, [{The partially reversed input. It has the same shape as `input`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
}

def TF_ReverseV2Op : TF_Op<"ReverseV2", [Pure]> {
  let summary = "Reverses specific dimensions of a tensor.";

  let description = [{
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
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Str, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Up to 8-D.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{1-D. The indices of the dimensions to reverse. Must be in the range
`[-rank(tensor), rank(tensor))`.}]>:$axis
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Str, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The same shape as `tensor`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_RFFTOp : TF_Op<"RFFT", [Pure]> {
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
    Arg<TF_F32OrF64Tensor, [{A float32 tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [1]. The FFT length.}]>:$fft_length
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex64 tensor of the same rank as `input`. The inner-most
  dimension of `input` is replaced with the `fft_length / 2 + 1` unique
  frequency components of its 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.rfft
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Treal = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tcomplex = TF_DerivedResultTypeAttr<0>;
}


def TF_RightShiftOp : TF_Op<"RightShift", [Pure, ResultsBroadcastableShape]>,
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

def TF_RoundOp : TF_Op<"Round", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Rounds the values of a tensor to the nearest integer, element-wise.
  }];

  let description = [{
Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_RsqrtOp : TF_Op<"Rsqrt", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}


def TF_ScatterNdOp : TF_Op<"ScatterNd", [Pure]> {
  let summary = [{
Scatters `updates` into a tensor of shape `shape` according to `indices`.
  }];

  let description = [{
Scatter sparse `updates` according to individual values at the specified
`indices`. This op returns an output tensor with the `shape` you specify. This
op is the inverse of the `tf.gather_nd` operator which extracts values or slices
from a given tensor.

This operation is similar to `tf.tensor_scatter_nd_add`, except that the tensor
is zero-initialized. Calling `tf.scatter_nd(indices, updates, shape)`
is identical to calling
`tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`

If `indices` contains duplicates, the associated `updates` are accumulated
(summed) into the output tensor.

**WARNING**: For floating-point data types, the output may be nondeterministic.
This is because the order in which the updates are applied is nondeterministic
and when floating-point numbers are added in different orders the resulting
numerical approximation error can be slightly different. However, the output
will be deterministic if op determinism is enabled via
`tf.config.experimental.enable_op_determinism`.

`indices` is an integer tensor containing indices into the output tensor. The
last dimension of `indices` can be at most the rank of `shape`:

    indices.shape[-1] <= shape.rank

The last dimension of `indices` corresponds to indices of elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.

`updates` is a tensor with shape:

    indices.shape[:-1] + shape[indices.shape[-1]:]

The simplest form of the scatter op is to insert individual elements in
a tensor by index. Consider an example where you want to insert 4 scattered
elements in a rank-1 tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

You can also insert entire slices of a higher rank tensor all at once. For
example, you can insert two slices in the first dimension of a rank-3 tensor
with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

```python
    indices = tf.constant([[1], [3]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]

If `indices` contains any out-of-bound indices, depending on
`bad_indices_policy`, the op will either return an error or ignore the
out-of-bound indices. `bad_indices_policy` can be one of the following values:
1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
   historically on CPU and GPU we handle errors in different ways, and for
   backward compatibility we keep the default behavior.
2. "ERROR": raises error; GPU does not support this value.
3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Tensor of indices.}]>:$indices,
    Arg<TF_Tensor, [{Values to scatter into the output tensor.}]>:$updates,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{1-D. The shape of the output tensor.}]>:$shape,

    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$bad_indices_policy
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor with the given shape and updates applied according
to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
}

def TF_SegmentSumOp : TF_Op<"SegmentSum", [Pure]> {
  let summary = "Computes the sum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as the same as a smaller following index.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
</div>

For example:

>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> tf.math.segment_sum(c, tf.constant([0, 0, 1])).numpy()
array([[5, 5, 5, 5],
       [5, 6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TF_NumberTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_NumberTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_ShapeOp : TF_Op<"Shape", [Pure]> {
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

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "BoolAttr":$use32Bit)>
  ];

  let hasFolder = 1;
}

def TF_SigmoidOp : TF_Op<"Sigmoid", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SigmoidGradOp : TF_Op<"SigmoidGrad", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the gradient of the sigmoid of `x` wrt its input.";

  let description = [{
Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient.
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$y,
    TF_FpOrComplexTensor:$dy
  );

  let results = (outs
    TF_FpOrComplexTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SignOp : TF_Op<"Sign", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns an element-wise indication of the sign of a number.";

  let description = [{
`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

Example usage:
>>> tf.math.sign([0., 2., -3.])
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SinOp : TF_Op<"Sin", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SliceOp : TF_Op<"Slice", [PredOpTrait<"input and output must have same element type", TCresVTEtIsSameAsOp<0, 0>>, Pure]> {
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
    Arg<TF_I32OrI64Tensor, [{begin[i] specifies the offset into the 'i'th dimension of
'input' to slice from.}]>:$begin,
    Arg<TF_I32OrI64Tensor, [{size[i] specifies the number of elements of the 'i'th dimension
of 'input' to slice. If size[i] is -1, all remaining elements in dimension
i are included in the slice (i.e. this is equivalent to setting
size[i] = input.dim_size(i) - begin[i]).}]>:$size
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_SnapshotOp : TF_Op<"Snapshot", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns a copy of the input tensor.";

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SoftmaxOp : TF_Op<"Softmax", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes softmax activations.";

  let description = [{
For each batch `i` and class `j` we have

    $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{2-D with shape `[batch_size, num_classes]`.}]>:$logits
  );

  let results = (outs
    Res<TF_FloatTensor, [{Same shape as `logits`.}]>:$softmax
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SoftplusOp : TF_Op<"Softplus", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "";

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SoftsignOp : TF_Op<"Softsign", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes softsign: `features / (abs(features) + 1)`.";

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SpaceToBatchNDOp : TF_Op<"SpaceToBatchND", [Pure]> {
  let summary = "SpaceToBatch for N-D tensors of type T.";

  let description = [{
This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
"batch" dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`. See below for a
precise description.

This operation is equivalent to the following steps:

1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
   input according to `paddings` to produce `padded` of shape `padded_shape`.

2. Reshape `padded` to `reshaped_padded` of shape:

     [batch] +
     [padded_shape[1] / block_shape[0],
       block_shape[0],
      ...,
      padded_shape[M] / block_shape[M-1],
      block_shape[M-1]] +
     remaining_shape

3. Permute dimensions of `reshaped_padded` to produce
   `permuted_reshaped_padded` of shape:

     block_shape +
     [batch] +
     [padded_shape[1] / block_shape[0],
      ...,
      padded_shape[M] / block_shape[M-1]] +
     remaining_shape

4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
   dimension, producing an output tensor of shape:

     [batch * prod(block_shape)] +
     [padded_shape[1] / block_shape[0],
      ...,
      padded_shape[M] / block_shape[M-1]] +
     remaining_shape

Some examples:

(1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:

```
x = [[[[1], [2]], [[3], [4]]]]
```

The output tensor has shape `[4, 1, 1, 1]` and value:

```
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

(2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:

```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

The output tensor has shape `[4, 1, 1, 3]` and value:

```
[[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
```

(3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:

```
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]],
      [[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```

The output tensor has shape `[4, 2, 2, 1]` and value:

```
x = [[[[1], [3]], [[9], [11]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```

(4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
    paddings = `[[0, 0], [2, 0]]`:

```
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]]],
     [[[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```

The output tensor has shape `[8, 1, 3, 1]` and value:

```
x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
     [[[0], [2], [4]]], [[[0], [10], [12]]],
     [[[0], [5], [7]]], [[[0], [13], [15]]],
     [[[0], [6], [8]]], [[[0], [14], [16]]]]
```

Among others, this operation is useful for reducing atrous convolution into
regular convolution.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
where spatial_shape has `M` dimensions.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D with shape `[M]`, all values must be >= 1.}]>:$block_shape,
    Arg<TF_I32OrI64Tensor, [{2-D with shape `[M, 2]`, all values must be >= 0.
  `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
  `i + 1`, which corresponds to spatial dimension `i`.  It is required that
  `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.}]>:$paddings
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tblock_shape = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];
}

def TF_SpaceToDepthOp : TF_Op<"SpaceToDepth", [Pure]> {
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

    ConfinedAttr<I64Attr, [IntMinValue<2>]>:$block_size,
    DefaultValuedOptionalAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SparseToDenseOp : TF_Op<"SparseToDense", [Pure]> {
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
    Arg<TF_I32OrI64Tensor, [{0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
index where `sparse_values[i]` will be placed.}]>:$sparse_indices,
    Arg<TF_I32OrI64Tensor, [{1-D.  Shape of the dense output tensor.}]>:$output_shape,
    Arg<TF_Tensor, [{1-D.  Values corresponding to each row of `sparse_indices`,
or a scalar value to be used for all sparse indices.}]>:$sparse_values,
    Arg<TF_Tensor, [{Scalar value to set for indices not specified in
`sparse_indices`.}]>:$default_value,

    DefaultValuedOptionalAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    Res<TF_Tensor, [{Dense output tensor of shape `output_shape`.}]>:$dense
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
}

def TF_SqrtOp : TF_Op<"Sqrt", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
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

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SquareOp : TF_Op<"Square", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes square of x element-wise.";

  let description = [{
I.e., \\(y = x * x = x^2\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_SquaredDifferenceOp : TF_Op<"SquaredDifference", [Commutative, Pure, ResultsBroadcastableShape]>,
                             WithBroadcastableBinOpBuilder {
  let summary = "Returns conj(x - y)(x - y) element-wise.";

  let description = [{
*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SqueezeOp : TF_Op<"Squeeze", [Pure]> {
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
    Arg<TF_Tensor, [{The `input` to squeeze.}]>:$input,

    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$squeeze_dims
  );

  let results = (outs
    Res<TF_Tensor, [{Contains the same data as `input`, but has one or more dimensions of
size 1 removed.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_StopGradientOp : TF_Op<"StopGradient", [Pure, TF_AllTypesMatch<["input", "output"]>]> {
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
to pretend that the value was a constant. For example, the softmax function
for a vector x can be written as

```python

  def softmax(x):
    numerator = tf.exp(x)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

This however is susceptible to overflow if the values in x are large. An
alternative more stable way is to subtract the maximum of x from each of the
values.

```python

  def stable_softmax(x):
    z = x - tf.reduce_max(x)
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

However, when we backprop through the softmax to x, we dont want to backprop
through the `tf.reduce_max(x)` (if the max values are not unique then the
gradient could flow to the wrong input) calculation and treat that as a
constant. Therefore, we should write this out as

```python

  def stable_softmax(x):
    z = x - tf.stop_gradient(tf.reduce_max(x))
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

Some other examples include:

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

def TF_StridedSliceOp : TF_Op<"StridedSlice", [Pure]> {
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
begin_mask = 1<<4 | 1<<5 = 48
end_mask = 1<<5 = 32
ellipsis_mask = 1<<3 = 8
new_axis_mask = 1<<2 = 4
shrink_axis_mask = 1<<0 = 1
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
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{`begin[k]` specifies the offset into the `k`th range specification.
The exact dimension this corresponds to will be determined by context.
Out-of-bounds values will be silently clamped. If the `k`th bit of
`begin_mask` then `begin[k]` is ignored and the full range of the
appropriate dimension is used instead. Negative values causes indexing
to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.}]>:$begin,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{`end[i]` is like `begin` with the exception that `end_mask` is
used to determine full ranges.}]>:$end,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{`strides[i]` specifies the increment in the `i`th specification
after extracting a given element. Negative indices will reverse
the original order. Out or range values are
clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`}]>:$strides,

    DefaultValuedOptionalAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedOptionalAttr<I64Attr, "0">:$end_mask,
    DefaultValuedOptionalAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedOptionalAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedOptionalAttr<I64Attr, "0">:$shrink_axis_mask
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // If sliced shape is able to be deduced, returns true, updates
    // `begin_indices`, `end_indices`, and `strides` with their canonical
    // values, respectively.
    bool GetSlicedBoundRanges(
      ::llvm::SmallVectorImpl<int64_t> *slice_begin,
      ::llvm::SmallVectorImpl<int64_t> *slice_end,
      ::llvm::SmallVectorImpl<int64_t> *slice_stride);
  }];
}

def TF_SubOp : TF_Op<"Sub", [Pure, ResultsBroadcastableShape, TF_CwiseBinary, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x - y element-wise.";

  let description = [{
*NOTE*: `Subtract` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_SumOp : TF_Op<"Sum", [Pure]> {
  let summary = "Computes the sum of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    Arg<TF_NumberTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_NumberTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$reduction_indices,
      "BoolAttr":$keep_dims)>
  ];

  let hasFolder = 1;
}

def TF_TanhOp : TF_Op<"Tanh", [Pure, TF_LayoutAgnostic, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes hyperbolic tangent of `x` element-wise.";

  let description = [{
Given an input tensor, this function computes hyperbolic tangent of every
  element in the tensor. Input range is `[-inf, inf]` and
  output range is `[-1,1]`.

  >>> x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
  >>> tf.math.tanh(x)
  <tf.Tensor: shape=(8,), dtype=float32, numpy=
  array([-1.0, -0.99990916, -0.46211717,  0.7615942 ,  0.8336547 ,
          0.9640276 ,  0.9950547 ,  1.0], dtype=float32)>
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_TensorListElementShapeOp : TF_Op<"TensorListElementShape", [Pure]> {
  let summary = "The shape of the elements of the given list, as a tensor.";

  let description = [{
input_handle: the list
  element_shape: the shape of elements of the list
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle
  );

  let results = (outs
    TF_I32OrI64Tensor:$element_shape
  );

  TF_DerivedResultTypeAttr shape_type = TF_DerivedResultTypeAttr<0>;

  let hasFolder = 1;
}

def TF_TensorListFromTensorOp : TF_Op<"TensorListFromTensor", [Pure]> {
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

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorListGetItemOp : TF_Op<"TensorListGetItem", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$index,
    TF_Int32Tensor:$element_shape
  );

  let results = (outs
    TF_Tensor:$item
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_TensorListLengthOp : TF_Op<"TensorListLength", [Pure]> {
  let summary = "Returns the number of tensors in the input tensor list.";

  let description = [{
input_handle: the input list
length: the number of tensors in the list
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle
  );

  let results = (outs
    TF_Int32Tensor:$length
  );
}

def TF_TensorListSetItemOp : TF_Op<"TensorListSetItem", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$index,
    TF_Tensor:$item,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$resize_if_index_out_of_bounds
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_TensorListStackOp : TF_Op<"TensorListStack", [Pure]> {
  let summary = "Stacks all tensors in the list.";

  let description = [{
Requires that all tensors have the same shape.

input_handle: the input list
tensor: the gathered result
num_elements: optional. If not -1, the number of elements in the list.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$element_shape,

    DefaultValuedOptionalAttr<I64Attr, "-1">:$num_elements
  );

  let results = (outs
    TF_Tensor:$tensor
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_TensorScatterAddOp : TF_Op<"TensorScatterAdd", [Pure]> {
  let summary = [{
Adds sparse `updates` to an existing tensor according to `indices`.
  }];

  let description = [{
This operation creates a new tensor by adding sparse `updates` to the passed
in `tensor`.
This operation is very similar to `tf.compat.v1.scatter_nd_add`, except that the
updates are added onto an existing tensor (as opposed to a variable). If the
memory for the existing tensor cannot be re-used, a copy is made and updated.

`indices` is an integer tensor containing indices into a new tensor of shape
`tensor.shape`.  The last dimension of `indices` can be at most the rank of
`tensor.shape`:

```
indices.shape[-1] <= tensor.shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = tensor.shape.rank`) or slices
(if `indices.shape[-1] < tensor.shape.rank`) along dimension
`indices.shape[-1]` of `tensor.shape`.  `updates` is a tensor with shape

```
indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
```

The simplest form of `tensor_scatter_nd_add` is to add individual elements to a
tensor by index. For example, say we want to add 4 elements in a rank-1
tensor with 8 elements.

In Python, this scatter add operation would look like this:

>>> indices = tf.constant([[4], [3], [1], [7]])
>>> updates = tf.constant([9, 10, 11, 12])
>>> tensor = tf.ones([8], dtype=tf.int32)
>>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
>>> updated
<tf.Tensor: shape=(8,), dtype=int32,
numpy=array([ 1, 12,  1, 11, 10,  1,  1, 13], dtype=int32)>

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter add operation would look like this:

>>> indices = tf.constant([[0], [2]])
>>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]],
...                        [[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
>>> tensor = tf.ones([4, 4, 4],dtype=tf.int32)
>>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
>>> updated
<tf.Tensor: shape=(4, 4, 4), dtype=int32,
numpy=array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
             [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=int32)>


If `indices` contains any out-of-bound indices, depending on
`bad_indices_policy`, the op will either return an error or ignore the
out-of-bound indices. `bad_indices_policy` can be one of the following values:
1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
   historically on CPU and GPU we handle errors in different ways, and for
   backward compatibility we keep the default behavior.
2. "ERROR": raises error; GPU does not support this value.
3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to copy/update.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates,

    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$bad_indices_policy
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor copied from tensor and updates added according to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$tensor, "Value":$indices, "Value":$updates),
    [{build($_builder, $_state, tensor.getType(), tensor, indices, updates);}]>
  ];
}

def TF_TensorScatterUpdateOp : TF_Op<"TensorScatterUpdate", [Pure]> {
  let summary = [{
Scatter `updates` into an existing tensor according to `indices`.
  }];

  let description = [{
This operation creates a new tensor by applying sparse `updates` to the passed
in `tensor`.
This operation is very similar to `tf.scatter_nd`, except that the updates are
scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
for the existing tensor cannot be re-used, a copy is made and updated.

If `indices` contains duplicates, then we pick the last update for the index.

**WARNING**: There are some GPU specific semantics for this operation.
- If an out of bound index is found, the index is ignored.
- The order in which updates are applied is nondeterministic, so the output
will be nondeterministic if `indices` contains duplicates.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.

* `indices` must have at least 2 axes: `(num_updates, index_depth)`.
* The last axis of `indices` is how deep to index into `tensor` so  this index
  depth must be less than the rank of `tensor`: `indices.shape[-1] <= tensor.ndim`

if `indices.shape[-1] = tensor.rank` this Op indexes and updates scalar elements.
if `indices.shape[-1] < tensor.rank` it indexes and updates slices of the input
`tensor`.

Each `update` has a rank of `tensor.rank - indices.shape[-1]`.
The overall shape of `updates` is:

```
indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
```

If `indices` contains any out-of-bound indices, depending on
`bad_indices_policy`, the op will either return an error or ignore the
out-of-bound indices. `bad_indices_policy` can be one of the following values:
1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
   historically on CPU and GPU we handle errors in different ways, and for
   backward compatibility we keep the default behavior.
2. "ERROR": raises error; GPU does not support this value.
3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.

For usage examples see the python [tf.tensor_scatter_nd_update](
https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to copy/update.}]>:$tensor,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64, TF_Uint16]>, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates,

    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$bad_indices_policy
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor with the given shape and updates applied according
to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$tensor, "Value":$indices, "Value":$updates),
    [{build($_builder, $_state, tensor.getType(), tensor, indices, updates);}]>
  ];
}

def TF_TileOp : TF_Op<"Tile", [Pure]> {
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
    Arg<TF_Tensor, [{Can be of any rank.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D. Length must be the same as the number of dimensions in `input`}]>:$multiples
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tmultiples = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def TF_TopKV2Op : TF_Op<"TopKV2", [Pure]> {
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
    Arg<TF_IntOrFpTensor, [{1-D or higher with last dimension at least `k`.}]>:$input,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{0-D.  Number of top elements to look for along the last dimension (along each
row for matrices).}]>:$k,

    DefaultValuedOptionalAttr<BoolAttr, "true">:$sorted
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{The `k` largest elements along each last dimensional slice.}]>:$values,
    Res<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{The indices of `values` within the last dimension of `input`.}]>:$indices
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tk = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr index_type = TF_DerivedResultTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_TransposeOp : TF_Op<"Transpose", [Pure]> {
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
    OpBuilder<(ins "Value":$x, "Value":$perm)>
  ];

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_UniqueOp : TF_Op<"Unique", [Pure]> {
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
    Arg<TF_Tensor, [{1-D.}]>:$x
  );

  let results = (outs
    Res<TF_Tensor, [{1-D.}]>:$y,
    Res<TF_I32OrI64Tensor, [{1-D.}]>:$idx
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_idx = TF_DerivedResultTypeAttr<1>;
}

def TF_UnsortedSegmentMaxOp : TF_Op<"UnsortedSegmentMax", [Pure]> {
  let summary = "Computes the maximum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to `tf.math.unsorted_segment_sum`,
Instead of computing the sum over segments, it computes the maximum such that:

\\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
that `segment_ids[j...] == i`.

If the maximum is empty for a given segment ID `i`, it outputs the smallest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::lowest()`.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

Caution: On CPU, values in `segment_ids` are always validated to be less than
`num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
result in safe but unspecified behavior, which may include ignoring
out-of-bound indices or outputting a tensor with a 0 stored in the first
dimension of its shape if `num_segments` is 0.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
</div>

For example:

>>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
>>> tf.math.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
array([[4, 3, 3, 4],
       [5,  6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_UnsortedSegmentMinOp : TF_Op<"UnsortedSegmentMin", [Pure]> {
  let summary = "Computes the minimum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to `tf.math.unsorted_segment_sum`,
Instead of computing the sum over segments, it computes the minimum such that:

\\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
that `segment_ids[j...] == i`.

If the minimum is empty for a given segment ID `i`, it outputs the largest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::max()`.

For example:

>>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
>>> tf.math.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
array([[1, 2, 2, 1],
       [5, 6, 7, 8]], dtype=int32)

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

Caution: On CPU, values in `segment_ids` are always validated to be less than
`num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
result in safe but unspecified behavior, which may include ignoring
out-of-bound indices or outputting a tensor with a 0 stored in the first
dimension of its shape if `num_segments` is 0.
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_UnsortedSegmentProdOp : TF_Op<"UnsortedSegmentProd", [Pure]> {
  let summary = "Computes the product along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to `tf.math.unsorted_segment_sum`,
Instead of computing the sum over segments, it computes the product of all
entries belonging to a segment such that:

\\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
`j...` such that `segment_ids[j...] == i`.

For example:

>>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
>>> tf.math.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
array([[4, 6, 6, 4],
       [5, 6, 7, 8]], dtype=int32)

If there is no entry for a given segment ID `i`, it outputs 1.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.
Caution: On CPU, values in `segment_ids` are always validated to be less than
`num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
result in safe but unspecified behavior, which may include ignoring
out-of-bound indices or outputting a tensor with a 0 stored in the first
dimension of its shape if `num_segments` is 0.
  }];

  let arguments = (ins
    TF_NumberTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TF_NumberTensor, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_UnsortedSegmentSumOp : TF_Op<"UnsortedSegmentSum", [Pure]> {
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

Caution: On CPU, values in `segment_ids` are always validated to be less than
`num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
result in safe but unspecified behavior, which may include ignoring
out-of-bound indices or outputting a tensor with a 0 stored in the first
dimension of its shape if `num_segments` is 0.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
</div>

>>> c = [[1,2,3,4], [5,6,7,8], [4,3,2,1]]
>>> tf.math.unsorted_segment_sum(c, [0, 1, 0], num_segments=2).numpy()
array([[5, 5, 5, 5],
       [5, 6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TF_NumberTensor:$data,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TF_NumberTensor, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_WhereOp : TF_Op<"Where", [Pure]> {
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input
  );

  let results = (outs
    TF_Int64Tensor:$index
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaDynamicUpdateSliceOp : TF_Op<"XlaDynamicUpdateSlice", [Pure, TF_NoConstantFold]> {
  let summary = "Wraps the XLA DynamicUpdateSlice operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
.

XlaDynamicUpdateSlice generates a result which is the value of the `input`
operand, with a slice update overwritten at `indices`. The shape of `update`
determines the shape of the sub-array of the result which is updated. The shape
of indices must be rank == 1, with dimension size equal to the rank of `input`.

Handling of out-of-bounds slice indices is implementation-defined.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A `Tensor` of type T.}]>:$input,
    Arg<TF_Tensor, [{A `Tensor` of type T. Same rank as `input`.}]>:$update,
    Arg<TF_I32OrI64Tensor, [{A vector of indices into `input`. Must have length equal to the rank of
`input`.}]>:$indices
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type T.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;
}

def TF_ZerosLikeOp : TF_Op<"ZerosLike", [Pure, TF_Idempotent, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns a tensor of zeros with the same shape and type as x.";

  let arguments = (ins
    Arg<TF_Tensor, [{a tensor of type T.}]>:$x
  );

  let results = (outs
    Res<TF_Tensor, [{a tensor of the same shape and type as x but filled with zeros.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}
