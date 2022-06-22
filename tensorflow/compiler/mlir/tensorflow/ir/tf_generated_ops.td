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
//
// If you absolutely need to modify the generated fields of an op, move the op
// definition to `tf_ops.td` and perform the modification there.
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

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def TF_AbsOp : TF_Op<"Abs", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
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
}

def TF_AcosOp : TF_Op<"Acos", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes acos of x element-wise.";

  let description = [{
Provided an input tensor, the `tf.math.acos` operation returns the inverse cosine of each element of the tensor. If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.

  Input range is `[-1, 1]` and the output has a range of `[0, pi]`.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AcoshOp : TF_Op<"Acosh", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes inverse hyperbolic cosine of x element-wise.";

  let description = [{
Given an input tensor, the function computes inverse hyperbolic cosine of every element.
Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.

```python
x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
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

def TF_AddOp : TF_Op<"Add", [NoSideEffect, ResultsBroadcastableShape, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
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
    Variadic<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>>:$inputs
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>:$sum
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_AddV2Op : TF_Op<"AddV2", [Commutative, NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_AdjustContrastv2Op : TF_Op<"AdjustContrastv2", [NoSideEffect]> {
  let summary = "Adjust the contrast of one or more images.";

  let description = [{
`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{Images to adjust.  At least 3-D.}]>:$images,
    Arg<TF_Float32Tensor, [{A float multiplier for adjusting contrast.}]>:$contrast_factor
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32]>, [{The contrast-adjusted image or images.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AdjustHueOp : TF_Op<"AdjustHue", [NoSideEffect]> {
  let summary = "Adjust the hue of one or more images.";

  let description = [{
`images` is a tensor of at least 3 dimensions.  The last dimension is
interpreted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{Images to adjust.  At least 3-D.}]>:$images,
    Arg<TF_Float32Tensor, [{A float delta to add to the hue.}]>:$delta
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32]>, [{The hue-adjusted image or images.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AdjustSaturationOp : TF_Op<"AdjustSaturation", [NoSideEffect]> {
  let summary = "Adjust the saturation of one or more images.";

  let description = [{
`images` is a tensor of at least 3 dimensions.  The last dimension is
interpreted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{Images to adjust.  At least 3-D.}]>:$images,
    Arg<TF_Float32Tensor, [{A float scale to add to the saturation.}]>:$scale
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32]>, [{The hue-adjusted image or images.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    Arg<TF_BoolTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_BoolTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_AllToAllOp : TF_Op<"AllToAll", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "An Op to exchange data across TPU replicas.";

  let description = [{
On each replica, the input is split into `split_count` blocks along
`split_dimension` and send to the other replicas given group_assignment. After
receiving `split_count` - 1 blocks from other replicas, we concatenate the
blocks along `concat_dimension` as the output.

For example, suppose there are 2 TPU replicas:
replica 0 receives input: `[[A, B]]`
replica 1 receives input: `[[C, D]]`

group_assignment=`[[0, 1]]`
concat_dimension=0
split_dimension=1
split_count=2

replica 0's output: `[[A], [C]]`
replica 1's output: `[[B], [D]]`
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The local input to the sum.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor with shape
[num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
replica ids in the ith subgroup.}]>:$group_assignment,

    I64Attr:$concat_dimension,
    I64Attr:$split_dimension,
    I64Attr:$split_count
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The exchanged result.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AngleOp : TF_Op<"Angle", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns the argument of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the argument of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part.

The argument returned by this operation is of the form \\(atan2(b, a)\\).

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.angle(input) ==> [2.0132, 1.056]
```

@compatibility(numpy)
Equivalent to np.angle.
@end_compatibility
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

def TF_AnonymousIteratorOp : TF_Op<"AnonymousIterator", [TF_UniqueResourceAllocation]> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to the iterator that can be passed to a "MakeIterator" or
"IteratorGetNext" op. In contrast to Iterator, AnonymousIterator prevents
resource sharing by name, and does not keep a reference to the resource
container.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_AnonymousIteratorV2Op : TF_Op<"AnonymousIteratorV2", [TF_UniqueResourceAllocation]> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to the iterator that can be passed to a "MakeIterator" or
"IteratorGetNext" op. In contrast to Iterator, AnonymousIterator prevents
resource sharing by name, and does not keep a reference to the resource
container.}], [TF_DatasetIteratorAlloc]>:$handle,
    Res<TF_VariantTensor, [{A variant deleter that should be passed into the op that deletes the iterator.}]>:$deleter
  );
}

def TF_AnonymousIteratorV3Op : TF_Op<"AnonymousIteratorV3", [TF_UniqueResourceAllocation]> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to the iterator that can be passed to a "MakeIterator" or
"IteratorGetNext" op. In contrast to Iterator, AnonymousIterator prevents
resource sharing by name, and does not keep a reference to the resource
container.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_AnonymousMemoryCacheOp : TF_Op<"AnonymousMemoryCache", [TF_UniqueResourceAllocation]> {
  let summary = "";

  let arguments = (ins);

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetMemoryCacheAlloc]>:$handle,
    TF_VariantTensor:$deleter
  );
}

def TF_AnonymousMultiDeviceIteratorOp : TF_Op<"AnonymousMultiDeviceIterator", [TF_UniqueResourceAllocation]> {
  let summary = "A container for a multi device iterator resource.";

  let arguments = (ins
    Confined<StrArrayAttr, [ArrayMinCount<1>]>:$devices,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to a multi device iterator that can be passed to a
"MultiDeviceIteratorGetNextFromShard" op. In contrast to MultiDeviceIterator,
AnonymousIterator prevents resource sharing by name, and does not keep a
reference to the resource container.}], [TF_DatasetIteratorAlloc]>:$handle,
    Res<TF_VariantTensor, [{A variant deleter that should be passed into the op that deletes the iterator.}]>:$deleter
  );
}

def TF_AnonymousMultiDeviceIteratorV3Op : TF_Op<"AnonymousMultiDeviceIteratorV3", [TF_UniqueResourceAllocation]> {
  let summary = "A container for a multi device iterator resource.";

  let arguments = (ins
    Confined<StrArrayAttr, [ArrayMinCount<1>]>:$devices,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to a multi device iterator that can be passed to a
"MultiDeviceIteratorGetNextFromShard" op. In contrast to MultiDeviceIterator,
AnonymousIterator prevents resource sharing by name, and does not keep a
reference to the resource container.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_AnonymousRandomSeedGeneratorOp : TF_Op<"AnonymousRandomSeedGenerator", [TF_UniqueResourceAllocation]> {
  let summary = "";

  let arguments = (ins
    TF_Int64Tensor:$seed,
    TF_Int64Tensor:$seed2
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorAlloc]>:$handle,
    TF_VariantTensor:$deleter
  );
}

def TF_AnonymousSeedGeneratorOp : TF_Op<"AnonymousSeedGenerator", [TF_UniqueResourceAllocation]> {
  let summary = "";

  let arguments = (ins
    TF_Int64Tensor:$seed,
    TF_Int64Tensor:$seed2,
    TF_BoolTensor:$reshuffle
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorAlloc]>:$handle,
    TF_VariantTensor:$deleter
  );
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
    Arg<TF_BoolTensor, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TF_BoolTensor, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_ApproxTopKOp : TF_Op<"ApproxTopK", [NoSideEffect]> {
  let summary = [{
Returns min/max k values and their indices of the input operand in an approximate manner.
  }];

  let description = [{
Returns min/max k values and their indices of the input operand in an approximate manner.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{Array to search. Must be at least 1-D of the floating type}]>:$input,

    Confined<I64Attr, [IntMinValue<0>]>:$k,
    DefaultValuedAttr<I64Attr, "-1">:$reduction_dimension,
    DefaultValuedAttr<F32Attr, "0.95f">:$recall_target,
    DefaultValuedAttr<BoolAttr, "true">:$is_max_k,
    DefaultValuedAttr<I64Attr, "-1">:$reduction_input_size_override,
    DefaultValuedAttr<BoolAttr, "true">:$aggregate_to_topk
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

def TF_ApproximateEqualOp : TF_Op<"ApproximateEqual", [Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of abs(x-y) < tolerance element-wise.";

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y,

    DefaultValuedAttr<F32Attr, "1e-05f">:$tolerance
  );

  let results = (outs
    TF_BoolTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
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

def TF_AsStringOp : TF_Op<"AsString", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Converts each entry in the given tensor to strings.";

  let description = [{
Supports many numeric types and boolean.

For Unicode, see the
[https://www.tensorflow.org/tutorials/representation/unicode](Working with Unicode text)
tutorial.

Examples:

>>> tf.strings.as_string([3, 2])
<tf.Tensor: shape=(2,), dtype=string, numpy=array([b'3', b'2'], dtype=object)>
>>> tf.strings.as_string([3.1415926, 2.71828], precision=2).numpy()
array([b'3.14', b'2.72'], dtype=object)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8, TF_Variant]>:$input,

    DefaultValuedAttr<I64Attr, "-1">:$precision,
    DefaultValuedAttr<BoolAttr, "false">:$scientific,
    DefaultValuedAttr<BoolAttr, "false">:$shortest,
    DefaultValuedAttr<I64Attr, "-1">:$width,
    DefaultValuedAttr<StrAttr, "\"\"">:$fill
  );

  let results = (outs
    TF_StrTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AsinOp : TF_Op<"Asin", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the trignometric inverse sine of x element-wise.";

  let description = [{
The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.

**Note**: The output of `tf.math.asin` will lie within the invertible range
of sine, i.e [-pi/2, pi/2].

For example:

```python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.sin(x) # [0.8659266, 0.7068252]

tf.math.asin(y) # [1.047, 0.785] = x
```
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AsinhOp : TF_Op<"Asinh", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes inverse hyperbolic sine of x element-wise.";

  let description = [{
Given an input tensor, this function computes inverse hyperbolic sine
  for every element in the tensor. Both input and output has a range of
  `[-inf, inf]`.

  ```python
  x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
  tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
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

def TF_AssertOp : TF_Op<"Assert", []> {
  let summary = "Asserts that the given condition is true.";

  let description = [{
If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.
  }];

  let arguments = (ins
    Arg<TF_BoolTensor, [{The condition to evaluate.}]>:$condition,
    Arg<Variadic<TF_Tensor>, [{The tensors to print out when condition is false.}]>:$data,

    DefaultValuedAttr<I64Attr, "3">:$summarize
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<1>;

  let hasCanonicalizer = 1;
}

def TF_AssignOp : TF_Op<"Assign", []> {
  let summary = "Update 'ref' by assigning 'value' to it.";

  let description = [{
This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Should be from a `Variable` node. May be uninitialized.}]>:$ref,
    Arg<TF_Tensor, [{The value to be assigned to the variable.}]>:$value,

    DefaultValuedAttr<BoolAttr, "true">:$validate_shape,
    DefaultValuedAttr<BoolAttr, "true">:$use_locking
  );

  let results = (outs
    Res<TF_Tensor, [{= Same as "ref".  Returned as a convenience for operations that want
to use the new value after the variable has been reset.}]>:$output_ref
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AssignAddVariableOp : TF_Op<"AssignAddVariableOp", []> {
  let summary = "Adds a value to the current value of a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to
see the incremented value or a subsequent newer one.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{handle to the resource in which to store the variable.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_Tensor, [{the value by which the variable will be incremented.}]>:$value
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
    Arg<TF_ResourceTensor, [{handle to the resource in which to store the variable.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_Tensor, [{the value by which the variable will be incremented.}]>:$value
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
    Arg<TF_ResourceTensor, [{handle to the resource in which to store the variable.}], [TF_VariableWrite]>:$resource,
    Arg<TF_Tensor, [{the value to set the new tensor to use.}]>:$value,

    DefaultValuedAttr<BoolAttr, "false">:$validate_shape
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_AtanOp : TF_Op<"Atan", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the trignometric inverse tangent of x element-wise.";

  let description = [{
The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.

**Note**: The output of `tf.math.atan` will lie within the invertible range
of tan, i.e (-pi/2, pi/2).

For example:

```python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.tan(x) # [1.731261, 0.99920404]

tf.math.atan(y) # [1.047, 0.785] = x
```
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_Atan2Op : TF_Op<"Atan2", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_AtanhOp : TF_Op<"Atanh", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes inverse hyperbolic tangent of x element-wise.";

  let description = [{
Given an input tensor, this function computes inverse hyperbolic tangent
  for every element in the tensor. Input range is `[-1,1]` and output range is
  `[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
  input is `1`, output will be `inf`. Values outside the range will have
  `nan` as output.

  ```python
  x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
  tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
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

def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{4-D with shape `[batch, height, width, channels]`.}]>:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_FloatTensor, [{The average pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AvgPool3DOp : TF_Op<"AvgPool3D", [NoSideEffect]> {
  let summary = "Performs 3D average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize` window in
`value`.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{Shape `[batch, depth, rows, cols, channels]` tensor to pool over.}]>:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_FloatTensor, [{The average pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AvgPool3DGradOp : TF_Op<"AvgPool3DGrad", [NoSideEffect]> {
  let summary = "Computes gradients of average pooling function.";

  let arguments = (ins
    Arg<TF_Int32Tensor, [{The original input dimensions.}]>:$orig_input_shape,
    Arg<TF_FloatTensor, [{Output backprop of shape `[batch, depth, rows, cols, channels]`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_FloatTensor, [{The backprop for input.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_AvgPoolGradOp : TF_Op<"AvgPoolGrad", [NoSideEffect]> {
  let summary = "Computes gradients of the average pooling function.";

  let arguments = (ins
    Arg<TF_Int32Tensor, [{1-D.  Shape of the original input to `avg_pool`.}]>:$orig_input_shape,
    Arg<TF_FloatTensor, [{4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
the output of `avg_pool`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_FloatTensor, [{4-D.  Gradients w.r.t. the input of `avg_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_BatchDatasetV2Op : TF_Op<"BatchDatasetV2", [NoSideEffect]> {
  let summary = [{
Creates a dataset that batches `batch_size` elements from `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Arg<TF_Int64Tensor, [{A scalar representing the number of elements to accumulate in a batch.}]>:$batch_size,
    Arg<TF_BoolTensor, [{A scalar representing whether the last batch should be dropped in case its size
is smaller than desired.}]>:$drop_remainder,

    DefaultValuedAttr<BoolAttr, "false">:$parallel_copy,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_BatchFunctionOp : TF_Op<"BatchFunction", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = [{
Batches all the inputs tensors to the computation done by the function.
  }];

  let description = [{
So, for example, in the following code

  ```python

  # This input will be captured.
  y = tf.placeholder_with_default(1.0, shape=[])

  @tf.Defun(tf.float32)
  def computation(a):
    return tf.matmul(a, a) + y

  b = gen_batch_ops.batch_function(
          f=computation
          in_tensors=[a],
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg],
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          batching_queue="")
  ```

If more than one session.run call is simultaneously trying to compute `b`
the values of `a` will be gathered, non-deterministically concatenated
along the first axis, and only one thread will run the computation.

Assumes that all arguments of the function are Tensors which will be batched
along their first dimension.

Arguments that are captured, are not batched. The session.run call which does
the concatenation, will use the values of the captured tensors available to it.
Therefore, typical uses of captured tensors should involve values which remain
unchanged across session.run calls. Inference is a good example of this.

SparseTensor is not supported. The return value of the decorated function
must be a Tensor or a list/tuple of Tensors.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{The tensors to be batched.}]>:$in_tensors,
    Arg<Variadic<TF_Tensor>, [{The tensors which are captured in the function, and don't need
to be batched.}]>:$captured_tensors,

    SymbolRefAttr:$f,
    I64Attr:$num_batch_threads,
    I64Attr:$max_batch_size,
    I64Attr:$batch_timeout_micros,
    DefaultValuedAttr<I64Attr, "10">:$max_enqueued_batches,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$allowed_batch_sizes,
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<StrAttr, "\"\"">:$batching_queue,
    DefaultValuedAttr<BoolAttr, "false">:$enable_large_batch_splitting
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{The output tensors.}]>:$out_tensors
  );

  TF_DerivedOperandTypeListAttr Tcaptured = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
}

def TF_BatchMatMulOp : TF_Op<"BatchMatMul", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
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

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def TF_BatchMatMulV2Op : TF_Op<"BatchMatMulV2", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64]>, [{2-D or higher with shape `[..., r_x, c_x]`.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64]>, [{2-D or higher with shape `[..., r_y, c_y]`.}]>:$y,

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_BatchMatMulV3Op : TF_Op<"BatchMatMulV3", [NoSideEffect]> {
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

    DefaultValuedAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedAttr<BoolAttr, "false">:$adj_y
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64]>, [{3-D or higher with shape `[..., r_o, c_o]`}]>:$output
  );

  TF_DerivedOperandTypeAttr Ta = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tb = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr Tout = TF_DerivedResultTypeAttr<0>;
}

def TF_BatchNormWithGlobalNormalizationOp : TF_Op<"BatchNormWithGlobalNormalization", [NoSideEffect]> {
  let summary = "Batch normalization.";

  let description = [{
This op is deprecated. Prefer `tf.nn.batch_normalization`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 4D input Tensor.}]>:$t,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D mean Tensor with size matching the last dimension of t.
This is the first output from tf.nn.moments,
or a saved moving average thereof.}]>:$m,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D variance Tensor with size matching the last dimension of t.
This is the second output from tf.nn.moments,
or a saved moving average thereof.}]>:$v,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D beta Tensor with size matching the last dimension of t.
An offset to be added to the normalized tensor.}]>:$beta,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D gamma Tensor with size matching the last dimension of t.
If "scale_after_normalization" is true, this tensor will be multiplied
with the normalized tensor.}]>:$gamma,

    F32Attr:$variance_epsilon,
    BoolAttr:$scale_after_normalization
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$result
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BatchToSpaceOp : TF_Op<"BatchToSpace", [NoSideEffect]> {
  let summary = "BatchToSpace for 4-D tensors of type T.";

  let description = [{
This is a legacy version of the more general BatchToSpaceND.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{4-D tensor with shape
`[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
  depth]`. Note that the batch size of the input tensor must be divisible by
`block_size * block_size`.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
how many elements to crop from the intermediate result across the spatial
dimensions as follows:

    crops = [[crop_top, crop_bottom], [crop_left, crop_right]]}]>:$crops,

    Confined<I64Attr, [IntMinValue<2>]>:$block_size
  );

  let results = (outs
    Res<TF_Tensor, [{4-D with shape `[batch, height, width, depth]`, where:

      height = height_pad - crop_top - crop_bottom
      width = width_pad - crop_left - crop_right

The attr `block_size` must be greater than one. It indicates the block size.

Some examples:

(1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:

```
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

The output tensor has shape `[1, 2, 2, 1]` and value:

```
x = [[[[1], [2]], [[3], [4]]]]
```

(2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:

```
[[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
```

The output tensor has shape `[1, 2, 2, 3]` and value:

```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

(3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:

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

(4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:

```
x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
     [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
```

The output tensor has shape `[2, 2, 4, 1]` and value:

```
x = [[[[1], [3]], [[5], [7]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
```}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

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

def TF_BetaincOp : TF_Op<"Betainc", [NoSideEffect]> {
  let summary = [{
Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
  }];

  let description = [{
The regularized incomplete beta integral is defined as:


\\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)

where


\\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)


is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function.
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$a,
    TF_F32OrF64Tensor:$b,
    TF_F32OrF64Tensor:$x
  );

  let results = (outs
    TF_F32OrF64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BiasAddOp : TF_Op<"BiasAdd", [NoSideEffect, TF_LayoutSensitiveInterface]> {
  let summary = "Adds `bias` to `value`.";

  let description = [{
This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Any number of dimensions.}]>:$value,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D with size the last dimension of `value`.}]>:$bias,

    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Broadcasted sum of `value` and `bias`.}]>:$output
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Any number of dimensions.}]>:$out_backprop,

    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D with size the feature dimension of `out_backprop`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_BiasAddV1Op : TF_Op<"BiasAddV1", [NoSideEffect]> {
  let summary = "Adds `bias` to `value`.";

  let description = [{
This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Any number of dimensions.}]>:$value,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D with size the last dimension of `value`.}]>:$bias
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Broadcasted sum of `value` and `bias`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_BincountOp : TF_Op<"Bincount", [NoSideEffect]> {
  let summary = [{
Counts the number of occurrences of each value in an integer array.
  }];

  let description = [{
Outputs a vector with length `size` and the same dtype as `weights`. If
`weights` are empty, then index `i` stores the number of times the value `i` is
counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
the value in `weights` at each index where the corresponding value in `arr` is
`i`.

Values in `arr` outside of the range [0, size) are ignored.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{int32 `Tensor`.}]>:$arr,
    Arg<TF_Int32Tensor, [{non-negative int32 scalar `Tensor`.}]>:$size,
    Arg<TensorOf<[TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{is an int32, int64, float32, or float64 `Tensor` with the same
shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
equal to 1.}]>:$weights
  );

  let results = (outs
    Res<TensorOf<[TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{1D `Tensor` with length equal to `size`. The counts or summed weights for
each value in the range [0, size).}]>:$bins
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
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

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr type = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_BitwiseAndOp : TF_Op<"BitwiseAnd", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_BitwiseOrOp : TF_Op<"BitwiseOr", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_BitwiseXorOp : TF_Op<"BitwiseXor", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_BoostedTreesBucketizeOp : TF_Op<"BoostedTreesBucketize", [NoSideEffect, SameVariadicOperandSize]> {
  let summary = "Bucketize each feature based on bucket boundaries.";

  let description = [{
An op that returns a list of float tensors, where each tensor represents the
bucketized values for a single feature.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Float32Tensor>, [{float; List of Rank 1 Tensor each containing float values for a single feature.}]>:$float_values,
    Arg<Variadic<TF_Float32Tensor>, [{float; List of Rank 1 Tensors each containing the bucket boundaries for a single
feature.}]>:$bucket_boundaries
  );

  let results = (outs
    Res<Variadic<TF_Int32Tensor>, [{int; List of Rank 1 Tensors each containing the bucketized values for a single feature.}]>:$buckets
  );

  TF_DerivedOperandSizeAttr num_features = TF_DerivedOperandSizeAttr<0>;
}

def TF_BroadcastArgsOp : TF_Op<"BroadcastArgs", [NoSideEffect]> {
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

def TF_BroadcastGradientArgsOp : TF_Op<"BroadcastGradientArgs", [NoSideEffect, SameOperandsAndResultElementType, TF_OperandHasRank<0, 1>, TF_OperandHasRank<1, 1>, TF_ResultHasRank<0, 1>, TF_ResultHasRank<1, 1>]> {
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

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_BroadcastToOp : TF_Op<"BroadcastTo", [NoSideEffect]> {
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

def TF_BucketizeOp : TF_Op<"Bucketize", [NoSideEffect, SameOperandsAndResultShape]> {
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

def TF_CacheDatasetV2Op : TF_Op<"CacheDatasetV2", []> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    TF_StrTensor:$filename,
    Arg<TF_ResourceTensor, "", [TF_DatasetMemoryCacheRead, TF_DatasetMemoryCacheWrite]>:$cache,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_CastOp : TF_Op<"Cast", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Cast x of type SrcT to y of DstT.";

  let arguments = (ins
    TF_Tensor:$x,

    DefaultValuedAttr<BoolAttr, "false">:$Truncate
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr SrcT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr DstT = TF_DerivedResultTypeAttr<0>;

  let hasFolder = 1;
}

def TF_CeilOp : TF_Op<"Ceil", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns element-wise smallest integer not less than x.";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CheckNumericsOp : TF_Op<"CheckNumerics", [TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_CholeskyOp : TF_Op<"Cholesky", [NoSideEffect]> {
  let summary = [{
Computes the Cholesky decomposition of one or more square matrices.
  }];

  let description = [{
The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices.

The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

**Note**: The gradient computation on GPU is faster for large matrices but
not for large batch dimensions when the submatrices are small. In this
case it might be faster to use the CPU.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, M]`.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, M]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ClipByValueOp : TF_Op<"ClipByValue", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
  let summary = "Clips tensor values to a specified min and max.";

  let description = [{
Given a tensor `t`, this operation returns a tensor of the same type and
shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
Any values less than `clip_value_min` are set to `clip_value_min`. Any values
greater than `clip_value_max` are set to `clip_value_max`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A `Tensor`.}]>:$t,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
as `t`. The minimum value to clip by.}]>:$clip_value_min,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
as `t`. The maximum value to clip by.}]>:$clip_value_max
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A clipped `Tensor` with the same shape as input 't'.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CollectiveAssignGroupV2Op : TF_Op<"CollectiveAssignGroupV2", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Assign group keys based on group assignment.";

  let arguments = (ins
    TF_Int32Tensor:$group_assignment,
    TF_Int32Tensor:$device_index,
    TF_Int32Tensor:$base_key
  );

  let results = (outs
    TF_Int32Tensor:$group_size,
    TF_Int32Tensor:$group_key
  );
}

def TF_CollectiveBcastRecvOp : TF_Op<"CollectiveBcastRecv", []> {
  let summary = "Receives a tensor value broadcast from another device.";

  let arguments = (ins
    I64Attr:$group_size,
    I64Attr:$group_key,
    I64Attr:$instance_key,
    TF_ShapeAttr:$shape,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$communication_hint,
    DefaultValuedAttr<F32Attr, "0.0f">:$timeout_seconds
  );

  let results = (outs
    TensorOf<[TF_Bool, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$data
  );

  TF_DerivedResultTypeAttr T = TF_DerivedResultTypeAttr<0>;
}

def TF_CollectiveBcastSendOp : TF_Op<"CollectiveBcastSend", []> {
  let summary = "Broadcasts a tensor value to one or more other devices.";

  let arguments = (ins
    TensorOf<[TF_Bool, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$input,

    I64Attr:$group_size,
    I64Attr:$group_key,
    I64Attr:$instance_key,
    TF_ShapeAttr:$shape,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$communication_hint,
    DefaultValuedAttr<F32Attr, "0.0f">:$timeout_seconds
  );

  let results = (outs
    TensorOf<[TF_Bool, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$data
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CollectiveGatherOp : TF_Op<"CollectiveGather", []> {
  let summary = [{
Mutually accumulates multiple tensors of identical type and shape.
  }];

  let arguments = (ins
    TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$input,

    I64Attr:$group_size,
    I64Attr:$group_key,
    I64Attr:$instance_key,
    TF_ShapeAttr:$shape,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$communication_hint,
    DefaultValuedAttr<F32Attr, "0.0f">:$timeout_seconds
  );

  let results = (outs
    TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$data
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CollectivePermuteOp : TF_Op<"CollectivePermute", []> {
  let summary = "An Op to permute tensors across replicated TPU instances.";

  let description = [{
Each instance supplies its own input.

For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
source_target_pairs=`[[0,1],[1,2],[2,3],[3,0]]` gets the outputs:
`[D, A, B, C]`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The local input to be permuted. Currently only supports float and
bfloat16.}]>:$input,
    Arg<TF_Int32Tensor, [{A tensor with shape [num_pairs, 2].}]>:$source_target_pairs
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The permuted input.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CollectiveReduceOp : TF_Op<"CollectiveReduce", [TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Mutually reduces multiple tensors of identical type and shape.
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$input,

    I64Attr:$group_size,
    I64Attr:$group_key,
    I64Attr:$instance_key,
    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add"]>:$merge_op,
    TF_AnyStrAttrOf<["Id", "Div"]>:$final_op,
    I64ArrayAttr:$subdiv_offsets,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$wait_for,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$communication_hint,
    DefaultValuedAttr<F32Attr, "0.0f">:$timeout_seconds
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$data
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CollectiveReduceV2Op : TF_Op<"CollectiveReduceV2", []> {
  let summary = [{
Mutually reduces multiple tensors of identical type and shape.
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$input,
    TF_Int32Tensor:$group_size,
    TF_Int32Tensor:$group_key,
    TF_Int32Tensor:$instance_key,
    Variadic<TF_ResourceTensor>:$ordering_token,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add"]>:$merge_op,
    TF_AnyStrAttrOf<["Id", "Div"]>:$final_op,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$communication_hint,
    DefaultValuedAttr<F32Attr, "0.0f">:$timeout_seconds,
    DefaultValuedAttr<I64Attr, "-1">:$max_subdivs_per_device
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$data
  );

  TF_DerivedOperandSizeAttr Nordering_token = TF_DerivedOperandSizeAttr<4>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ComplexOp : TF_Op<"Complex", [NoSideEffect, ResultsBroadcastableShape]> {
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

def TF_ConcatOp : TF_Op<"Concat", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let arguments = (ins
    Arg<TF_Int32Tensor, [{0-D.  The dimension along which to concatenate.  Must be in the
range [0, rank(values)).}]>:$concat_dim,
    Arg<Variadic<TF_Tensor>, [{The `N` Tensors to concatenate. Their ranks and types must match,
and their sizes must match in all dimensions except `concat_dim`.}]>:$values
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` with the concatenation of values stacked along the
`concat_dim` dimension.  This tensor's shape matches that of `values` except
in `concat_dim` where it has the sum of the sizes.}]>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_ConcatOffsetOp : TF_Op<"ConcatOffset", [NoSideEffect]> {
  let summary = "Computes offsets of concat inputs within its output.";

  let description = [{
For example:

```
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```

This is typically used by gradient computations for a concat operation.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{The dimension along which to concatenate.}]>:$concat_dim,
    Arg<Variadic<TF_Int32Tensor>, [{The `N` int32 vectors representing shape of tensors being concatenated.}]>:$shape
  );

  let results = (outs
    Res<Variadic<TF_Int32Tensor>, [{The `N` int32 vectors representing the starting offset
of input tensors within the concatenated output.}]>:$offset
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<1>;

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_ConcatV2Op : TF_Op<"ConcatV2", [NoSideEffect]> {
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

def TF_ConfigureDistributedTPUOp : TF_Op<"ConfigureDistributedTPU", []> {
  let summary = [{
Sets up the centralized structures for a distributed TPU system.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "\"\"">:$embedding_config,
    DefaultValuedAttr<StrAttr, "\"\"">:$tpu_embedding_config,
    DefaultValuedAttr<BoolAttr, "false">:$is_global_init,
    DefaultValuedAttr<BoolAttr, "false">:$enable_whole_mesh_compilations,
    DefaultValuedAttr<BoolAttr, "true">:$compilation_failure_closes_chips,
    DefaultValuedAttr<I64Attr, "0">:$tpu_cancellation_closes_chips
  );

  let results = (outs
    Res<TF_StrTensor, [{A serialized tensorflow.tpu.TopologyProto that describes the TPU
topology.}]>:$topology
  );
}

def TF_ConfigureTPUEmbeddingOp : TF_Op<"ConfigureTPUEmbedding", []> {
  let summary = "Sets up TPUEmbedding in a distributed TPU system.";

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs);
}

def TF_ConjOp : TF_Op<"Conj", [Involution, NoSideEffect, SameOperandsAndResultType]> {
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
}

def TF_ConjugateTransposeOp : TF_Op<"ConjugateTranspose", [NoSideEffect]> {
  let summary = [{
Shuffle dimensions of x according to a permutation and conjugate the result.
  }];

  let description = [{
The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
  `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`
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
}

def TF_Conv2DOp : TF_Op<"Conv2D", [InferTensorType, NoSideEffect, TF_LayoutSensitiveInterface]> {
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
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
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

def TF_Conv2DBackpropFilterOp : TF_Op<"Conv2DBackpropFilter", [NoSideEffect, TF_LayoutSensitiveInterface]> {
  let summary = [{
Computes the gradients of convolution with respect to the filter.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{4-D with shape `[batch, in_height, in_width, in_channels]`.}]>:$input,
    Arg<TF_Int32Tensor, [{An integer vector representing the tensor shape of `filter`,
where `filter` is a 4-D
`[filter_height, filter_width, in_channels, out_channels]` tensor.}]>:$filter_sizes,
    Arg<TF_FloatTensor, [{4-D with shape `[batch, out_height, out_width, out_channels]`.
Gradients w.r.t. the output of the convolution.}]>:$out_backprop,

    I64ArrayAttr:$strides,
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    Res<TF_FloatTensor, [{4-D with shape
`[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
the `filter` input of the convolution.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    // TF_LayoutSensitiveInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0, 2}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {}; }
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
  }];
}

def TF_Conv2DBackpropInputOp : TF_Op<"Conv2DBackpropInput", [NoSideEffect, TF_LayoutSensitiveInterface]> {
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
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
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

def TF_Conv3DOp : TF_Op<"Conv3D", [InferTensorType, NoSideEffect]> {
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
    Arg<TF_FloatTensor, [{Shape `[batch, in_depth, in_height, in_width, in_channels]`.}]>:$input,
    Arg<TF_FloatTensor, [{Shape `[filter_depth, filter_height, filter_width, in_channels,
out_channels]`. `in_channels` must match between `input` and `filter`.}]>:$filter,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];

}

def TF_Conv3DBackpropFilterV2Op : TF_Op<"Conv3DBackpropFilterV2", [NoSideEffect]> {
  let summary = [{
Computes the gradients of 3-D convolution with respect to the filter.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{Shape `[batch, depth, rows, cols, in_channels]`.}]>:$input,
    Arg<TF_Int32Tensor, [{An integer vector representing the tensor shape of `filter`,
where `filter` is a 5-D
`[filter_depth, filter_height, filter_width, in_channels, out_channels]`
tensor.}]>:$filter_sizes,
    Arg<TF_FloatTensor, [{Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
out_channels]`.}]>:$out_backprop,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_Conv3DBackpropInputV2Op : TF_Op<"Conv3DBackpropInputV2", [NoSideEffect]> {
  let summary = [{
Computes the gradients of 3-D convolution with respect to the input.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{An integer vector representing the tensor shape of `input`,
where `input` is a 5-D
`[batch, depth, rows, cols, in_channels]` tensor.}]>:$input_sizes,
    Arg<TF_FloatTensor, [{Shape `[depth, rows, cols, in_channels, out_channels]`.
`in_channels` must match between `input` and `filter`.}]>:$filter,
    Arg<TF_FloatTensor, [{Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
out_channels]`.}]>:$out_backprop,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
}

def TF_CosOp : TF_Op<"Cos", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

def TF_CoshOp : TF_Op<"Cosh", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes hyperbolic cosine of x element-wise.";

  let description = [{
Given an input tensor, this function computes hyperbolic cosine of every
  element in the tensor. Input range is `[-inf, inf]` and output range
  is `[1, inf]`.

  ```python
  x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
  tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
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

def TF_CrossOp : TF_Op<"Cross", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Compute the pairwise cross product.";

  let description = [{
`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.
  }];

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{A tensor containing 3-element vectors.}]>:$a,
    Arg<TF_IntOrFpTensor, [{Another tensor, of same type and shape as `a`.}]>:$b
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Pairwise cross product of the vectors in `a` and `b`.}]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CrossReplicaSumOp : TF_Op<"CrossReplicaSum", [NoSideEffect, TF_AllTypesMatch<["input", "output"]>, TF_NoConstantFold]> {
  let summary = "An Op to sum inputs across replicated TPU instances.";

  let description = [{
Each instance supplies its own input.

For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
and `B, D, F, H` as group 1. Thus we get the outputs:
`[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Uint32]>, [{The local input to the sum.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor with shape
[num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
replica ids in the ith subgroup.}]>:$group_assignment
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Uint32]>, [{The sum of all the distributed inputs.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CumprodOp : TF_Op<"Cumprod", [NoSideEffect, TF_AllTypesMatch<["x", "out"]>]> {
  let summary = [{
Compute the cumulative product of the tensor `x` along `axis`.
  }];

  let description = [{
By default, this op performs an inclusive cumprod, which means that the first
element of the input is identical to the first element of the output:

```python
tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
performed instead:

```python
tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
```

By setting the `reverse` kwarg to `True`, the cumprod is performed in the
opposite direction:

```python
tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
```

This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:

```python
tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A `Tensor`. Must be one of the following types: `float32`, `float64`,
`int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
`complex128`, `qint8`, `quint8`, `qint32`, `half`.}]>:$x,
    Arg<TF_I32OrI64Tensor, [{A `Tensor` of type `int32` (default: 0). Must be in the range
`[-rank(x), rank(x))`.}]>:$axis,

    DefaultValuedAttr<BoolAttr, "false">:$exclusive,
    DefaultValuedAttr<BoolAttr, "false">:$reverse
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_CumsumOp : TF_Op<"Cumsum", [NoSideEffect, TF_AllTypesMatch<["x", "out"]>]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A `Tensor`. Must be one of the following types: `float32`, `float64`,
`int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
`complex128`, `qint8`, `quint8`, `qint32`, `half`.}]>:$x,
    Arg<TF_I32OrI64Tensor, [{A `Tensor` of type `int32` (default: 0). Must be in the range
`[-rank(x), rank(x))`.}]>:$axis,

    DefaultValuedAttr<BoolAttr, "false">:$exclusive,
    DefaultValuedAttr<BoolAttr, "false">:$reverse
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_DataFormatDimMapOp : TF_Op<"DataFormatDimMap", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Returns the dimension index in the destination data format given the one in
  }];

  let description = [{
the source data format.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{A Tensor with each element as a dimension index in source data format.
Must be in the range [-4, 4).}]>:$x,

    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$src_format,
    DefaultValuedAttr<StrAttr, "\"NCHW\"">:$dst_format
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{A Tensor with each element as a dimension index in destination data format.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DataFormatVecPermuteOp : TF_Op<"DataFormatVecPermute", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Permute input tensor from `src_format` to `dst_format`.";

  let description = [{
Given source and destination format strings of length n=4 or 5, the input
tensor must be a vector of size n or n-2, or a 2D tensor of shape
(n, 2) or (n-2, 2).

If the first dimension of the input tensor is n-2, it is assumed that
non-spatial dimensions are omitted (i.e `N`, `C`).

For example, with `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
```
[1, 2, 3, 4]
```
, the output will be:
```
[1, 4, 2, 3]
```
With `src_format` of `NDHWC`, `dst_format` of `NCDHW`, and input:
```
[[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
```
, the output will be:
```
[[1, 6], [5, 10], [2, 7], [3, 8], [4, 9]]
```
With `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
```
[1, 2]
```
, the output will be:
```
[1, 2]
```
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{Tensor of rank 1 or 2 in source data format.}]>:$x,

    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$src_format,
    DefaultValuedAttr<StrAttr, "\"NCHW\"">:$dst_format
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{Tensor of rank 1 or 2 in destination data format.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_DebugIdentityV2Op : TF_Op<"DebugIdentityV2", []> {
  let summary = "Debug Identity V2 Op.";

  let description = [{
Provides an identity mapping from input to output, while writing the content of
the input tensor by calling DebugEventsWriter.

The semantics of the input tensor depends on tensor_debug_mode. In typical
usage, the input tensor comes directly from the user computation only when
graph_debug_mode is FULL_TENSOR (see protobuf/debug_event.proto for a
list of all the possible values of graph_debug_mode). For the other debug modes,
the input tensor should be produced by an additional op or subgraph that
computes summary information about one or more tensors.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Input tensor, non-Reference type}]>:$input,

    DefaultValuedAttr<StrAttr, "\"\"">:$tfdbg_context_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$op_name,
    DefaultValuedAttr<I64Attr, "-1">:$output_slot,
    DefaultValuedAttr<I64Attr, "-1">:$tensor_debug_mode,
    DefaultValuedAttr<StrArrayAttr, "{}">:$debug_urls,
    DefaultValuedAttr<I64Attr, "1000">:$circular_buffer_size,
    DefaultValuedAttr<StrAttr, "\"\"">:$tfdbg_run_id
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DecodeAndCropJpegOp : TF_Op<"DecodeAndCropJpeg", [NoSideEffect]> {
  let summary = "Decode and Crop a JPEG-encoded image to a uint8 tensor.";

  let description = [{
The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.


It is equivalent to a combination of decode and crop, but much faster by only
decoding partial jpeg image.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{0-D.  The JPEG-encoded image.}]>:$contents,
    Arg<TF_Int32Tensor, [{1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].}]>:$crop_window,

    DefaultValuedAttr<I64Attr, "0">:$channels,
    DefaultValuedAttr<I64Attr, "1">:$ratio,
    DefaultValuedAttr<BoolAttr, "true">:$fancy_upscaling,
    DefaultValuedAttr<BoolAttr, "false">:$try_recover_truncated,
    DefaultValuedAttr<F32Attr, "1.0f">:$acceptable_fraction,
    DefaultValuedAttr<StrAttr, "\"\"">:$dct_method
  );

  let results = (outs
    Res<TF_Uint8Tensor, [{3-D with shape `[height, width, channels]`..}]>:$image
  );
}

def TF_DecodeGifOp : TF_Op<"DecodeGif", [NoSideEffect]> {
  let summary = "Decode the frame(s) of a GIF-encoded image to a uint8 tensor.";

  let description = [{
GIF images with frame or transparency compression are not supported.
On Linux and MacOS systems, convert animated GIFs from compressed to
uncompressed by running:

    convert $src.gif -coalesce $dst.gif

This op also supports decoding JPEGs and PNGs, though it is cleaner to use
`tf.io.decode_image`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{0-D.  The GIF-encoded image.}]>:$contents
  );

  let results = (outs
    Res<TF_Uint8Tensor, [{4-D with shape `[num_frames, height, width, 3]`. RGB channel order.}]>:$image
  );
}

def TF_DecodeJpegOp : TF_Op<"DecodeJpeg", [NoSideEffect]> {
  let summary = "Decode a JPEG-encoded image to a uint8 tensor.";

  let description = [{
The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.


This op also supports decoding PNGs and non-animated GIFs since the interface is
the same, though it is cleaner to use `tf.io.decode_image`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{0-D.  The JPEG-encoded image.}]>:$contents,

    DefaultValuedAttr<I64Attr, "0">:$channels,
    DefaultValuedAttr<I64Attr, "1">:$ratio,
    DefaultValuedAttr<BoolAttr, "true">:$fancy_upscaling,
    DefaultValuedAttr<BoolAttr, "false">:$try_recover_truncated,
    DefaultValuedAttr<F32Attr, "1.0f">:$acceptable_fraction,
    DefaultValuedAttr<StrAttr, "\"\"">:$dct_method
  );

  let results = (outs
    Res<TF_Uint8Tensor, [{3-D with shape `[height, width, channels]`..}]>:$image
  );
}

def TF_DecodePaddedRawOp : TF_Op<"DecodePaddedRaw", [NoSideEffect]> {
  let summary = "Reinterpret the bytes of a string as a vector of numbers.";

  let arguments = (ins
    Arg<TF_StrTensor, [{Tensor of string to be decoded.}]>:$input_bytes,
    Arg<TF_Int32Tensor, [{Length in bytes for each element of the decoded output. Must be a multiple
of the size of the output type.}]>:$fixed_length,

    DefaultValuedAttr<BoolAttr, "true">:$little_endian
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{A Tensor with one more dimension than the input `bytes`. The added dimension
will have size equal to the length of the elements of `bytes` divided by the
number of bytes to represent `out_type`.}]>:$output
  );

  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;
}

def TF_DecodePngOp : TF_Op<"DecodePng", [NoSideEffect]> {
  let summary = "Decode a PNG-encoded image to a uint8 or uint16 tensor.";

  let description = [{
The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number
of color channels.

This op also supports decoding JPEGs and non-animated GIFs since the interface
is the same, though it is cleaner to use `tf.io.decode_image`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{0-D.  The PNG-encoded image.}]>:$contents,

    DefaultValuedAttr<I64Attr, "0">:$channels
  );

  let results = (outs
    Res<TensorOf<[TF_Uint16, TF_Uint8]>, [{3-D with shape `[height, width, channels]`.}]>:$image
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_DeleteIteratorOp : TF_Op<"DeleteIterator", []> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A handle to the iterator to delete.}], [TF_DatasetIteratorFree]>:$handle,
    Arg<TF_VariantTensor, [{A variant deleter.}]>:$deleter
  );

  let results = (outs);
}

def TF_DeleteMemoryCacheOp : TF_Op<"DeleteMemoryCache", []> {
  let summary = "";

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetMemoryCacheFree]>:$handle,
    TF_VariantTensor:$deleter
  );

  let results = (outs);
}

def TF_DeleteMultiDeviceIteratorOp : TF_Op<"DeleteMultiDeviceIterator", []> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A handle to the multi device iterator to delete.}], [TF_DatasetIteratorFree]>:$multi_device_iterator,
    Arg<Variadic<TF_ResourceTensor>, [{A list of iterator handles (unused). This is added so that automatic control dependencies get added during function tracing that ensure this op runs after all the dependent iterators are deleted.}], [TF_DatasetIteratorRead]>:$iterators,
    Arg<TF_VariantTensor, [{A variant deleter.}]>:$deleter
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<1>;
}

def TF_DeleteRandomSeedGeneratorOp : TF_Op<"DeleteRandomSeedGenerator", []> {
  let summary = "";

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorFree]>:$handle,
    TF_VariantTensor:$deleter
  );

  let results = (outs);
}

def TF_DeleteSeedGeneratorOp : TF_Op<"DeleteSeedGenerator", []> {
  let summary = "";

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorFree]>:$handle,
    TF_VariantTensor:$deleter
  );

  let results = (outs);
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

    Confined<I64Attr, [IntMinValue<2>]>:$block_size,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
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
    TF_FloatTensor:$input,
    TF_FloatTensor:$filter,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DepthwiseConv2dNativeBackpropFilterOp : TF_Op<"DepthwiseConv2dNativeBackpropFilter", [NoSideEffect]> {
  let summary = [{
Computes the gradients of depthwise convolution with respect to the filter.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{4-D with shape based on `data_format`.  For example, if
`data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
in_width, in_channels]` tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An integer vector representing the tensor shape of `filter`,
where `filter` is a 4-D
`[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.}]>:$filter_sizes,
    Arg<TF_FloatTensor, [{4-D with shape  based on `data_format`.
For example, if `data_format` is 'NHWC' then
out_backprop shape is `[batch, out_height, out_width, out_channels]`.
Gradients w.r.t. the output of the convolution.}]>:$out_backprop,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    Res<TF_FloatTensor, [{4-D with shape
`[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
the `filter` input of the convolution.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DepthwiseConv2dNativeBackpropInputOp : TF_Op<"DepthwiseConv2dNativeBackpropInput", [NoSideEffect]> {
  let summary = [{
Computes the gradients of depthwise convolution with respect to the input.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{An integer vector representing the shape of `input`, based
on `data_format`.  For example, if `data_format` is 'NHWC' then
 `input` is a 4-D `[batch, height, width, channels]` tensor.}]>:$input_sizes,
    Arg<TF_FloatTensor, [{4-D with shape
`[filter_height, filter_width, in_channels, depthwise_multiplier]`.}]>:$filter,
    Arg<TF_FloatTensor, [{4-D with shape  based on `data_format`.
For example, if `data_format` is 'NHWC' then
out_backprop shape is `[batch, out_height, out_width, out_channels]`.
Gradients w.r.t. the output of the convolution.}]>:$out_backprop,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    Res<TF_FloatTensor, [{4-D with shape according to `data_format`.  For example, if
`data_format` is 'NHWC', output shape is `[batch, in_height,
in_width, in_channels]`.  Gradient w.r.t. the input of the
convolution.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_DequantizeOp : TF_Op<"Dequantize", [NoSideEffect]> {
  let summary = [{
Dequantize the 'input' tensor into a float or bfloat16 Tensor.
  }];

  let description = [{
[min_range, max_range] are scalar floats that specify the range for
the output. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
if T == qint8: in[i] += (range(T) + 1)/ 2.0
out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

If the input comes from a QuantizedRelu6, the output type is
quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
Dequantize on quint8 will take each value, cast to float, and multiply
by 6 / 255.
Note that if quantizedtype is qint8, the operation will additionally add
each value by 128 prior to casting.

If the mode is 'MIN_FIRST', then this approach is used:

```c++
num_discrete_values = 1 << (# of bits in T)
range_adjust = num_discrete_values / (num_discrete_values - 1)
range = (range_max - range_min) * range_adjust
range_scale = range / num_discrete_values
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
```

If the mode is `SCALED`, dequantization is performed by multiplying each
input value by a scaling_factor. (Thus an input of 0 always maps to 0.0).

The scaling_factor is determined from `min_range`, `max_range`, and
`narrow_range` in a way that is compatible with `QuantizeAndDequantize{V2|V3}`
and `QuantizeV2`, using the following algorithm:

```c++

  const int min_expected_T = std::numeric_limits<T>::min() +
    (narrow_range ? 1 : 0);
  const int max_expected_T = std::numeric_limits<T>::max();
  const float max_expected_T = std::numeric_limits<float>::max();

  const float scale_factor =
    (std::numeric_limits<T>::min() == 0) ? (max_range / max_expected_T)
                                         : std::max(min_range / min_expected_T,
                                                    max_range / max_expected_T);
```
  }];

  let arguments = (ins
    TensorOf<[TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8]>:$input,
    Arg<TF_Float32Tensor, [{The minimum scalar value possibly produced for the input.}]>:$min_range,
    Arg<TF_Float32Tensor, [{The maximum scalar value possibly produced for the input.}]>:$max_range,

    DefaultValuedAttr<TF_AnyStrAttrOf<["MIN_COMBINED", "MIN_FIRST", "SCALED"]>, "\"MIN_COMBINED\"">:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_DeserializeIteratorOp : TF_Op<"DeserializeIterator", []> {
  let summary = [{
Converts the given variant tensor to an iterator and stores it in the given resource.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A handle to an iterator resource.}], [TF_DatasetIteratorWrite]>:$resource_handle,
    Arg<TF_VariantTensor, [{A variant tensor storing the state of the iterator contained in the
resource.}]>:$serialized
  );

  let results = (outs);
}

def TF_DeserializeSparseOp : TF_Op<"DeserializeSparse", [NoSideEffect]> {
  let summary = "Deserialize `SparseTensor` objects.";

  let description = [{
The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
the last dimension stores serialized `SparseTensor` objects and the other N
dimensions (N >= 0) correspond to a batch. The ranks of the original
`SparseTensor` objects must all match. When the final `SparseTensor` is
created, its rank is the rank of the incoming `SparseTensor` objects plus N;
the sparse tensors have been concatenated along new dimensions, one for each
batch.

The output `SparseTensor` object's shape values for the original dimensions
are the max across the input `SparseTensor` objects' shape values for the
corresponding dimensions. The new dimensions match the size of the batch.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]

and

    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]

then the final deserialized `SparseTensor` will be:

    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Str, TF_Variant]>, [{The serialized `SparseTensor` objects. The last dimension
must have 3 columns.}]>:$serialized_sparse
  );

  let results = (outs
    TF_Int64Tensor:$sparse_indices,
    TF_Tensor:$sparse_values,
    TF_Int64Tensor:$sparse_shape
  );

  TF_DerivedOperandTypeAttr Tserialized = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<1>;
}

def TF_DestroyResourceOp : TF_Op<"DestroyResourceOp", []> {
  let summary = "Deletes the resource specified by the handle.";

  let description = [{
All subsequent operations using the resource will result in a NotFound
error status.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{handle to the resource to delete.}]>:$resource,

    DefaultValuedAttr<BoolAttr, "true">:$ignore_lookup_error
  );

  let results = (outs);
}

def TF_DeviceIndexOp : TF_Op<"DeviceIndex", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Return the index of device the op runs.";

  let description = [{
Given a list of device names, this operation returns the index of the device
this op runs. The length of the list is returned in two cases:
(1) Device does not exist in the given device list.
(2) It is in XLA compilation.
  }];

  let arguments = (ins
    StrArrayAttr:$device_names
  );

  let results = (outs
    TF_Int32Tensor:$index
  );
}

def TF_DiagOp : TF_Op<"Diag", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
  let summary = "Returns a diagonal tensor with a given diagonal values.";

  let description = [{
Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

For example:

```
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{Rank k tensor where k is at most 1.}]>:$diagonal
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DiagPartOp : TF_Op<"DiagPart", [NoSideEffect]> {
  let summary = "Returns the diagonal part of the tensor.";

  let description = [{
This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

```
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{Rank k tensor where k is even and not zero.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The extracted diagonal.}]>:$diagonal
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DigammaOp : TF_Op<"Digamma", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes Psi, the derivative of Lgamma (the log of the absolute value of
  }];

  let description = [{
`Gamma(x)`), element-wise.
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DivOp : TF_Op<"Div", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_DivNoNanOp : TF_Op<"DivNoNan", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if the denominator is zero.";

  let description = [{
*NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x,
    TF_FpOrComplexTensor:$y
  );

  let results = (outs
    TF_FpOrComplexTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_DummyMemoryCacheOp : TF_Op<"DummyMemoryCache", []> {
  let summary = "";

  let arguments = (ins);

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetMemoryCacheAlloc]>:$handle
  );
}

def TF_DummySeedGeneratorOp : TF_Op<"DummySeedGenerator", []> {
  let summary = "";

  let arguments = (ins);

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorAlloc]>:$handle
  );
}

def TF_DynamicEnqueueTPUEmbeddingArbitraryTensorBatchOp : TF_Op<"DynamicEnqueueTPUEmbeddingArbitraryTensorBatch", [SameVariadicOperandSize, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
Eases the porting of code that uses tf.nn.embedding_lookup_sparse().
  }];

  let description = [{
embedding_indices[i] and aggregation_weights[i] correspond
to the ith feature.

The tensors at corresponding positions in the three input lists (sample_indices,
embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
with dim_size() equal to the total number of lookups into the table described by
the corresponding feature.
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 2 Tensors specifying the training example to which the
corresponding embedding_indices and aggregation_weights values belong.
If the size of its first dimension is 0, we assume each embedding_indices
belongs to a different sample. Both int32 and int64 are allowed and will
be converted to int32 internally.

Or a list of rank 1 Tensors specifying the row splits for splitting
embedding_indices and aggregation_weights into rows. It corresponds to
ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. When
enqueuing N-D ragged tensor, only the last dimension is allowed to be ragged.
the row splits is 1-D dense tensor. When empty, we assume a dense tensor is
passed to the op Both int32 and int64 are allowed and will be converted to
int32 internally.}]>:$sample_indices_or_row_splits,
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors, indices into the embedding
tables. Both int32 and int64 are allowed and will be converted to
int32 internally.}]>:$embedding_indices,
    Arg<Variadic<TF_F32OrF64Tensor>, [{A list of rank 1 Tensors containing per training
example aggregation weights. Both float32 and float64 are allowed and will
be converted to float32 internally.}]>:$aggregation_weights,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,
    Arg<TF_Int32Tensor, [{The TPU device to use. Should be >= 0 and less than the number
of TPU cores in the task on which the node is placed.}]>:$device_ordinal,

    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T1 = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T2 = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T3 = TF_DerivedOperandTypeAttr<2>;
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
    Variadic<TF_Int32Tensor>:$indices,
    Variadic<TF_Tensor>:$data
  );

  let results = (outs
    TF_Tensor:$merged
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
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
    Arg<Variadic<TF_Tensor>, [{List of 1 or 2 Tensors.}]>:$inputs,

    StrAttr:$equation
  );

  let results = (outs
    Res<TF_Tensor, [{Output Tensor with shape depending upon `equation`.}]>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_EluOp : TF_Op<"Elu", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_EluGradOp : TF_Op<"EluGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes gradients for the exponential linear (Elu) operation.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{The backpropagated gradients to the corresponding Elu operation.}]>:$gradients,
    Arg<TF_FloatTensor, [{The outputs of the corresponding Elu operation.}]>:$outputs
  );

  let results = (outs
    Res<TF_FloatTensor, [{The gradients: `gradients * (outputs + 1)` if outputs < 0,
`gradients` otherwise.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_EmptyOp : TF_Op<"Empty", []> {
  let summary = [{
Creates a tensor with the given shape.

This operation creates a tensor of `shape` and `dtype`.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{1-D. Represents the shape of the output tensor.}]>:$shape,

    DefaultValuedAttr<BoolAttr, "false">:$init
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type `T`.}]>:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let hasFolder = 1;
}

def TF_EnqueueTPUEmbeddingArbitraryTensorBatchOp : TF_Op<"EnqueueTPUEmbeddingArbitraryTensorBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, SameVariadicOperandSize, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
Eases the porting of code that uses tf.nn.embedding_lookup_sparse().
  }];

  let description = [{
embedding_indices[i] and aggregation_weights[i] correspond
to the ith feature.

The tensors at corresponding positions in the three input lists (sample_indices,
embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
with dim_size() equal to the total number of lookups into the table described by
the corresponding feature.
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 2 Tensors specifying the training example to which the
corresponding embedding_indices and aggregation_weights values belong.
If the size of its first dimension is 0, we assume each embedding_indices
belongs to a different sample. Both int32 and int64 are allowed and will
be converted to int32 internally.

Or a list of rank 1 Tensors specifying the row splits for splitting
embedding_indices and aggregation_weights into rows. It corresponds to
ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. When
enqueuing N-D ragged tensor, only the last dimension is allowed to be ragged.
the row splits is 1-D dense tensor. When empty, we assume a dense tensor is
passed to the op Both int32 and int64 are allowed and will be converted to
int32 internally.}]>:$sample_indices_or_row_splits,
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors, indices into the embedding
tables. Both int32 and int64 are allowed and will be converted to
int32 internally.}]>:$embedding_indices,
    Arg<Variadic<TF_F32OrF64Tensor>, [{A list of rank 1 Tensors containing per training
example aggregation weights. Both float32 and float64 are allowed and will
be converted to float32 internally.}]>:$aggregation_weights,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal,
    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T1 = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T2 = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T3 = TF_DerivedOperandTypeAttr<2>;
}

def TF_EnqueueTPUEmbeddingBatchOp : TF_Op<"EnqueueTPUEmbeddingBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
An op that enqueues a list of input batch tensors to TPUEmbedding.
  }];

  let description = [{
An op that enqueues a list of input batch tensors to TPUEmbedding.
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{A list of 1D tensors, one for each embedding table, containing the
batch inputs encoded as dist_belief.SparseFeatures protos. If the weight
field in the SparseFeatures proto is not populated for an ID, a weight of
1.0 is assumed.}]>:$batch,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal,
    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_EnqueueTPUEmbeddingIntegerBatchOp : TF_Op<"EnqueueTPUEmbeddingIntegerBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
An op that enqueues a list of input batch tensors to TPUEmbedding.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Int32Tensor>, [{A list of 1D tensors, one for each embedding table, containing the
indices into the tables.}]>:$batch,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_EnqueueTPUEmbeddingRaggedTensorBatchOp : TF_Op<"EnqueueTPUEmbeddingRaggedTensorBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, SameVariadicOperandSize, TF_TPUEmbeddingWriteEffect]> {
  let summary = "Eases the porting of code that uses tf.nn.embedding_lookup().";

  let description = [{
sample_splits[i], embedding_indices[i] and aggregation_weights[i] correspond
to the ith feature. table_ids[i] indicates which embedding table to look up ith
feature.

The tensors at corresponding positions in two of the input lists,
embedding_indices and aggregation_weights, must have the same shape, i.e. rank 1
with dim_size() equal to the total number of lookups into the table described by
the corresponding feature.
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors specifying the break points for splitting
embedding_indices and aggregation_weights into rows.
It corresponds to ids.row_splits in embedding_lookup(), when ids is a
RaggedTensor.}]>:$sample_splits,
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors, indices into the embedding tables.
It corresponds to ids.values in embedding_lookup(), when ids is a RaggedTensor.}]>:$embedding_indices,
    Arg<Variadic<TF_F32OrF64Tensor>, [{A list of rank 1 Tensors containing per training example
aggregation weights. It corresponds to the values field of a RaggedTensor
with the same row_splits as ids in embedding_lookup(), when ids is a
RaggedTensor.}]>:$aggregation_weights,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal,
    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners,
    I64ArrayAttr:$table_ids,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$max_sequence_lengths,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$num_features
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T1 = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T2 = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T3 = TF_DerivedOperandTypeAttr<2>;
}

def TF_EnqueueTPUEmbeddingSparseBatchOp : TF_Op<"EnqueueTPUEmbeddingSparseBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, SameVariadicOperandSize, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
An op that enqueues TPUEmbedding input indices from a SparseTensor.
  }];

  let description = [{
This Op eases the porting of code that uses embedding_lookup_sparse(),
although some Python preprocessing of the SparseTensor arguments to
embedding_lookup_sparse() is required to produce the arguments to this Op,
since only a single EnqueueTPUEmbeddingSparseBatch Op is allowed per training
step.

The tensors at corresponding positions in the three input lists
must have the same shape, i.e. rank 1 with dim_size() equal to the total
number of lookups into the table described by the corresponding table_id.
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors specifying the training example and
feature to which the corresponding embedding_indices and aggregation_weights
values belong. sample_indices[i] must equal b * nf + f, where nf is the
number of features from the corresponding table, f is in [0, nf), and
b is in [0, batch size).}]>:$sample_indices,
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors, indices into the embedding tables.}]>:$embedding_indices,
    Arg<Variadic<TF_F32OrF64Tensor>, [{A list of rank 1 Tensors containing per sample -- i.e. per
(training example, feature) -- aggregation weights.}]>:$aggregation_weights,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal,
    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T1 = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T2 = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T3 = TF_DerivedOperandTypeAttr<2>;
}

def TF_EnqueueTPUEmbeddingSparseTensorBatchOp : TF_Op<"EnqueueTPUEmbeddingSparseTensorBatch", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, SameVariadicOperandSize, TF_TPUEmbeddingWriteEffect]> {
  let summary = [{
Eases the porting of code that uses tf.nn.embedding_lookup_sparse().
  }];

  let description = [{
sample_indices[i], embedding_indices[i] and aggregation_weights[i] correspond
to the ith feature. table_ids[i] indicates which embedding table to look up ith
feature.

The tensors at corresponding positions in the three input lists (sample_indices,
embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
with dim_size() equal to the total number of lookups into the table described by
the corresponding feature.
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors specifying the training example to
which the corresponding embedding_indices and aggregation_weights values
belong. It corresponds to sp_ids.indices[:,0] in  embedding_lookup_sparse().}]>:$sample_indices,
    Arg<Variadic<TF_I32OrI64Tensor>, [{A list of rank 1 Tensors, indices into the embedding tables.
It corresponds to sp_ids.values in embedding_lookup_sparse().}]>:$embedding_indices,
    Arg<Variadic<TF_F32OrF64Tensor>, [{A list of rank 1 Tensors containing per training example
aggregation weights. It corresponds to sp_weights.values in
embedding_lookup_sparse().}]>:$aggregation_weights,
    Arg<TF_StrTensor, [{A string input that overrides the mode specified in the
TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
in TPUEmbeddingConfiguration is used, otherwise mode_override is used.}]>:$mode_override,

    DefaultValuedAttr<I64Attr, "-1">:$device_ordinal,
    DefaultValuedAttr<StrArrayAttr, "{}">:$combiners,
    I64ArrayAttr:$table_ids,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$max_sequence_lengths,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$num_features
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T1 = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T2 = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T3 = TF_DerivedOperandTypeAttr<2>;
}

def TF_EnsureShapeOp : TF_Op<"EnsureShape", [NoSideEffect]> {
  let summary = "Ensures that the tensor's shape matches the expected shape.";

  let description = [{
Raises an error if the input tensor's shape does not match the specified shape.
Returns the input tensor otherwise.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A tensor, whose shape is to be validated.}]>:$input,

    TF_ShapeAttr:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{A tensor with the same shape and contents as the input tensor or value.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
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
    TF_Tensor:$x,
    TF_Tensor:$y,

    DefaultValuedAttr<BoolAttr, "true">:$incompatible_shape_error
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

def TF_ErfOp : TF_Op<"Erf", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_ErfcOp : TF_Op<"Erfc", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_ErfinvOp : TF_Op<"Erfinv", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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

def TF_Expm1Op : TF_Op<"Expm1", [InferTensorType, NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

}

def TF_ExtractImagePatchesOp : TF_Op<"ExtractImagePatches", [NoSideEffect]> {
  let summary = [{
Extract `patches` from `images` and put them in the "depth" output dimension.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.}]>:$images,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksizes,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$rates,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
ksize_cols * depth]` containing image patches with size
`ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension. Note
`out_rows` and `out_cols` are the dimensions of the output patches.}]>:$patches
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FFTOp : TF_Op<"FFT", [NoSideEffect]> {
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

def TF_FFT2DOp : TF_Op<"FFT2D", [NoSideEffect]> {
  let summary = "2D fast Fourier transform.";

  let description = [{
Computes the 2-dimensional discrete Fourier transform over the inner-most
2 dimensions of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fft2
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_FFT3DOp : TF_Op<"FFT3D", [NoSideEffect]> {
  let summary = "3D fast Fourier transform.";

  let description = [{
Computes the 3-dimensional discrete Fourier transform over the inner-most 3
dimensions of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their 3D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fftn with 3 dimensions.
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_FakeParamOp : TF_Op<"FakeParam", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
  This op is used as a placeholder in If branch functions. It doesn't provide a
  valid output when run, so must either be removed (e.g. replaced with a
  function input) or guaranteed not to be used (e.g. if mirroring an
  intermediate output needed for the gradient computation of the other branch).
  }];

  let arguments = (ins
    TF_ShapeAttr:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{    \"Fake\" output value. This should not be consumed by another op.}]>:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_FakeQuantWithMinMaxArgsOp : TF_Op<"FakeQuantWithMinMaxArgs", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
  }];

  let description = [{
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

Quantization is called fake since the output is still in floating point.
  }];

  let arguments = (ins
    TF_Float32Tensor:$inputs,

    DefaultValuedAttr<F32Attr, "-6.0f">:$min,
    DefaultValuedAttr<F32Attr, "6.0f">:$max,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_Float32Tensor:$outputs
  );

  let hasVerifier = 1;
}

def TF_FakeQuantWithMinMaxArgsGradientOp : TF_Op<"FakeQuantWithMinMaxArgsGradient", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Compute gradients for a FakeQuantWithMinMaxArgs operation.";

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.}]>:$gradients,
    Arg<TF_Float32Tensor, [{Values passed as inputs to the FakeQuantWithMinMaxArgs operation.}]>:$inputs,

    DefaultValuedAttr<F32Attr, "-6.0f">:$min,
    DefaultValuedAttr<F32Attr, "6.0f">:$max,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
`gradients * (inputs >= min && inputs <= max)`.}]>:$backprops
  );
}

def TF_FakeQuantWithMinMaxVarsOp : TF_Op<"FakeQuantWithMinMaxVars", [NoSideEffect]> {
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
  }];

  let arguments = (ins
    TF_Float32Tensor:$inputs,
    TF_Float32Tensor:$min,
    TF_Float32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_Float32Tensor:$outputs
  );

  let hasVerifier = 1;
}

def TF_FakeQuantWithMinMaxVarsGradientOp : TF_Op<"FakeQuantWithMinMaxVarsGradient", [NoSideEffect]> {
  let summary = "Compute gradients for a FakeQuantWithMinMaxVars operation.";

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Backpropagated gradients above the FakeQuantWithMinMaxVars operation.}]>:$gradients,
    Arg<TF_Float32Tensor, [{Values passed as inputs to the FakeQuantWithMinMaxVars operation.
min, max: Quantization interval, scalar floats.}]>:$inputs,
    TF_Float32Tensor:$min,
    TF_Float32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. inputs:
`gradients * (inputs >= min && inputs <= max)`.}]>:$backprops_wrt_input,
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. min parameter:
`sum(gradients * (inputs < min))`.}]>:$backprop_wrt_min,
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. max parameter:
`sum(gradients * (inputs > max))`.}]>:$backprop_wrt_max
  );
}

def TF_FakeQuantWithMinMaxVarsPerChannelOp : TF_Op<"FakeQuantWithMinMaxVarsPerChannel", [NoSideEffect]> {
  let summary = [{
Fake-quantize the 'inputs' tensor of type float via per-channel floats
  }];

  let description = [{
Fake-quantize the `inputs` tensor of type float per-channel and one of the
shapes: `[d]`, `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max`
of shape `[d]` to `outputs` tensor of same shape as `inputs`.

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
  }];

  let arguments = (ins
    TF_Float32Tensor:$inputs,
    TF_Float32Tensor:$min,
    TF_Float32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_Float32Tensor:$outputs
  );

  let hasVerifier = 1;
}

def TF_FakeQuantWithMinMaxVarsPerChannelGradientOp : TF_Op<"FakeQuantWithMinMaxVarsPerChannelGradient", [NoSideEffect]> {
  let summary = [{
Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.}]>:$gradients,
    Arg<TF_Float32Tensor, [{Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
  same as `gradients`.
min, max: Quantization interval, floats of shape `[d]`.}]>:$inputs,
    TF_Float32Tensor:$min,
    TF_Float32Tensor:$max,

    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. inputs, shape same as
`inputs`:
  `gradients * (inputs >= min && inputs <= max)`.}]>:$backprops_wrt_input,
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. min parameter, shape `[d]`:
`sum_per_d(gradients * (inputs < min))`.}]>:$backprop_wrt_min,
    Res<TF_Float32Tensor, [{Backpropagated gradients w.r.t. max parameter, shape `[d]`:
`sum_per_d(gradients * (inputs > max))`.}]>:$backprop_wrt_max
  );
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

def TF_FinalizeDatasetOp : TF_Op<"FinalizeDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset by applying `tf.data.Options` to `input_dataset`.
  }];

  let arguments = (ins
    Arg<TF_VariantTensor, [{A variant tensor representing the input dataset.}]>:$input_dataset,

    DefaultValuedAttr<BoolAttr, "false">:$has_captured_ref,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_FlatMapDatasetOp : TF_Op<"FlatMapDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let description = [{
Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset variant, and FlatMapDataset will flatten successive results
into a single Dataset.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_FloorOp : TF_Op<"Floor", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns element-wise largest integer not greater than x.";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FloorDivOp : TF_Op<"FloorDiv", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_FloorModOp : TF_Op<"FloorMod", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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
    TF_IntOrFpTensor:$x,
    TF_IntOrFpTensor:$y
  );

  let results = (outs
    TF_IntOrFpTensor:$z
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
    Arg<TF_Float32Tensor, [{A 4D Tensor for input data.}]>:$x,
    Arg<TF_Float32Tensor, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TF_Float32Tensor, [{A 1D Tensor for offset, to shift to the normalized x.}]>:$offset,
    Arg<TF_Float32Tensor, [{A 1D Tensor for population mean. Used for inference only;
must be empty for training.}]>:$mean,
    Arg<TF_Float32Tensor, [{A 1D Tensor for population variance. Used for inference only;
must be empty for training.}]>:$variance,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0f">:$exponential_avg_factor,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TF_Float32Tensor, [{A 4D Tensor for output data.}]>:$y,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch mean, to be used by TensorFlow
to compute the running mean.}]>:$batch_mean,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch variance, to be used by
TensorFlow to compute the running variance.}]>:$batch_variance,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch mean, to be reused
in the gradient computation.}]>:$reserve_space_1,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch variance (inverted variance
in the cuDNN case), to be reused in the gradient computation.}]>:$reserve_space_2
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def TF_FusedBatchNormGradOp : TF_Op<"FusedBatchNormGrad", [NoSideEffect]> {
  let summary = "Gradient for batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{A 4D Tensor for the gradient with respect to y.}]>:$y_backprop,
    Arg<TF_Float32Tensor, [{A 4D Tensor for input data.}]>:$x,
    Arg<TF_Float32Tensor, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
mean to be reused in gradient computation. When is_training is
False, a 1D Tensor for the population mean to be reused in both
1st and 2nd order gradient computation.}]>:$reserve_space_1,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
variance (inverted variance in the cuDNN case) to be reused in
gradient computation. When is_training is False, a 1D Tensor
for the population variance to be reused in both 1st and 2nd
order gradient computation.}]>:$reserve_space_2,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TF_Float32Tensor, [{A 4D Tensor for the gradient with respect to x.}]>:$x_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to scale.}]>:$scale_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to offset.}]>:$offset_backprop,
    Res<TF_Float32Tensor, [{Unused placeholder to match the mean input in FusedBatchNorm.}]>:$reserve_space_3,
    Res<TF_Float32Tensor, [{Unused placeholder to match the variance input
in FusedBatchNorm.}]>:$reserve_space_4
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_FusedBatchNormGradV2Op : TF_Op<"FusedBatchNormGradV2", [NoSideEffect]> {
  let summary = "Gradient for batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for the gradient with respect to y.}]>:$y_backprop,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for input data.}]>:$x,
    Arg<TF_Float32Tensor, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
mean to be reused in gradient computation. When is_training is
False, a 1D Tensor for the population mean to be reused in both
1st and 2nd order gradient computation.}]>:$reserve_space_1,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
variance (inverted variance in the cuDNN case) to be reused in
gradient computation. When is_training is False, a 1D Tensor
for the population variance to be reused in both 1st and 2nd
order gradient computation.}]>:$reserve_space_2,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for the gradient with respect to x.}]>:$x_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to scale.}]>:$scale_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to offset.}]>:$offset_backprop,
    Res<TF_Float32Tensor, [{Unused placeholder to match the mean input in FusedBatchNorm.}]>:$reserve_space_3,
    Res<TF_Float32Tensor, [{Unused placeholder to match the variance input
in FusedBatchNorm.}]>:$reserve_space_4
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<3>;
}

def TF_FusedBatchNormGradV3Op : TF_Op<"FusedBatchNormGradV3", [NoSideEffect, TF_LayoutSensitiveInterface]> {
  let summary = "Gradient for batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for the gradient with respect to y.}]>:$y_backprop,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for input data.}]>:$x,
    Arg<TF_Float32Tensor, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
mean to be reused in gradient computation. When is_training is
False, a 1D Tensor for the population mean to be reused in both
1st and 2nd order gradient computation.}]>:$reserve_space_1,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for the computed batch
variance (inverted variance in the cuDNN case) to be reused in
gradient computation. When is_training is False, a 1D Tensor
for the population variance to be reused in both 1st and 2nd
order gradient computation.}]>:$reserve_space_2,
    Arg<TF_Float32Tensor, [{When is_training is True, a 1D Tensor for some intermediate results to be reused
in gradient computation. When is_training is False, a dummy empty Tensor will be
created.}]>:$reserve_space_3,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NDHWC", "NCDHW"]>, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for the gradient with respect to x.}]>:$x_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to scale.}]>:$scale_backprop,
    Res<TF_Float32Tensor, [{A 1D Tensor for the gradient with respect to offset.}]>:$offset_backprop,
    Res<TF_Float32Tensor, [{Unused placeholder to match the mean input in FusedBatchNorm.}]>:$reserve_space_4,
    Res<TF_Float32Tensor, [{Unused placeholder to match the variance input
in FusedBatchNorm.}]>:$reserve_space_5
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<3>;

  let extraClassDeclaration = [{
    // TF_LayoutSensitiveInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0, 1}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {0}; }
    StringRef GetOptimalLayout(const RuntimeDevices& devices);
    LogicalResult UpdateDataFormat(StringRef data_format);
  }];
}

def TF_FusedBatchNormV2Op : TF_Op<"FusedBatchNormV2", [NoSideEffect, TF_FoldOperandsTransposeInterface, TF_LayoutSensitiveInterface]> {
  let summary = "Batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for input data.}]>:$x,
    Arg<TF_Float32Tensor, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TF_Float32Tensor, [{A 1D Tensor for offset, to shift to the normalized x.}]>:$offset,
    Arg<TF_Float32Tensor, [{A 1D Tensor for population mean. Used for inference only;
must be empty for training.}]>:$mean,
    Arg<TF_Float32Tensor, [{A 1D Tensor for population variance. Used for inference only;
must be empty for training.}]>:$variance,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0f">:$exponential_avg_factor,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for output data.}]>:$y,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch mean, to be used by TensorFlow
to compute the running mean.}]>:$batch_mean,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch variance, to be used by
TensorFlow to compute the running variance.}]>:$batch_variance,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch mean, to be reused
in the gradient computation.}]>:$reserve_space_1,
    Res<TF_Float32Tensor, [{A 1D Tensor for the computed batch variance (inverted variance
in the cuDNN case), to be reused in the gradient computation.}]>:$reserve_space_2
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;

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

def TF_FusedBatchNormV3Op : TF_Op<"FusedBatchNormV3", [NoSideEffect, TF_FoldOperandsTransposeInterface, TF_LayoutSensitiveInterface]> {
  let summary = "Batch normalization.";

  let description = [{
Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for input data.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for scaling factor, to scale the normalized x.}]>:$scale,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for offset, to shift to the normalized x.}]>:$offset,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for population mean. Used for inference only;
must be empty for training.}]>:$mean,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for population variance. Used for inference only;
must be empty for training.}]>:$variance,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0f">:$exponential_avg_factor,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NDHWC", "NCDHW"]>, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{A 4D Tensor for output data.}]>:$y,
    Res<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for the computed batch mean, to be used by TensorFlow
to compute the running mean.}]>:$batch_mean,
    Res<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for the computed batch variance, to be used by
TensorFlow to compute the running variance.}]>:$batch_variance,
    Res<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for the computed batch mean, to be reused
in the gradient computation.}]>:$reserve_space_1,
    Res<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for the computed batch variance (inverted variance
in the cuDNN case), to be reused in the gradient computation.}]>:$reserve_space_2,
    Res<TensorOf<[TF_Bfloat16, TF_Float32]>, [{A 1D Tensor for some intermediate results, to be reused in the gradient
computation for better efficiency.}]>:$reserve_space_3
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;

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

  let hasCanonicalizer = 1;
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
    Arg<TF_Tensor, [{The tensor from which to gather values.}]>:$params,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Index tensor.}]>:$indices
  );

  let results = (outs
    Res<TF_Tensor, [{Values from `params` gathered from indices given by `indices`, with
shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.}]>:$output
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

See also `tf.batch_gather` and `tf.gather_nd`.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor from which to gather values. Must be at least rank
`axis + 1`.}]>:$params,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Index tensor. Must be in range `[0, params.shape[axis])`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{The axis in `params` to gather `indices` from. Defaults to the first
dimension. Supports negative indexes.}]>:$axis,

    DefaultValuedAttr<I64Attr, "0">:$batch_dims
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

def TF_GeneratorDatasetOp : TF_Op<"GeneratorDataset", [AttrSizedOperandSegments, TF_GeneratorOpSideEffect]> {
  let summary = [{
Creates a dataset that invokes a function to generate elements.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$init_func_other_args,
    Variadic<TF_Tensor>:$next_func_other_args,
    Variadic<TF_Tensor>:$finalize_func_other_args,

    SymbolRefAttr:$init_func,
    SymbolRefAttr:$next_func,
    SymbolRefAttr:$finalize_func,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Tfinalize_func_args = TF_DerivedOperandTypeListAttr<2>;
  TF_DerivedOperandTypeListAttr Tinit_func_args = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedOperandTypeListAttr Tnext_func_args = TF_DerivedOperandTypeListAttr<1>;
}

def TF_GreaterOp : TF_Op<"Greater", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_GreaterEqualOp : TF_Op<"GreaterEqual", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_HSVToRGBOp : TF_Op<"HSVToRGB", [NoSideEffect]> {
  let summary = "Convert one or more images from HSV to RGB.";

  let description = [{
Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{1-D or higher rank. HSV data to convert. Last dimension must be size 3.}]>:$images
  );

  let results = (outs
    Res<TF_FloatTensor, [{`images` converted to RGB.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_HashTableOp : TF_Op<"HashTable", []> {
  let summary = "Creates a non-initialized hash table.";

  let description = [{
This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype
  );

  let results = (outs
    Res<TF_StrTensor, [{Handle to a table.}], [TF_LookupTableAlloc]>:$table_handle
  );

  let hasCanonicalizer = 1;
}

def TF_HashTableV2Op : TF_Op<"HashTableV2", []> {
  let summary = "Creates a non-initialized hash table.";

  let description = [{
This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype
  );

  let results = (outs
    Res<TF_ResourceTensor, [{Handle to a table.}], [TF_LookupTableAlloc]>:$table_handle
  );

  let builders = [
    OpBuilder<(ins "StringAttr":$container, "StringAttr":$shared_name,
      "BoolAttr":$use_node_name_sharing, "TypeAttr":$key_dtype, "TypeAttr":$value_dtype),
    [{
      build($_builder, $_state,
      mlir::RankedTensorType::get({},
      $_builder.getType<mlir::TF::ResourceType>()),
      container, shared_name, use_node_name_sharing, key_dtype, value_dtype);
    }]>];
}

def TF_IFFTOp : TF_Op<"IFFT", [NoSideEffect]> {
  let summary = "Inverse fast Fourier transform.";

  let description = [{
Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its inverse 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifft
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_IFFT2DOp : TF_Op<"IFFT2D", [NoSideEffect]> {
  let summary = "Inverse 2D fast Fourier transform.";

  let description = [{
Computes the inverse 2-dimensional discrete Fourier transform over the
inner-most 2 dimensions of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their inverse 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifft2
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_IFFT3DOp : TF_Op<"IFFT3D", [NoSideEffect]> {
  let summary = "Inverse 3D fast Fourier transform.";

  let description = [{
Computes the inverse 3-dimensional discrete Fourier transform over the
inner-most 3 dimensions of `input`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their inverse 3D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifftn with 3 dimensions.
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
}

def TF_IRFFTOp : TF_Op<"IRFFT", [NoSideEffect]> {
  let summary = "Inverse real-valued fast Fourier transform.";

  let description = [{
Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
signal over the inner-most dimension of `input`.

The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
`fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
`fft_length` is not provided, it is computed from the size of the inner-most
dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
than the corresponding dimension of `input`, the dimension is cropped. If it is
larger, the dimension is padded with zeros.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [1]. The FFT length.}]>:$fft_length
  );

  let results = (outs
    Res<TF_F32OrF64Tensor, [{A float32 tensor of the same rank as `input`. The inner-most
  dimension of `input` is replaced with the `fft_length` samples of its inverse
  1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.irfft
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Treal = TF_DerivedResultTypeAttr<0>;
}

def TF_IRFFT2DOp : TF_Op<"IRFFT2D", [NoSideEffect]> {
  let summary = "Inverse 2D real-valued fast Fourier transform.";

  let description = [{
Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 2 dimensions of `input`.

The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 2 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along each axis `IRFFT2D` is computed on, if `fft_length` (or
`fft_length / 2 + 1` for the inner-most dimension) is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [2]. The FFT length for each dimension.}]>:$fft_length
  );

  let results = (outs
    Res<TF_F32OrF64Tensor, [{A float32 tensor of the same rank as `input`. The inner-most 2
  dimensions of `input` are replaced with the `fft_length` samples of their
  inverse 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.irfft2
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Treal = TF_DerivedResultTypeAttr<0>;
}

def TF_IRFFT3DOp : TF_Op<"IRFFT3D", [NoSideEffect]> {
  let summary = "Inverse 3D real-valued fast Fourier transform.";

  let description = [{
Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 3 dimensions of `input`.

The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 3 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along each axis `IRFFT3D` is computed on, if `fft_length` (or
`fft_length / 2 + 1` for the inner-most dimension) is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [3]. The FFT length for each dimension.}]>:$fft_length
  );

  let results = (outs
    Res<TF_F32OrF64Tensor, [{A float32 tensor of the same rank as `input`. The inner-most 3
  dimensions of `input` are replaced with the `fft_length` samples of their
  inverse 3D real Fourier transform.

@compatibility(numpy)
Equivalent to np.irfftn with 3 dimensions.
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Tcomplex = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Treal = TF_DerivedResultTypeAttr<0>;
}

def TF_IdentityOp : TF_Op<"Identity", [NoSideEffect, TF_NoConstantFold, TF_OperandsSameAsResultsTypeOrRef]> {
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

def TF_IgammaOp : TF_Op<"Igamma", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                  WithBroadcastableBinOpBuilder {
  let summary = [{
Compute the lower regularized incomplete Gamma function `P(a, x)`.
  }];

  let description = [{
The lower regularized incomplete Gamma function is defined as:


\\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)

where

\\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)

is the lower incomplete Gamma function.

Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
Gamma function.
  }];

  let arguments = (ins
    TF_FloatTensor:$a,
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IgammaGradAOp : TF_Op<"IgammaGradA", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                       WithBroadcastableBinOpBuilder {
  let summary = "Computes the gradient of `igamma(a, x)` wrt `a`.";

  let arguments = (ins
    TF_F32OrF64Tensor:$a,
    TF_F32OrF64Tensor:$x
  );

  let results = (outs
    TF_F32OrF64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IgammacOp : TF_Op<"Igammac", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                   WithBroadcastableBinOpBuilder {
  let summary = [{
Compute the upper regularized incomplete Gamma function `Q(a, x)`.
  }];

  let description = [{
The upper regularized incomplete Gamma function is defined as:

\\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)

where

\\(Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt\\)

is the upper incomplete Gamma function.

Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function.
  }];

  let arguments = (ins
    TF_FloatTensor:$a,
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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

def TF_InTopKV2Op : TF_Op<"InTopKV2", [NoSideEffect]> {
  let summary = "Says whether the targets are in the top `K` predictions.";

  let description = [{
This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

  \\(predictions_i\\) be the predictions for all classes for example `i`,
  \\(targets_i\\) be the target class for example `i`,
  \\(out_i\\) be the output for example `i`,

$$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{A `batch_size` x `classes` tensor.}]>:$predictions,
    Arg<TF_I32OrI64Tensor, [{A `batch_size` vector of class ids.}]>:$targets,
    Arg<TF_I32OrI64Tensor, [{Number of top elements to look at for computing precision.}]>:$k
  );

  let results = (outs
    Res<TF_BoolTensor, [{Computed precision at `k` as a `bool Tensor`.}]>:$precision
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_InfeedDequeueOp : TF_Op<"InfeedDequeue", []> {
  let summary = [{
A placeholder op for a value that will be fed into the computation.
  }];

  let arguments = (ins
    TF_ShapeAttr:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{A tensor that will be provided using the infeed mechanism.}]>:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_InitializeTableOp : TF_Op<"InitializeTable", []> {
  let summary = [{
Table initializer that takes two tensors for keys and values respectively.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Handle to a table which will be initialized.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_Tensor, [{Keys of type Tkey.}]>:$keys,
    Arg<TF_Tensor, [{Values of type Tval.}]>:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tkey = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tval = TF_DerivedOperandTypeAttr<2>;
}

def TF_InitializeTableFromDatasetOp : TF_Op<"InitializeTableFromDataset", []> {
  let summary = "";

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_LookupTableWrite]>:$table_handle,
    TF_VariantTensor:$dataset
  );

  let results = (outs);
}

def TF_InitializeTableFromTextFileOp : TF_Op<"InitializeTableFromTextFile", []> {
  let summary = "Initializes a table from a text file.";

  let description = [{
It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Handle to a table which will be initialized.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_StrTensor, [{Filename of a vocabulary text file.}]>:$filename,

    Confined<I64Attr, [IntMinValue<-2>]>:$key_index,
    Confined<I64Attr, [IntMinValue<-2>]>:$value_index,
    Confined<DefaultValuedAttr<I64Attr, "-1">, [IntMinValue<-1>]>:$vocab_size,
    DefaultValuedAttr<StrAttr, "\"\\t\"">:$delimiter,
    DefaultValuedAttr<I64Attr, "0">:$offset
  );

  let results = (outs);
}

def TF_InitializeTableFromTextFileV2Op : TF_Op<"InitializeTableFromTextFileV2", []> {
  let summary = "Initializes a table from a text file.";

  let description = [{
It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to a table which will be initialized.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_StrTensor, [{Filename of a vocabulary text file.}]>:$filename,

    Confined<I64Attr, [IntMinValue<-2>]>:$key_index,
    Confined<I64Attr, [IntMinValue<-2>]>:$value_index,
    Confined<DefaultValuedAttr<I64Attr, "-1">, [IntMinValue<-1>]>:$vocab_size,
    DefaultValuedAttr<StrAttr, "\"\\t\"">:$delimiter,
    DefaultValuedAttr<I64Attr, "0">:$offset
  );

  let results = (outs);
}

def TF_InitializeTableV2Op : TF_Op<"InitializeTableV2", []> {
  let summary = [{
Table initializer that takes two tensors for keys and values respectively.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to a table which will be initialized.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_Tensor, [{Keys of type Tkey.}]>:$keys,
    Arg<TF_Tensor, [{Values of type Tval.}]>:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tkey = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tval = TF_DerivedOperandTypeAttr<2>;
}

def TF_InplaceAddOp : TF_Op<"InplaceAdd", [NoSideEffect, TF_AllTypesMatch<["x", "y"]>]> {
  let summary = "Adds v into specified rows of x.";

  let description = [{
Computes y = x; y[i, :] += v; return y.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A `Tensor` of type T.}]>:$x,
    Arg<TF_Int32Tensor, [{A vector. Indices into the left-most dimension of `x`.}]>:$i,
    Arg<TF_Tensor, [{A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.}]>:$v
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_InplaceUpdateOp : TF_Op<"InplaceUpdate", [NoSideEffect]> {
  let summary = "Updates specified rows 'i' with values 'v'.";

  let description = [{
Computes `x[i, :] = v; return x`.

Originally this function is mutative however for compilation we make this
operation create / operate on a copy of `x`.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A tensor of type `T`.}]>:$x,
    Arg<TF_Int32Tensor, [{A vector. Indices into the left-most dimension of `x`.}]>:$i,
    Arg<TF_Tensor, [{A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.}]>:$v
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_InvOp : TF_Op<"Inv", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the reciprocal of x element-wise.";

  let description = [{
I.e., \\(y = 1 / x\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_InvertOp : TF_Op<"Invert", [Involution, NoSideEffect, SameOperandsAndResultType]> {
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
}

def TF_InvertPermutationOp : TF_Op<"InvertPermutation", [NoSideEffect]> {
  let summary = "Computes the inverse permutation of a tensor.";

  let description = [{
This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:

`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

The values must include 0. There can be no duplicate values or negative values.

For example:

```
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
```
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{1-D.}]>:$x
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{1-D.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
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
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_BoolTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IsInfOp : TF_Op<"IsInf", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns which elements of x are Inf.";

  let description = [{
@compatibility(numpy)
Equivalent to np.isinf
@end_compatibility

Example:

```python
x = tf.constant([5.0, np.inf, 6.8, np.inf])
tf.math.is_inf(x) ==> [False, True, False, True]
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

def TF_IsNanOp : TF_Op<"IsNan", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Returns which elements of x are NaN.";

  let description = [{
@compatibility(numpy)
Equivalent to np.isnan
@end_compatibility

Example:

```python
x = tf.constant([5.0, np.nan, 6.8, np.nan, np.inf])
tf.math.is_nan(x) ==> [False, True, False, True, False]
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

def TF_IteratorOp : TF_Op<"Iterator", []> {
  let summary = "A container for an iterator resource.";

  let arguments = (ins
    StrAttr:$shared_name,
    StrAttr:$container,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to the iterator that can be passed to a "MakeIterator"
or "IteratorGetNext" op.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_IteratorFromStringHandleOp : TF_Op<"IteratorFromStringHandle", []> {
  let summary = [{
Converts the given string representing a handle to an iterator to a resource.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{A string representation of the given handle.}]>:$string_handle,

    DefaultValuedAttr<TypeArrayAttr, "{}">:$output_types,
    DefaultValuedAttr<TF_ShapeAttrArray, "{}">:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to an iterator resource.}], [TF_DatasetIteratorAlloc]>:$resource_handle
  );
}

def TF_IteratorFromStringHandleV2Op : TF_Op<"IteratorFromStringHandleV2", []> {
  let summary = "";

  let arguments = (ins
    TF_StrTensor:$string_handle,

    DefaultValuedAttr<TypeArrayAttr, "{}">:$output_types,
    DefaultValuedAttr<TF_ShapeAttrArray, "{}">:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetIteratorAlloc]>:$resource_handle
  );
}

def TF_IteratorGetNextOp : TF_Op<"IteratorGetNext", []> {
  let summary = "Gets the next output from the given iterator .";

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetIteratorRead, TF_DatasetIteratorWrite]>:$iterator
  );

  let results = (outs
    Variadic<TF_Tensor>:$components
  );

  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr output_types = TF_DerivedResultTypeListAttr<0>;
}

def TF_IteratorGetNextAsOptionalOp : TF_Op<"IteratorGetNextAsOptional", []> {
  let summary = [{
Gets the next output from the given iterator as an Optional variant.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetIteratorRead, TF_DatasetIteratorWrite]>:$iterator,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$optional
  );
}

def TF_IteratorGetNextSyncOp : TF_Op<"IteratorGetNextSync", []> {
  let summary = "Gets the next output from the given iterator.";

  let description = [{
This operation is a synchronous version IteratorGetNext. It should only be used
in situations where the iterator does not block the calling thread, or where
the calling thread is not a member of the thread pool used to execute parallel
operations (e.g. in eager mode).
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_DatasetIteratorRead, TF_DatasetIteratorWrite]>:$iterator
  );

  let results = (outs
    Variadic<TF_Tensor>:$components
  );

  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr output_types = TF_DerivedResultTypeListAttr<0>;
}

def TF_IteratorToStringHandleOp : TF_Op<"IteratorToStringHandle", []> {
  let summary = [{
Converts the given `resource_handle` representing an iterator to a string.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A handle to an iterator resource.}], [TF_DatasetIteratorRead]>:$resource_handle
  );

  let results = (outs
    Res<TF_StrTensor, [{A string representation of the given handle.}]>:$string_handle
  );
}

def TF_IteratorV2Op : TF_Op<"IteratorV2", []> {
  let summary = "";

  let arguments = (ins
    StrAttr:$shared_name,
    StrAttr:$container,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_KthOrderStatisticOp : TF_Op<"KthOrderStatistic", [NoSideEffect]> {
  let summary = "Computes the Kth order statistic of a data set. The current";

  let description = [{
implementation uses a binary search requiring exactly 32 passes over
the input data. The running time is linear with respect to input
size. The median-of-medians algorithm is probably faster, but is
difficult to implement efficiently in XLA. The implementation imposes
a total ordering on floats. The ordering is consistent with the usual
partial order.  Positive NaNs are greater than positive
infinity. Negative NaNs are less than negative infinity. NaNs with
distinct payloads are treated as distinct. Subnormal numbers are
preserved (not flushed to zero). Positive infinity is greater than all
numbers. Negative infinity is less than all numbers. Positive is
greater than negative zero. There are less than k values greater than
the kth order statistic. There are at least k values greater than or
equal to the Kth order statistic. The semantics are not the same as
top_k_unique.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input,

    I64Attr:$k
  );

  let results = (outs
    TF_Float32Tensor:$output
  );
}

def TF_L2LossOp : TF_Op<"L2Loss", [NoSideEffect]> {
  let summary = "L2 Loss.";

  let description = [{
Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{Typically 2-D, but may have any dimensions.}]>:$t
  );

  let results = (outs
    Res<TF_FloatTensor, [{0-D.}]>:$output
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{4-D.}]>:$input,

    DefaultValuedAttr<I64Attr, "5">:$depth_radius,
    DefaultValuedAttr<F32Attr, "1.0f">:$bias,
    DefaultValuedAttr<F32Attr, "1.0f">:$alpha,
    DefaultValuedAttr<F32Attr, "0.5f">:$beta
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LRNGradOp : TF_Op<"LRNGrad", [NoSideEffect]> {
  let summary = "Gradients for Local Response Normalization.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$input_grads,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$input_image,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$output_image,

    DefaultValuedAttr<I64Attr, "5">:$depth_radius,
    DefaultValuedAttr<F32Attr, "1.0f">:$bias,
    DefaultValuedAttr<F32Attr, "1.0f">:$alpha,
    DefaultValuedAttr<F32Attr, "0.5f">:$beta
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{The gradients for LRN.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LeakyReluOp : TF_Op<"LeakyRelu", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear: `max(features, features * alpha)`.";

  let arguments = (ins
    TF_FloatTensor:$features,

    DefaultValuedAttr<F32Attr, "0.2f">:$alpha
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasFolder = 1;
}

def TF_LeakyReluGradOp : TF_Op<"LeakyReluGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes rectified linear gradients for a LeakyRelu operation.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{The backpropagated gradients to the corresponding LeakyRelu operation.}]>:$gradients,
    Arg<TF_FloatTensor, [{The features passed as input to the corresponding LeakyRelu operation,
OR the outputs of that operation (both work equivalently).}]>:$features,

    DefaultValuedAttr<F32Attr, "0.2f">:$alpha
  );

  let results = (outs
    Res<TF_FloatTensor, [{`gradients * (features > 0) + alpha * gradients * (features <= 0)`.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LeftShiftOp : TF_Op<"LeftShift", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_LessOp : TF_Op<"Less", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_LessEqualOp : TF_Op<"LessEqual", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_LgammaOp : TF_Op<"Lgamma", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes the log of the absolute value of `Gamma(x)` element-wise.
  }];

  let description = [{
For positive numbers, this function computes log((input - 1)!) for every element in the tensor.
  `lgamma(5) = log((5-1)!) = log(4!) = log(24) = 3.1780539`

Example:

```python
x = tf.constant([0, 0.5, 1, 4.5, -4, -5.6])
tf.math.lgamma(x) ==> [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
```
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LinSpaceOp : TF_Op<"LinSpace", [NoSideEffect]> {
  let summary = "Generates values in an interval.";

  let description = [{
A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{0-D tensor. First entry in the range.}]>:$start,
    Arg<TF_FloatTensor, [{0-D tensor. Last entry in the range.}]>:$stop,
    Arg<TF_I32OrI64Tensor, [{0-D tensor. Number of values to generate.}]>:$num
  );

  let results = (outs
    Res<TF_FloatTensor, [{1-D. The generated values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<2>;
}

def TF_ListDiffOp : TF_Op<"ListDiff", [NoSideEffect]> {
  let summary = [{
Computes the difference between two lists of numbers or strings.
  }];

  let description = [{
Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

```
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

```
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{1-D. Values to keep.}]>:$x,
    Arg<TF_Tensor, [{1-D. Values to remove.}]>:$y
  );

  let results = (outs
    Res<TF_Tensor, [{1-D. Values present in `x` but not in `y`.}]>:$out,
    Res<TF_I32OrI64Tensor, [{1-D. Positions of `x` values preserved in `out`.}]>:$idx
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_idx = TF_DerivedResultTypeAttr<1>;
}

def TF_LoadTPUEmbeddingADAMParametersOp : TF_Op<"LoadTPUEmbeddingADAMParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load ADAM embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the ADAM optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of momenta used in the ADAM optimization algorithm.}]>:$momenta,
    Arg<TF_Float32Tensor, [{Value of velocities used in the ADAM optimization algorithm.}]>:$velocities,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingADAMParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingADAMParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$momenta,
    TF_Float32Tensor:$velocities,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingAdadeltaParametersOp : TF_Op<"LoadTPUEmbeddingAdadeltaParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load Adadelta embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the Adadelta optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of accumulators used in the Adadelta optimization algorithm.}]>:$accumulators,
    Arg<TF_Float32Tensor, [{Value of updates used in the Adadelta optimization algorithm.}]>:$updates,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingAdadeltaParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingAdadeltaParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$updates,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingAdagradParametersOp : TF_Op<"LoadTPUEmbeddingAdagradParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load Adagrad embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the Adagrad optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of accumulators used in the Adagrad optimization algorithm.}]>:$accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingAdagradParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingAdagradParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingCenteredRMSPropParametersOp : TF_Op<"LoadTPUEmbeddingCenteredRMSPropParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load centered RMSProp embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the centered RMSProp optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of ms used in the centered RMSProp optimization algorithm.}]>:$ms,
    Arg<TF_Float32Tensor, [{Value of mom used in the centered RMSProp optimization algorithm.}]>:$mom,
    Arg<TF_Float32Tensor, [{Value of mg used in the centered RMSProp optimization algorithm.}]>:$mg,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingFTRLParametersOp : TF_Op<"LoadTPUEmbeddingFTRLParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load FTRL embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the FTRL optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of accumulators used in the FTRL optimization algorithm.}]>:$accumulators,
    Arg<TF_Float32Tensor, [{Value of linears used in the FTRL optimization algorithm.}]>:$linears,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingFTRLParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingFTRLParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$linears,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingMDLAdagradLightParametersOp : TF_Op<"LoadTPUEmbeddingMDLAdagradLightParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load MDL Adagrad Light embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the MDL Adagrad Light optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of accumulators used in the MDL Adagrad Light optimization algorithm.}]>:$accumulators,
    Arg<TF_Float32Tensor, [{Value of weights used in the MDL Adagrad Light optimization algorithm.}]>:$weights,
    Arg<TF_Float32Tensor, [{Value of benefits used in the MDL Adagrad Light optimization algorithm.}]>:$benefits,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingMomentumParametersOp : TF_Op<"LoadTPUEmbeddingMomentumParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load Momentum embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the Momentum optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of momenta used in the Momentum optimization algorithm.}]>:$momenta,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingMomentumParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingMomentumParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$momenta,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingProximalAdagradParametersOp : TF_Op<"LoadTPUEmbeddingProximalAdagradParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load proximal Adagrad embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the proximal Adagrad optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of accumulators used in the proximal Adagrad optimization algorithm.}]>:$accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingProximalAdagradParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingProximalAdagradParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingProximalYogiParametersOp : TF_Op<"LoadTPUEmbeddingProximalYogiParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$v,
    TF_Float32Tensor:$m,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingProximalYogiParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingProximalYogiParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$v,
    TF_Float32Tensor:$m,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingRMSPropParametersOp : TF_Op<"LoadTPUEmbeddingRMSPropParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load RMSProp embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the RMSProp optimization algorithm.}]>:$parameters,
    Arg<TF_Float32Tensor, [{Value of ms used in the RMSProp optimization algorithm.}]>:$ms,
    Arg<TF_Float32Tensor, [{Value of mom used in the RMSProp optimization algorithm.}]>:$mom,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingRMSPropParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingRMSPropParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$ms,
    TF_Float32Tensor:$mom,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingStochasticGradientDescentParametersOp : TF_Op<"LoadTPUEmbeddingStochasticGradientDescentParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Load SGD embedding parameters.";

  let description = [{
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{Value of parameters used in the stochastic gradient descent optimization algorithm.}]>:$parameters,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
}

def TF_LoadTPUEmbeddingStochasticGradientDescentParametersGradAccumDebugOp : TF_Op<"LoadTPUEmbeddingStochasticGradientDescentParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$gradient_accumulators,

    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs);
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

def TF_Log1pOp : TF_Op<"Log1p", [NoSideEffect, SameOperandsAndResultType, TF_CwiseUnary]> {
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
}

def TF_LogSoftmaxOp : TF_Op<"LogSoftmax", [NoSideEffect, SameOperandsAndResultType]> {
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
}

def TF_LogicalAndOp : TF_Op<"LogicalAnd", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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
}

def TF_LogicalNotOp : TF_Op<"LogicalNot", [Involution, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns the truth value of `NOT x` element-wise.";

  let arguments = (ins
    Arg<TF_BoolTensor, [{A `Tensor` of type `bool`.}]>:$x
  );

  let results = (outs
    Res<TF_BoolTensor, [{A `Tensor` of type `bool` with the same shape as `x`. The logical negation of `x`.}]>:$y
  );

  let hasCanonicalizer = 1;
}

def TF_LogicalOrOp : TF_Op<"LogicalOr", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_LookupTableExportV2Op : TF_Op<"LookupTableExportV2", []> {
  let summary = "Outputs all keys and values in the table.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableRead]>:$table_handle
  );

  let results = (outs
    Res<TF_Tensor, [{Vector of all keys present in the table.}]>:$keys,
    Res<TF_Tensor, [{Tensor of all values in the table. Indexed in parallel with `keys`.}]>:$values
  );

  TF_DerivedResultTypeAttr Tkeys = TF_DerivedResultTypeAttr<0>;
  TF_DerivedResultTypeAttr Tvalues = TF_DerivedResultTypeAttr<1>;
}

def TF_LookupTableFindOp : TF_Op<"LookupTableFind", []> {
  let summary = "Looks up keys in a table, outputs the corresponding values.";

  let description = [{
The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Handle to the table.}], [TF_LookupTableRead]>:$table_handle,
    Arg<TF_Tensor, [{Any shape.  Keys to look up.}]>:$keys,
    TF_Tensor:$default_value
  );

  let results = (outs
    Res<TF_Tensor, [{Same shape as `keys`.  Values found in the table, or `default_values`
for missing keys.}]>:$values
  );

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<2>;
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
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableRead]>:$table_handle,
    Arg<TF_Tensor, [{Any shape.  Keys to look up.}]>:$keys,
    TF_Tensor:$default_value
  );

  let results = (outs
    Res<TF_Tensor, [{Same shape as `keys`.  Values found in the table, or `default_values`
for missing keys.}]>:$values
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
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_Tensor, [{Any shape.  Keys to look up.}]>:$keys,
    Arg<TF_Tensor, [{Values to associate with keys.}]>:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<2>;
}

def TF_LookupTableInsertV2Op : TF_Op<"LookupTableInsertV2", []> {
  let summary = "Updates the table to associates keys with values.";

  let description = [{
The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_Tensor, [{Any shape.  Keys to look up.}]>:$keys,
    Arg<TF_Tensor, [{Values to associate with keys.}]>:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<2>;
}

def TF_LookupTableRemoveV2Op : TF_Op<"LookupTableRemoveV2", []> {
  let summary = "Removes keys and its associated values from a table.";

  let description = [{
The tensor `keys` must of the same type as the keys of the table. Keys not
already in the table are silently ignored.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableWrite]>:$table_handle,
    Arg<TF_Tensor, [{Any shape.  Keys of the elements to remove.}]>:$keys
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tin = TF_DerivedOperandTypeAttr<1>;
}

def TF_LookupTableSizeOp : TF_Op<"LookupTableSize", []> {
  let summary = "Computes the number of elements in the given table.";

  let arguments = (ins
    Arg<TF_StrTensor, [{Handle to the table.}], [TF_LookupTableRead]>:$table_handle
  );

  let results = (outs
    Res<TF_Int64Tensor, [{Scalar that contains number of elements in the table.}]>:$size
  );
}

def TF_LookupTableSizeV2Op : TF_Op<"LookupTableSizeV2", []> {
  let summary = "Computes the number of elements in the given table.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Handle to the table.}], [TF_LookupTableRead]>:$table_handle
  );

  let results = (outs
    Res<TF_Int64Tensor, [{Scalar that contains number of elements in the table.}]>:$size
  );
}

def TF_LowerBoundOp : TF_Op<"LowerBound", [NoSideEffect]> {
  let summary = [{
Applies lower_bound(sorted_search_values, values) along each row.
  }];

  let description = [{
Each set of rows with the same index in (sorted_inputs, values) is treated
independently.  The resulting row is the equivalent of calling
`np.searchsorted(sorted_inputs, values, side='left')`.

The result is not a global index to the entire
`Tensor`, but rather just the index in the last dimension.

A 2-D example:
  sorted_sequence = [[0, 3, 9, 9, 10],
                     [1, 2, 3, 4, 5]]
  values = [[2, 4, 9],
            [0, 2, 6]]

  result = LowerBound(sorted_sequence, values)

  result == [[1, 2, 2],
             [0, 1, 5]]
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{2-D Tensor where each row is ordered.}]>:$sorted_inputs,
    Arg<TF_Tensor, [{2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
the values that will be searched for in `sorted_search_values`.}]>:$values
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{A `Tensor` with the same shape as `values`.  It contains the first scalar index
into the last dimension where values can be inserted without changing the
ordered property.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;
}

def TF_MakeIteratorOp : TF_Op<"MakeIterator", []> {
  let summary = [{
Makes a new iterator from the given `dataset` and stores it in `iterator`.
  }];

  let description = [{
This operation may be executed multiple times. Each execution will reset the
iterator in `iterator` to the first element of `dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$dataset,
    Arg<TF_ResourceTensor, "", [TF_DatasetIteratorWrite]>:$iterator
  );

  let results = (outs);
}

def TF_MakeUniqueOp : TF_Op<"MakeUnique", [NoSideEffect]> {
  let summary = [{
Make all elements in the non-Batch dimension unique, but \"close\" to
  }];

  let description = [{
their initial value. Never returns a sub-normal number. Never returns
zero. The sign of each input element is always identical to the sign
of the corresponding output element. Behavior for infinite elements is
undefined. Behavior for subnormal elements is undefined.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input
  );

  let results = (outs
    TF_Float32Tensor:$output
  );
}

def TF_MapAndBatchDatasetOp : TF_Op<"MapAndBatchDataset", [NoSideEffect]> {
  let summary = "Creates a dataset that fuses mapping with batching.";

  let description = [{
Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.
  }];

  let arguments = (ins
    Arg<TF_VariantTensor, [{A variant tensor representing the input dataset.}]>:$input_dataset,
    Arg<Variadic<TF_Tensor>, [{A list of tensors, typically values that were captured when building a closure
for `f`.}]>:$other_arguments,
    Arg<TF_Int64Tensor, [{A scalar representing the number of elements to accumulate in a
batch. It determines the number of concurrent invocations of `f` that process
elements from `input_dataset` in parallel.}]>:$batch_size,
    Arg<TF_Int64Tensor, [{A scalar representing the maximum number of parallel invocations of the `map_fn`
function. Applying the `map_fn` on consecutive input elements in parallel has
the potential to improve input pipeline throughput.}]>:$num_parallel_calls,
    Arg<TF_BoolTensor, [{A scalar representing whether the last batch should be dropped in case its size
is smaller than desired.}]>:$drop_remainder,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_MapDatasetOp : TF_Op<"MapDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_MatMulOp : TF_Op<"MatMul", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$a,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$b,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixBandPartOp : TF_Op<"MatrixBandPart", [NoSideEffect, TF_AllTypesMatch<["input", "band"]>]> {
  let summary = [{
Copy a tensor setting everything outside a central band in each innermost matrix to zero.
  }];

  let description = [{
The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function

`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                 (num_upper < 0 || (n-m) <= num_upper)`.

For example:

```
# if 'input' is [[ 0,  1,  2, 3]
#                [-1,  0,  1, 2]
#                [-2, -1,  0, 1]
#                [-3, -2, -1, 0]],

tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                       [-1,  0,  1, 2]
                                       [ 0, -1,  0, 1]
                                       [ 0,  0, -1, 0]],

tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                      [-1,  0,  1, 0]
                                      [-2, -1,  0, 1]
                                      [ 0, -2, -1, 0]]
```

Useful special cases:

```
 tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
 tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
 tf.linalg.band_part(input, 0, 0) ==> Diagonal.
```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Rank `k` tensor.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{0-D tensor. Number of subdiagonals to keep. If negative, keep entire
lower triangle.}]>:$num_lower,
    Arg<TF_I32OrI64Tensor, [{0-D tensor. Number of superdiagonals to keep. If negative, keep
entire upper triangle.}]>:$num_upper
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `k` tensor of the same shape as input. The extracted banded tensor.}]>:$band
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindex = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
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
    Arg<TF_Tensor, [{Rank `k`, where `k >= 1`.}]>:$diagonal
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MatrixDiagPartV3Op : TF_Op<"MatrixDiagPartV3", [NoSideEffect]> {
  let summary = "Returns the batched diagonal part of a batched tensor.";

  let description = [{
Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
`input`.

Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
Let `max_diag_len` be the maximum length among all diagonals to be extracted,
`max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
Let `num_diags` be the number of diagonals to extract,
`num_diags = k[1] - k[0] + 1`.

If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
`[I, J, ..., L, max_diag_len]` and values:

```
diagonal[i, j, ..., l, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```
where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

Otherwise, the output tensor has rank `r` with dimensions
`[I, J, ..., L, num_diags, max_diag_len]` with values:

```
diagonal[i, j, ..., l, m, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```
where `d = k[1] - m`, `y = max(-d, 0) - offset`, and `x = max(d, 0) - offset`.

`offset` is zero except when the alignment of the diagonal is to the right.
```
offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                           and `d >= 0`) or
                                         (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                           and `d <= 0`)
         0                          ; otherwise
```
where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

The input must be at least a matrix.

For example:

```
input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                   [5, 6, 7, 8],
                   [9, 8, 7, 6]],
                  [[5, 4, 3, 2],
                   [1, 2, 3, 4],
                   [5, 6, 7, 8]]])

# A main diagonal from each batch.
tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                [5, 2, 7]]

# A superdiagonal from each batch.
tf.matrix_diag_part(input, k = 1)
  ==> [[2, 7, 6],  # Output shape: (2, 3)
       [4, 3, 8]]

# A band from each batch.
tf.matrix_diag_part(input, k = (-1, 2))
  ==> [[[0, 3, 8],  # Output shape: (2, 4, 3)
        [2, 7, 6],
        [1, 6, 7],
        [5, 8, 0]],
       [[0, 3, 4],
        [4, 3, 8],
        [5, 2, 7],
        [1, 6, 0]]]

# LEFT_RIGHT alignment.
tf.matrix_diag_part(input, k = (-1, 2), align="LEFT_RIGHT")
  ==> [[[3, 8, 0],  # Output shape: (2, 4, 3)
        [2, 7, 6],
        [1, 6, 7],
        [0, 5, 8]],
       [[3, 4, 0],
        [4, 3, 8],
        [5, 2, 7],
        [0, 1, 6]]]

# max_diag_len can be shorter than the main diagonal.
tf.matrix_diag_part(input, k = (-2, -1))
  ==> [[[5, 8],
        [9, 0]],
       [[1, 6],
        [5, 0]]]

# padding_value = 9
tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
  ==> [[[9, 9, 4],  # Output shape: (2, 3, 3)
        [9, 3, 8],
        [2, 7, 6]],
       [[9, 9, 2],
        [9, 3, 4],
        [4, 3, 8]]]

```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Rank `r` tensor where `r >= 2`.}]>:$input,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k,
    Arg<TF_Tensor, [{The value to fill the area outside the specified diagonal band with.
Default is 0.}]>:$padding_value,

    DefaultValuedAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "\"RIGHT_LEFT\"">:$align
  );

  let results = (outs
    Res<TF_Tensor, [{The extracted diagonal(s).}]>:$diagonal
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
    Arg<TF_Tensor, [{Rank `r`, where `r >= 1`}]>:$diagonal,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k,
    Arg<TF_Int32Tensor, [{The number of rows of the output matrix. If it is not provided, the op assumes
the output matrix is a square matrix and infers the matrix size from k and the
innermost dimension of `diagonal`.}]>:$num_rows,
    Arg<TF_Int32Tensor, [{The number of columns of the output matrix. If it is not provided, the op
assumes the output matrix is a square matrix and infers the matrix size from
k and the innermost dimension of `diagonal`.}]>:$num_cols,
    Arg<TF_Tensor, [{The number to fill the area outside the specified diagonal band with.
Default is 0.}]>:$padding_value
  );

  let results = (outs
    Res<TF_Tensor, [{Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise.}]>:$output
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
    Arg<TF_Tensor, [{Rank `r`, where `r >= 1`}]>:$diagonal,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k,
    Arg<TF_Int32Tensor, [{The number of rows of the output matrix. If it is not provided, the op assumes
the output matrix is a square matrix and infers the matrix size from k and the
innermost dimension of `diagonal`.}]>:$num_rows,
    Arg<TF_Int32Tensor, [{The number of columns of the output matrix. If it is not provided, the op
assumes the output matrix is a square matrix and infers the matrix size from
k and the innermost dimension of `diagonal`.}]>:$num_cols,
    Arg<TF_Tensor, [{The number to fill the area outside the specified diagonal band with.
Default is 0.}]>:$padding_value,

    DefaultValuedAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "\"RIGHT_LEFT\"">:$align
  );

  let results = (outs
    Res<TF_Tensor, [{Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixInverseOp : TF_Op<"MatrixInverse", [NoSideEffect]> {
  let summary = [{
Computes the inverse of one or more square invertible matrices or their adjoints (conjugate transposes).
  }];

  let description = [{
The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, M]`.}]>:$input,

    DefaultValuedAttr<BoolAttr, "false">:$adjoint
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, M]`.

@compatibility(numpy)
Equivalent to np.linalg.inv
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixSetDiagOp : TF_Op<"MatrixSetDiag", [NoSideEffect]> {
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

def TF_MatrixSetDiagV2Op : TF_Op<"MatrixSetDiagV2", [NoSideEffect]> {
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
    Arg<TF_Tensor, [{Rank `r+1`, where `r >= 1`.}]>:$input,
    Arg<TF_Tensor, [{Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
`k >= 1`.}]>:$diagonal,
    Arg<TF_Int32Tensor, [{Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
diagonal, and negative value means subdiagonals. `k` can be a single integer
(for a single diagonal) or a pair of integers specifying the low and high ends
of a matrix band. `k[0]` must not be larger than `k[1]`.}]>:$k,

    DefaultValuedAttr<TF_AnyStrAttrOf<["LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"]>, "\"RIGHT_LEFT\"">:$align
  );

  let results = (outs
    Res<TF_Tensor, [{Rank `r+1`, with `output.shape = input.shape`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixSolveOp : TF_Op<"MatrixSolve", [NoSideEffect]> {
  let summary = "Solves systems of linear equations.";

  let description = [{
`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, M]`.}]>:$matrix,
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, K]`.}]>:$rhs,

    DefaultValuedAttr<BoolAttr, "false">:$adjoint
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Shape is `[..., M, K]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MatrixTriangularSolveOp : TF_Op<"MatrixTriangularSolve", [NoSideEffect]> {
  let summary = [{
Solves systems of linear equations with upper or lower triangular matrices by backsubstitution.
  }];

  let description = [{
`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of each inner-most
matrix is assumed to be zero and not accessed.
`rhs` is a tensor of shape `[..., M, N]`.

The output is a tensor of shape `[..., M, N]`. If `adjoint` is
`True` then the innermost matrices in `output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

Note, the batch shapes for the inputs only need to broadcast.

Example:
```python

a = tf.constant([[3,  0,  0,  0],
                 [2,  1,  0,  0],
                 [1,  0,  1,  0],
                 [1,  1,  1,  1]], dtype=tf.float32)

b = tf.constant([[4],
                 [2],
                 [4],
                 [2]], dtype=tf.float32)

x = tf.linalg.triangular_solve(a, b, lower=True)
x
# <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
# array([[ 1.3333334 ],
#        [-0.66666675],
#        [ 2.6666665 ],
#        [-1.3333331 ]], dtype=float32)>

# in python3 one can use `a@x`
tf.matmul(a, x)
# <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
# array([[4.       ],
#        [2.       ],
#        [4.       ],
#        [1.9999999]], dtype=float32)>
```
  }];

  let arguments = (ins
    Arg<TF_FpOrComplexTensor, [{Shape is `[..., M, M]`.}]>:$matrix,
    Arg<TF_FpOrComplexTensor, [{Shape is `[..., M, K]`.}]>:$rhs,

    DefaultValuedAttr<BoolAttr, "true">:$lower,
    DefaultValuedAttr<BoolAttr, "false">:$adjoint
  );

  let results = (outs
    Res<TF_FpOrComplexTensor, [{Shape is `[..., M, K]`.}]>:$output
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
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

def TF_MaxPoolOp : TF_Op<"MaxPool", [NoSideEffect, TF_FoldOperandsTransposeInterface, TF_LayoutSensitiveInterface]> {
  let summary = "Performs max pooling on the input.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint8]>, [{4-D input to pool over.}]>:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
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

def TF_MaxPool3DOp : TF_Op<"MaxPool3D", [NoSideEffect]> {
  let summary = "Performs 3D max pooling on the input.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{Shape `[batch, depth, rows, cols, channels]` tensor to pool over.}]>:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{The max pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPool3DGradOp : TF_Op<"MaxPool3DGrad", [NoSideEffect]> {
  let summary = "Computes gradients of 3D max pooling function.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{The original input tensor.}]>:$orig_input,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{The original output tensor.}]>:$orig_output,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>, [{Output backprop of shape `[batch, depth, rows, cols, channels]`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr TInput = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPool3DGradGradOp : TF_Op<"MaxPool3DGradGrad", [NoSideEffect]> {
  let summary = "Computes second-order gradients of the maxpooling function.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The original input tensor.}]>:$orig_input,
    Arg<TF_IntOrFpTensor, [{The original output tensor.}]>:$orig_output,
    Arg<TF_IntOrFpTensor, [{Output backprop of shape `[batch, depth, rows, cols, channels]`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<5>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NDHWC", "NCDHW"]>, "\"NDHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Gradients of gradients w.r.t. the input to `max_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPoolGradOp : TF_Op<"MaxPoolGrad", [NoSideEffect]> {
  let summary = "Computes gradients of the maxpooling function.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The original input tensor.}]>:$orig_input,
    Arg<TF_IntOrFpTensor, [{The original output tensor.}]>:$orig_output,
    Arg<TF_IntOrFpTensor, [{4-D.  Gradients w.r.t. the output of `max_pool`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Gradients w.r.t. the input to `max_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_MaxPoolGradGradOp : TF_Op<"MaxPoolGradGrad", [NoSideEffect]> {
  let summary = "Computes second-order gradients of the maxpooling function.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The original input tensor.}]>:$orig_input,
    Arg<TF_IntOrFpTensor, [{The original output tensor.}]>:$orig_output,
    Arg<TF_IntOrFpTensor, [{4-D.  Gradients of gradients w.r.t. the input of `max_pool`.}]>:$grad,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Gradients of gradients w.r.t. the input to `max_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPoolGradGradV2Op : TF_Op<"MaxPoolGradGradV2", [NoSideEffect]> {
  let summary = "Computes second-order gradients of the maxpooling function.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The original input tensor.}]>:$orig_input,
    Arg<TF_IntOrFpTensor, [{The original output tensor.}]>:$orig_output,
    Arg<TF_IntOrFpTensor, [{4-D.  Gradients of gradients w.r.t. the input of `max_pool`.}]>:$grad,
    Arg<TF_Int32Tensor, [{The size of the window for each dimension of the input tensor.}]>:$ksize,
    Arg<TF_Int32Tensor, [{The stride of the sliding window for each dimension of the
input tensor.}]>:$strides,

    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Gradients of gradients w.r.t. the input to `max_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPoolGradV2Op : TF_Op<"MaxPoolGradV2", [NoSideEffect]> {
  let summary = "Computes gradients of the maxpooling function.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The original input tensor.}]>:$orig_input,
    Arg<TF_IntOrFpTensor, [{The original output tensor.}]>:$orig_output,
    Arg<TF_IntOrFpTensor, [{4-D.  Gradients w.r.t. the output of `max_pool`.}]>:$grad,
    Arg<TF_Int32Tensor, [{The size of the window for each dimension of the input tensor.}]>:$ksize,
    Arg<TF_Int32Tensor, [{The stride of the sliding window for each dimension of the
input tensor.}]>:$strides,

    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Gradients w.r.t. the input to `max_pool`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaxPoolV2Op : TF_Op<"MaxPoolV2", [NoSideEffect]> {
  let summary = "Performs max pooling on the input.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint8]>, [{4-D input to pool over.}]>:$input,
    Arg<TF_Int32Tensor, [{The size of the window for each dimension of the input tensor.}]>:$ksize,
    Arg<TF_Int32Tensor, [{The stride of the sliding window for each dimension of the
input tensor.}]>:$strides,

    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint8, TF_Uint16, TF_Uint8]>, [{The max pooled output tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaximumOp : TF_Op<"Maximum", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_MeanOp : TF_Op<"Mean", [NoSideEffect, TF_FoldOperandsTransposeInterface]> {
  let summary = "Computes the mean of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
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

def TF_MergeSummaryOp : TF_Op<"MergeSummary", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Merges summaries.";

  let description = [{
This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{Can be of any shape.  Each must contain serialized `Summary` protocol
buffers.}]>:$inputs
  );

  let results = (outs
    Res<TF_StrTensor, [{Scalar. Serialized `Summary` protocol buffer.}]>:$summary
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_MergeV2CheckpointsOp : TF_Op<"MergeV2Checkpoints", []> {
  let summary = [{
V2 format specific: merges the metadata files of sharded checkpoints.  The
  }];

  let description = [{
result is one logical checkpoint, with one physical metadata file and renamed
data files.

Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.

If delete_old_dirs is true, attempts to delete recursively the dirname of each
path in the input checkpoint_prefixes.  This is useful when those paths are non
user-facing temporary locations.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{prefixes of V2 checkpoints to merge.}]>:$checkpoint_prefixes,
    Arg<TF_StrTensor, [{scalar.  The desired final prefix.  Allowed to be the same
as one of the checkpoint_prefixes.}]>:$destination_prefix,

    DefaultValuedAttr<BoolAttr, "true">:$delete_old_dirs
  );

  let results = (outs);
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_MinimumOp : TF_Op<"Minimum", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_MirrorPadOp : TF_Op<"MirrorPad", [NoSideEffect, TF_OperandHasRank<1, 2>]> {
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

def TF_MirrorPadGradOp : TF_Op<"MirrorPadGrad", [NoSideEffect, TF_OperandHasRank<1, 2>]> {
  let summary = [{
Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
  }];

  let description = [{
This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.

The folded size of each dimension D of the output is:

`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

For example:

```
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The input tensor to be folded.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{A two-column matrix specifying the padding sizes. The number of
rows must be the same as the rank of `input`.}]>:$paddings,

    TF_AnyStrAttrOf<["REFLECT", "SYMMETRIC"]>:$mode
  );

  let results = (outs
    Res<TF_Tensor, [{The folded tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
}

def TF_MlirLocalVarOp : TF_Op<"MlirLocalVarOp", []> {
  let summary = "Creates a handle to an in-scope variable.";

  let description = [{
Used by internal passes for temporary representation of local state, which will
be eventually removed.
  }];

  let arguments = (ins);

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_VariableAlloc]>:$resource
  );
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
  return mlir_passthrough_op([x, y], mlir_module, Toutputs=[tf.float32])

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

def TF_ModOp : TF_Op<"Mod", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_ModelDatasetOp : TF_Op<"ModelDataset", [NoSideEffect]> {
  let summary = "Identity transformation that models performance.";

  let description = [{
Identity transformation that models performance.
  }];

  let arguments = (ins
    Arg<TF_VariantTensor, [{A variant tensor representing the input dataset.}]>:$input_dataset,

    DefaultValuedAttr<I64Attr, "0">:$algorithm,
    DefaultValuedAttr<I64Attr, "0">:$cpu_budget,
    DefaultValuedAttr<I64Attr, "0">:$ram_budget,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_MulOp : TF_Op<"Mul", [Commutative, NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_MulNoNanOp : TF_Op<"MulNoNan", [NoSideEffect, ResultsBroadcastableShape]>,
                    WithBroadcastableBinOpBuilder {
  let summary = [{
Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.
  }];

  let description = [{
*NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x,
    TF_FpOrComplexTensor:$y
  );

  let results = (outs
    TF_FpOrComplexTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MultiDeviceIteratorOp : TF_Op<"MultiDeviceIterator", []> {
  let summary = "Creates a MultiDeviceIterator resource.";

  let arguments = (ins
    Confined<StrArrayAttr, [ArrayMinCount<1>]>:$devices,
    StrAttr:$shared_name,
    StrAttr:$container,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{Handle to the resource created.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_MultiDeviceIteratorFromStringHandleOp : TF_Op<"MultiDeviceIteratorFromStringHandle", []> {
  let summary = [{
Generates a MultiDeviceIterator resource from its provided string handle.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{String representing the resource.}]>:$string_handle,

    DefaultValuedAttr<TypeArrayAttr, "{}">:$output_types,
    DefaultValuedAttr<TF_ShapeAttrArray, "{}">:$output_shapes
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A MultiDeviceIterator resource.}], [TF_DatasetIteratorAlloc]>:$multi_device_iterator
  );
}

def TF_MultiDeviceIteratorGetNextFromShardOp : TF_Op<"MultiDeviceIteratorGetNextFromShard", []> {
  let summary = "Gets next element for the provided shard number.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A MultiDeviceIterator resource.}], [TF_DatasetIteratorRead, TF_DatasetIteratorWrite]>:$multi_device_iterator,
    Arg<TF_Int32Tensor, [{Integer representing which shard to fetch data for.}]>:$shard_num,
    Arg<TF_Int64Tensor, [{Which incarnation of the MultiDeviceIterator is running.}]>:$incarnation_id
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{Result of the get_next on the dataset.}]>:$components
  );

  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr output_types = TF_DerivedResultTypeListAttr<0>;
}

def TF_MultiDeviceIteratorInitOp : TF_Op<"MultiDeviceIteratorInit", []> {
  let summary = "Initializes the multi device iterator with the given dataset.";

  let arguments = (ins
    Arg<TF_VariantTensor, [{Dataset to be iterated upon.}]>:$dataset,
    Arg<TF_ResourceTensor, [{A MultiDeviceIteratorResource.}], [TF_DatasetIteratorWrite]>:$multi_device_iterator,
    Arg<TF_Int64Tensor, [{The maximum size of the host side per device buffer to keep.}]>:$max_buffer_size
  );

  let results = (outs
    Res<TF_Int64Tensor, [{An int64 indicating which incarnation of the MultiDeviceIterator
is running.}]>:$incarnation_id
  );
}

def TF_MultiDeviceIteratorToStringHandleOp : TF_Op<"MultiDeviceIteratorToStringHandle", []> {
  let summary = "Produces a string handle for the given MultiDeviceIterator.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A MultiDeviceIterator resource.}], [TF_DatasetIteratorRead]>:$multi_device_iterator
  );

  let results = (outs
    Res<TF_StrTensor, [{A string representing the resource.}]>:$string_handle
  );
}

def TF_MultinomialOp : TF_Op<"Multinomial", [TF_CannotDuplicate]> {
  let summary = "Draws samples from a multinomial distribution.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
represents the unnormalized log probabilities for all classes.}]>:$logits,
    Arg<TF_Int32Tensor, [{0-D.  Number of independent samples to draw for each row slice.}]>:$num_samples,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
contains the drawn class labels with range `[0, num_classes)`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr output_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_MutableDenseHashTableV2Op : TF_Op<"MutableDenseHashTableV2", []> {
  let summary = [{
Creates an empty hash table that uses tensors as the backing store.
  }];

  let description = [{
It uses "open addressing" with quadratic reprobing to resolve
collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The key used to represent empty key buckets internally. Must not
be used in insert or lookup operations.}]>:$empty_key,
    TF_Tensor:$deleted_key,

    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$value_dtype,
    DefaultValuedAttr<TF_ShapeAttr, "llvm::ArrayRef<int64_t>({})">:$value_shape,
    DefaultValuedAttr<I64Attr, "131072">:$initial_num_buckets,
    DefaultValuedAttr<F32Attr, "0.8f">:$max_load_factor
  );

  let results = (outs
    Res<TF_ResourceTensor, [{Handle to a table.}], [TF_LookupTableAlloc]>:$table_handle
  );

  TF_DerivedOperandTypeAttr key_dtype = TF_DerivedOperandTypeAttr<0>;
}

def TF_MutableHashTableOfTensorsV2Op : TF_Op<"MutableHashTableOfTensorsV2", []> {
  let summary = "Creates an empty hash table.";

  let description = [{
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype,
    DefaultValuedAttr<TF_ShapeAttr, "llvm::ArrayRef<int64_t>({})">:$value_shape
  );

  let results = (outs
    Res<TF_ResourceTensor, [{Handle to a table.}], [TF_LookupTableAlloc]>:$table_handle
  );
}

def TF_MutableHashTableV2Op : TF_Op<"MutableHashTableV2", []> {
  let summary = "Creates an empty hash table.";

  let description = [{
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name,
    DefaultValuedAttr<BoolAttr, "false">:$use_node_name_sharing,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype
  );

  let results = (outs
    Res<TF_ResourceTensor, [{Handle to a table.}], [TF_LookupTableAlloc]>:$table_handle
  );
}

def TF_NdtriOp : TF_Op<"Ndtri", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NegOp : TF_Op<"Neg", [Involution, NoSideEffect, SameOperandsAndResultType, TF_CwiseUnary]> {
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
}

def TF_NextAfterOp : TF_Op<"NextAfter", [NoSideEffect, ResultsBroadcastableShape]>,
                     WithBroadcastableBinOpBuilder {
  let summary = [{
Returns the next representable value of `x1` in the direction of `x2`, element-wise.
  }];

  let description = [{
This operation returns the same result as the C++ std::nextafter function.

It can also return a subnormal number.

@compatibility(cpp)
Equivalent to C++ std::nextafter function.
@end_compatibility
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$x1,
    TF_F32OrF64Tensor:$x2
  );

  let results = (outs
    TF_F32OrF64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_NoOp : TF_Op<"NoOp", [NoSideEffect]> {
  let summary = "Does nothing. Only useful as a placeholder for control edges.";

  let arguments = (ins);

  let results = (outs);
}

def TF_NonMaxSuppressionV3Op : TF_Op<"NonMaxSuppressionV3", [NoSideEffect]> {
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
boxes based on score.}]>:$score_threshold
  );

  let results = (outs
    Res<TF_Int32Tensor, [{A 1-D integer tensor of shape `[M]` representing the selected
indices from the boxes tensor, where `M <= max_output_size`.}]>:$selected_indices
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T_threshold = TF_DerivedOperandTypeAttr<3>;

  let hasCanonicalizer = 1;
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
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 2-D float tensor of shape `[num_boxes, 4]`.}]>:$boxes,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 1-D float tensor of shape `[num_boxes]` representing a single
score corresponding to each box (each row of boxes).}]>:$scores,
    Arg<TF_Int32Tensor, [{A scalar integer tensor representing the maximum number of
boxes to be selected by non max suppression.}]>:$max_output_size,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding whether
boxes overlap too much with respect to IOU.}]>:$iou_threshold,
    Arg<TensorOf<[TF_Float16, TF_Float32]>, [{A 0-D float tensor representing the threshold for deciding when to remove
boxes based on score.}]>:$score_threshold,

    DefaultValuedAttr<BoolAttr, "false">:$pad_to_max_output_size
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

    DefaultValuedAttr<BoolAttr, "false">:$pad_to_max_output_size
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

def TF_NotEqualOp : TF_Op<"NotEqual", [Commutative, NoSideEffect]> {
  let summary = "Returns the truth value of (x != y) element-wise.";

  let description = [{
*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_Tensor:$x,
    TF_Tensor:$y,

    DefaultValuedAttr<BoolAttr, "true">:$incompatible_shape_error
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
    Arg<TensorOf<[TF_Int32, TF_Int64, TF_Uint8]>, [{A tensor of indices.}]>:$indices,
    Arg<TF_Int32Tensor, [{A scalar defining the depth of the one hot dimension.}]>:$depth,
    Arg<TF_Tensor, [{A scalar defining the value to fill in output when `indices[j] = i`.}]>:$on_value,
    Arg<TF_Tensor, [{A scalar defining the value to fill in output when `indices[j] != i`.}]>:$off_value,

    DefaultValuedAttr<I64Attr, "-1">:$axis
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

def TF_OneShotIteratorOp : TF_Op<"OneShotIterator", []> {
  let summary = [{
Makes a "one-shot" iterator that can be iterated only once.
  }];

  let description = [{
A one-shot iterator bundles the logic for defining the dataset and
the state of the iterator in a single op, which allows simple input
pipelines to be defined without an additional initialization
("MakeIterator") step.

One-shot iterators have the following limitations:

* They do not support parameterization: all logic for creating the underlying
  dataset must be bundled in the `dataset_factory` function.
* They are not resettable. Once a one-shot iterator reaches the end of its
  underlying dataset, subsequent "IteratorGetNext" operations on that
  iterator will always produce an `OutOfRange` error.

For greater flexibility, use "Iterator" and "MakeIterator" to define
an iterator using an arbitrary subgraph, which may capture tensors
(including fed values) as parameters, and which may be reset multiple
times by rerunning "MakeIterator".
  }];

  let arguments = (ins
    SymbolRefAttr:$dataset_factory,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name
  );

  let results = (outs
    Res<TF_ResourceTensor, [{A handle to the iterator that can be passed to an "IteratorGetNext"
op.}], [TF_DatasetIteratorAlloc]>:$handle
  );
}

def TF_OnesLikeOp : TF_Op<"OnesLike", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns a tensor of ones with the same shape and type as x.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a tensor of type T.}]>:$x
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a tensor of the same shape and type as x but filled with ones.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_OptimizeDatasetV2Op : TF_Op<"OptimizeDatasetV2", [NoSideEffect]> {
  let summary = [{
Creates a dataset by applying related optimizations to `input_dataset`.
  }];

  let description = [{
Creates a dataset by applying related optimizations to `input_dataset`.
  }];

  let arguments = (ins
    Arg<TF_VariantTensor, [{A variant tensor representing the input dataset.}]>:$input_dataset,
    Arg<TF_StrTensor, [{A `tf.string` vector `tf.Tensor` identifying user enabled optimizations.}]>:$optimizations_enabled,
    Arg<TF_StrTensor, [{A `tf.string` vector `tf.Tensor` identifying user disabled optimizations.}]>:$optimizations_disabled,
    Arg<TF_StrTensor, [{A `tf.string` vector `tf.Tensor` identifying optimizations by default.}]>:$optimizations_default,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrArrayAttr, "{}">:$optimization_configs
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_OptionalFromValueOp : TF_Op<"OptionalFromValue", [NoSideEffect]> {
  let summary = "Constructs an Optional variant from a tuple of tensors.";

  let arguments = (ins
    Variadic<TF_Tensor>:$components
  );

  let results = (outs
    TF_VariantTensor:$optional
  );

  TF_DerivedOperandTypeListAttr Toutput_types = TF_DerivedOperandTypeListAttr<0>;
}

def TF_OptionalGetValueOp : TF_Op<"OptionalGetValue", [NoSideEffect]> {
  let summary = [{
Returns the value stored in an Optional variant or raises an error if none exists.
  }];

  let arguments = (ins
    TF_VariantTensor:$optional
  );

  let results = (outs
    Variadic<TF_Tensor>:$components
  );

  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr output_types = TF_DerivedResultTypeListAttr<0>;
}

def TF_OptionalHasValueOp : TF_Op<"OptionalHasValue", [NoSideEffect]> {
  let summary = [{
Returns true if and only if the given Optional variant has a value.
  }];

  let arguments = (ins
    TF_VariantTensor:$optional
  );

  let results = (outs
    TF_BoolTensor:$has_value
  );
}

def TF_OptionalNoneOp : TF_Op<"OptionalNone", [NoSideEffect]> {
  let summary = "Creates an Optional variant with no value.";

  let arguments = (ins);

  let results = (outs
    TF_VariantTensor:$optional
  );
}

def TF_OutfeedEnqueueTupleOp : TF_Op<"OutfeedEnqueueTuple", []> {
  let summary = "Enqueue multiple Tensor values on the computation outfeed.";

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be inserted into the outfeed queue as an
XLA tuple.}]>:$inputs
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr dtypes = TF_DerivedOperandTypeListAttr<0>;
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
    Arg<Variadic<TF_Tensor>, [{Must be of same shape and type.}]>:$values,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    Res<TF_Tensor, [{The packed tensor.}]>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_PadOp : TF_Op<"Pad", [NoSideEffect, TF_FoldOperandsTransposeInterface, TF_OperandHasRank<1, 2>]> {
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

def TF_PadV2Op : TF_Op<"PadV2", [NoSideEffect, TF_OperandHasRank<1, 2>]> {
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

def TF_ParallelDynamicStitchOp : TF_Op<"ParallelDynamicStitch", [NoSideEffect, SameVariadicOperandSize]> {
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

Values may be merged in parallel, so if an index appears in both `indices[m][i]`
and `indices[n][j]`, the result may be invalid. This differs from the normal
DynamicStitch operator that defines the behavior in that case.

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
    Variadic<TF_Int32Tensor>:$indices,
    Variadic<TF_Tensor>:$data
  );

  let results = (outs
    TF_Tensor:$merged
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ParallelMapDatasetOp : TF_Op<"ParallelMapDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let description = [{
Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `num_parallel_calls` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    Arg<TF_Int32Tensor, [{The number of concurrent invocations of `f` that process
elements from `input_dataset` in parallel.}]>:$num_parallel_calls,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$sloppy,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_ParallelMapDatasetV2Op : TF_Op<"ParallelMapDatasetV2", [NoSideEffect]> {
  let summary = [{
Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let description = [{
Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `num_parallel_calls` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    Arg<TF_Int64Tensor, [{The number of concurrent invocations of `f` that process
elements from `input_dataset` in parallel.}]>:$num_parallel_calls,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<StrAttr, "\"default\"">:$deterministic,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_ParameterizedTruncatedNormalOp : TF_Op<"ParameterizedTruncatedNormal", [TF_CannotDuplicate]> {
  let summary = [{
Outputs random values from a normal distribution. The parameters may each be a
  }];

  let description = [{
scalar which applies to the entire output, or a vector of length shape[0] which
stores the parameters for each batch.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor. Batches are indexed by the 0th dimension.}]>:$shape,
    Arg<TF_FloatTensor, [{The mean parameter of each batch.}]>:$means,
    Arg<TF_FloatTensor, [{The standard deviation parameter of each batch. Must be greater than 0.}]>:$stdevs,
    Arg<TF_FloatTensor, [{The minimum cutoff. May be -infinity.}]>:$minvals,
    Arg<TF_FloatTensor, [{The maximum cutoff. May be +infinity, and must be more than the minval
for each batch.}]>:$maxvals,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A matrix of shape num_batches x samples_per_batch, filled with random
truncated normal values using the parameters for each row.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_PartitionedCallOp : TF_Op<"PartitionedCall", [CallOpInterface, SymbolUserOpInterface, NoSideEffect]> {
  let summary = [{
returns `f(inputs)`, where `f`'s body is placed and partitioned.
  }];

  let description = [{
Asynchronously executes a function, potentially across multiple devices but
within a single process. The kernel places and partitions a given function's
underlying graph, and executes each of the partitioned subgraphs as a function.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of input tensors.}]>:$args,

    SymbolRefAttr:$f,
    DefaultValuedAttr<StrAttr, "\"\"">:$config,
    DefaultValuedAttr<StrAttr, "\"\"">:$config_proto,
    DefaultValuedAttr<StrAttr, "\"\"">:$executor_type
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of return values.}]>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return fAttr(); }

    // returns the callee of this operation.
    func::FuncOp func() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, f());
    }

    // SymbolUserOpInterface verifier.
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable);
  }];
}

def TF_PolygammaOp : TF_Op<"Polygamma", [NoSideEffect, ResultsBroadcastableShape]>,
                     WithBroadcastableBinOpBuilder {
  let summary = [{
Compute the polygamma function \\(\psi^{(n)}(x)\\).
  }];

  let description = [{
The polygamma function is defined as:


\\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)

where \\(\psi(x)\\) is the digamma function.
The polygamma function is defined only for non-negative integer orders \\a\\.
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$a,
    TF_F32OrF64Tensor:$x
  );

  let results = (outs
    TF_F32OrF64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_PopulationCountOp : TF_Op<"PopulationCount", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = [{
Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).
  }];

  let description = [{
For each entry in `x`, calculates the number of `1` (on) bits in the binary
representation of that entry.

**NOTE**: It is more efficient to first `tf.bitcast` your tensors into
`int32` or `int64` and perform the bitcount on the result, than to feed in
8- or 16-bit inputs and then aggregate the resulting counts.
  }];

  let arguments = (ins
    TF_IntTensor:$x
  );

  let results = (outs
    TF_Uint8Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_PowOp : TF_Op<"Pow", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_PrefetchDatasetOp : TF_Op<"PrefetchDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that asynchronously prefetches elements from `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Arg<TF_Int64Tensor, [{The maximum number of elements to buffer in an iterator over
this dataset.}]>:$buffer_size,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<I64Attr, "0">:$slack_period,
    DefaultValuedAttr<BoolAttr, "true">:$legacy_autotune,
    DefaultValuedAttr<I64Attr, "0">:$buffer_size_min,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_PreventGradientOp : TF_Op<"PreventGradient", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

    DefaultValuedAttr<StrAttr, "\"\"">:$message
  );

  let results = (outs
    Res<TF_Tensor, [{the same input tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_PrintOp : TF_Op<"Print", []> {
  let summary = "Prints a list of tensors.";

  let description = [{
Passes `input` through to `output` and prints `data` when evaluating.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor passed to `output`}]>:$input,
    Arg<Variadic<TF_Tensor>, [{A list of tensors to print out when op is evaluated.}]>:$data,

    DefaultValuedAttr<StrAttr, "\"\"">:$message,
    DefaultValuedAttr<I64Attr, "-1">:$first_n,
    DefaultValuedAttr<I64Attr, "3">:$summarize
  );

  let results = (outs
    Res<TF_Tensor, [{The unmodified `input` tensor}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeListAttr U = TF_DerivedOperandTypeListAttr<1>;
}

def TF_PrintV2Op : TF_Op<"PrintV2", []> {
  let summary = "Prints a string scalar.";

  let description = [{
Prints a string scalar to the desired output_stream.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{The string scalar to print.}]>:$input,

    DefaultValuedAttr<StrAttr, "\"stderr\"">:$output_stream,
    DefaultValuedAttr<StrAttr, "\"\\n\"">:$end
  );

  let results = (outs);
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
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_QrOp : TF_Op<"Qr", [NoSideEffect]> {
  let summary = "Computes the QR decompositions of one or more matrices.";

  let description = [{
Computes the QR decomposition of each inner matrix in `tensor` such that
`tensor[..., :, :] = q[..., :, :] * r[..., :,:])`

Currently, the gradient for the QR decomposition is well-defined only when
the first `P` columns of the inner matrix are linearly independent, where
`P` is the minimum of `M` and `N`, the 2 inner-most dimmensions of `tensor`.

```python
# a is a tensor.
# q is a tensor of orthonormal matrices.
# r is a tensor of upper triangular matrices.
q, r = qr(a)
q_full, r_full = qr(a, full_matrices=True)
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.}]>:$input,

    DefaultValuedAttr<BoolAttr, "false">:$full_matrices
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Orthonormal basis for range of `a`. If `full_matrices` is `False` then
shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
`[..., M, M]`.}]>:$q,
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Triangular factor. If `full_matrices` is `False` then shape is
`[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.}]>:$r
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_QuantizeAndDequantizeOp : TF_Op<"QuantizeAndDequantize", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Use QuantizeAndDequantizeV2 instead.";

  let arguments = (ins
    TF_FloatTensor:$input,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$range_given,
    DefaultValuedAttr<F32Attr, "0.0f">:$input_min,
    DefaultValuedAttr<F32Attr, "0.0f">:$input_max
  );

  let results = (outs
    TF_FloatTensor:$output
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
    Arg<TF_FloatTensor, [{Tensor to quantize and then dequantize.}]>:$input,
    Arg<TF_FloatTensor, [{If `range_given == True`, this specifies the minimum input value that needs to
be represented, otherwise it is determined from the min value of the `input`
tensor.}]>:$input_min,
    Arg<TF_FloatTensor, [{If `range_given == True`, this specifies the maximum input value that needs to
be represented, otherwise it is determined from the max value of the `input`
tensor.}]>:$input_max,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$range_given,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HALF_TO_EVEN", "HALF_UP"]>, "\"HALF_TO_EVEN\"">:$round_mode,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_QuantizeAndDequantizeV3Op : TF_Op<"QuantizeAndDequantizeV3", [NoSideEffect]> {
  let summary = "Quantizes then dequantizes a tensor.";

  let description = [{
This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
tensor, so its value can change during training.
  }];

  let arguments = (ins
    TF_FloatTensor:$input,
    TF_FloatTensor:$input_min,
    TF_FloatTensor:$input_max,
    TF_Int32Tensor:$num_bits,

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<BoolAttr, "true">:$range_given,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_QuantizeAndDequantizeV4Op : TF_Op<"QuantizeAndDequantizeV4", [NoSideEffect]> {
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

    DefaultValuedAttr<BoolAttr, "true">:$signed_input,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "false">:$range_given,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HALF_TO_EVEN", "HALF_UP"]>, "\"HALF_TO_EVEN\"">:$round_mode,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_QuantizeV2Op : TF_Op<"QuantizeV2", [NoSideEffect]> {
  let summary = [{
Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
  }];

  let description = [{
[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.  The
'round_mode' attribute controls which rounding tie-breaking algorithm is used
when rounding float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8: out[i] -= (range(T) + 1) / 2.0
```

here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

Assume the input is type float and has a possible range of [0.0, 6.0] and the
output type is quint8 ([0, 255]). The min_range and max_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.

If the output type was qint8 ([-128, 127]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.

If the mode is 'MIN_FIRST', then this approach is used:

```
num_discrete_values = 1 << (# of bits in T)
range_adjust = num_discrete_values / (num_discrete_values - 1)
range = (range_max - range_min) * range_adjust
range_scale = num_discrete_values / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

*SCALED mode Example*

`SCALED` mode matches the quantization approach used in
`QuantizeAndDequantize{V2|V3}`.

If the mode is `SCALED`, the quantization is performed by multiplying each
input value by a scaling_factor.
The scaling_factor is determined from `min_range` and `max_range` to be as large
as possible such that the range from `min_range` to `max_range` is representable
within values of type T.

```c++

  const int min_T = std::numeric_limits<T>::min();
  const int max_T = std::numeric_limits<T>::max();
  const float max_float = std::numeric_limits<float>::max();

  const float scale_factor_from_min_side =
      (min_T * min_range > 0) ? min_T / min_range : max_float;
  const float scale_factor_from_max_side =
      (max_T * max_range > 0) ? max_T / max_range : max_float;

  const float scale_factor = std::min(scale_factor_from_min_side,
                                      scale_factor_from_max_side);
```

We next use the scale_factor to adjust min_range and max_range as follows:

```c++
      min_range = min_T / scale_factor;
      max_range = max_T / scale_factor;
```


e.g. if T = qint8, and initially min_range = -10, and max_range = 9, we would
compare -128/-10.0 = 12.8 to 127/9.0 = 14.11, and set scaling_factor = 12.8
In this case, min_range would remain -10, but max_range would be adjusted to
127 / 12.8 = 9.921875

So we will quantize input values in the range (-10, 9.921875) to (-128, 127).

The input tensor can now be quantized by clipping values to the range
`min_range` to `max_range`, then multiplying by scale_factor as follows:

```c++
result = round(min(max_range, max(min_range, input)) * scale_factor)
```

The adjusted `min_range` and `max_range` are returned as outputs 2 and 3 of
this operation. These outputs should be used as the range for any further
calculations.


*narrow_range (bool) attribute*

If true, we do not use the minimum quantized value.
i.e. for int8 the quantized output, it would be restricted to the range
-127..127 instead of the full -128..127 range.
This is provided for compatibility with certain inference backends.
(Only applies to SCALED mode)


*axis (int) attribute*

An optional `axis` attribute can specify a dimension index of the input tensor,
such that quantization ranges will be calculated and applied separately for each
slice of the tensor along that dimension. This is useful for per-channel
quantization.

If axis is specified, min_range and max_range

if `axis`=None, per-tensor quantization is performed as normal.


*ensure_minimum_range (float) attribute*

Ensures the minimum quantization range is at least this value.
The legacy default value for this is 0.01, but it is strongly suggested to
set it to 0 for new uses.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input,
    Arg<TF_Float32Tensor, [{The minimum value of the quantization range. This value may be adjusted by the
op depending on other parameters. The adjusted value is written to `output_min`.
If the `axis` attribute is specified, this must be a 1-D tensor whose size
matches the `axis` dimension of the input and output tensors.}]>:$min_range,
    Arg<TF_Float32Tensor, [{The maximum value of the quantization range. This value may be adjusted by the
op depending on other parameters. The adjusted value is written to `output_max`.
If the `axis` attribute is specified, this must be a 1-D tensor whose size
matches the `axis` dimension of the input and output tensors.}]>:$max_range,

    DefaultValuedAttr<TF_AnyStrAttrOf<["MIN_COMBINED", "MIN_FIRST", "SCALED"]>, "\"MIN_COMBINED\"">:$mode,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HALF_AWAY_FROM_ZERO", "HALF_TO_EVEN"]>, "\"HALF_AWAY_FROM_ZERO\"">:$round_mode,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "0.01f">:$ensure_minimum_range
  );

  let results = (outs
    Res<TensorOf<[TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8]>, [{The quantized data produced from the float input.}]>:$output,
    Res<TF_Float32Tensor, [{The final quantization range minimum, used to clip input values before scaling
and rounding them to quantized values.
If the `axis` attribute is specified, this will be a 1-D tensor whose size
matches the `axis` dimension of the input and output tensors.}]>:$output_min,
    Res<TF_Float32Tensor, [{The final quantization range maximum, used to clip input values before scaling
and rounding them to quantized values.
If the `axis` attribute is specified, this will be a 1-D tensor whose size
matches the `axis` dimension of the input and output tensors.}]>:$output_max
  );

  TF_DerivedResultTypeAttr T = TF_DerivedResultTypeAttr<0>;
}

def TF_QueueDequeueV2Op : TF_Op<"QueueDequeueV2", []> {
  let summary = "Dequeues a tuple of one or more tensors from the given queue.";

  let description = [{
This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.

N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a queue.}]>:$handle,

    DefaultValuedAttr<I64Attr, "-1">:$timeout_ms
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{One or more tensors that were dequeued as a tuple.}]>:$components
  );

  TF_DerivedResultTypeListAttr component_types = TF_DerivedResultTypeListAttr<0>;
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

def TF_RFFT2DOp : TF_Op<"RFFT2D", [NoSideEffect]> {
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

def TF_RFFT3DOp : TF_Op<"RFFT3D", [NoSideEffect]> {
  let summary = "3D real-valued fast Fourier transform.";

  let description = [{
Computes the 3-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 3 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    Arg<TF_F32OrF64Tensor, [{A float32 tensor.}]>:$input,
    Arg<TF_Int32Tensor, [{An int32 tensor of shape [3]. The FFT length for each dimension.}]>:$fft_length
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64]>, [{A complex64 tensor of the same rank as `input`. The inner-most 3
  dimensions of `input` are replaced with the their 3D Fourier transform. The
  inner-most dimension contains `fft_length / 2 + 1` unique frequency
  components.

@compatibility(numpy)
Equivalent to np.fft.rfftn with 3 dimensions.
@end_compatibility}]>:$output
  );

  TF_DerivedOperandTypeAttr Treal = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tcomplex = TF_DerivedResultTypeAttr<0>;
}

def TF_RGBToHSVOp : TF_Op<"RGBToHSV", [NoSideEffect]> {
  let summary = "Converts one or more images from RGB to HSV.";

  let description = [{
Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

Usage Example:

>>> blue_image = tf.stack([
...    tf.zeros([5,5]),
...    tf.zeros([5,5]),
...    tf.ones([5,5])],
...    axis=-1)
>>> blue_hsv_image = tf.image.rgb_to_hsv(blue_image)
>>> blue_hsv_image[0,0].numpy()
array([0.6666667, 1. , 1. ], dtype=float32)
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{1-D or higher rank. RGB data to convert. Last dimension must be size 3.}]>:$images
  );

  let results = (outs
    Res<TF_FloatTensor, [{`images` converted to HSV.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RaggedGatherOp : TF_Op<"RaggedGather", [NoSideEffect]> {
  let summary = [{
Gather ragged slices from `params` axis `0` according to `indices`.
  }];

  let description = [{
Outputs a `RaggedTensor` output composed from `output_dense_values` and
`output_nested_splits`, such that:

```python
output.shape = indices.shape + params.shape[1:]
output.ragged_rank = indices.shape.ndims + params.ragged_rank
output[i...j, d0...dn] = params[indices[i...j], d0...dn]
```

where

* `params =
   ragged.from_nested_row_splits(params_dense_values, params_nested_splits)`
   provides the values that should be gathered.
* `indices` ia a dense tensor with dtype `int32` or `int64`, indicating which
   values should be gathered.
* `output =
   ragged.from_nested_row_splits(output_dense_values, output_nested_splits)`
   is the output tensor.

(Note: This c++ op is used to implement the higher-level python
`tf.ragged.gather` op, which also supports ragged indices.)
  }];

  let arguments = (ins
    Arg<Variadic<TF_I32OrI64Tensor>, [{The `nested_row_splits` tensors that define the row-partitioning for the
`params` RaggedTensor input.}]>:$params_nested_splits,
    Arg<TF_Tensor, [{The `flat_values` for the `params` RaggedTensor. There was a terminology change
at the python level from dense_values to flat_values, so dense_values is the
deprecated name.}]>:$params_dense_values,
    Arg<TF_I32OrI64Tensor, [{Indices in the outermost dimension of `params` of the values that should be
gathered.}]>:$indices
  );

  let results = (outs
    Res<Variadic<TF_I32OrI64Tensor>, [{The `nested_row_splits` tensors that define the row-partitioning for the
returned RaggedTensor.}]>:$output_nested_splits,
    Res<TF_Tensor, [{The `flat_values` for the returned RaggedTensor.}]>:$output_dense_values
  );

  TF_DerivedOperandSizeAttr PARAMS_RAGGED_RANK = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tsplits = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tvalues = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultSizeAttr OUTPUT_RAGGED_RANK = TF_DerivedResultSizeAttr<0>;
}

def TF_RaggedRangeOp : TF_Op<"RaggedRange", [NoSideEffect]> {
  let summary = [{
Returns a `RaggedTensor` containing the specified sequences of numbers.
  }];

  let description = [{
Returns a `RaggedTensor` `result` composed from `rt_dense_values` and
`rt_nested_splits`, such that
`result[i] = range(starts[i], limits[i], deltas[i])`.

```python
(rt_nested_splits, rt_dense_values) = ragged_range(
      starts=[2, 5, 8], limits=[3, 5, 12], deltas=1)
result = tf.ragged.from_row_splits(rt_dense_values, rt_nested_splits)
print(result)
<tf.RaggedTensor [[2], [], [8, 9, 10, 11]] >
```

The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
The vector inputs must all have the same size.  Scalar inputs are broadcast
to match the size of the vector inputs.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The starts of each range.}]>:$starts,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The limits of each range.}]>:$limits,
    Arg<TensorOf<[TF_Bfloat16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The deltas of each range.}]>:$deltas
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{The `row_splits` for the returned `RaggedTensor`.}]>:$rt_nested_splits,
    Res<TensorOf<[TF_Bfloat16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The `flat_values` for the returned `RaggedTensor`.}]>:$rt_dense_values
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr Tsplits = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomGammaOp : TF_Op<"RandomGamma", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = [{
Outputs random values from the Gamma distribution(s) described by alpha.
  }];

  let description = [{
This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{1-D integer tensor. Shape of independent samples to draw from each
distribution described by the shape parameters given in alpha.}]>:$shape,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{A tensor in which each scalar is a "shape" parameter describing the
associated gamma distribution.}]>:$alpha,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{A tensor with shape `shape + shape(alpha)`. Each slice
`[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
`alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.}]>:$output
  );

  TF_DerivedOperandTypeAttr S = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_RandomGammaGradOp : TF_Op<"RandomGammaGrad", [NoSideEffect, ResultsBroadcastableShape]>,
                           WithBroadcastableBinOpBuilder {
  let summary = [{
Computes the derivative of a Gamma random sample w.r.t. `alpha`.
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$alpha,
    TF_F32OrF64Tensor:$sample
  );

  let results = (outs
    TF_F32OrF64Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RandomPoissonOp : TF_Op<"RandomPoisson", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Use RandomPoissonV2 instead.";

  let arguments = (ins
    TF_I32OrI64Tensor:$shape,
    TensorOf<[TF_Float16, TF_Float32, TF_Float64]>:$rate,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    TensorOf<[TF_Float16, TF_Float32, TF_Float64]>:$output
  );

  TF_DerivedOperandTypeAttr S = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_RandomPoissonV2Op : TF_Op<"RandomPoissonV2", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = [{
Outputs random values from the Poisson distribution(s) described by rate.
  }];

  let description = [{
This op uses two algorithms, depending on rate. If rate >= 10, then
the algorithm by Hormann is used to acquire samples via
transformation-rejection.
See http://www.sciencedirect.com/science/article/pii/0167668793909974.

Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
random variables.
See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
Programming, Volume 2. Addison Wesley
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{1-D integer tensor. Shape of independent samples to draw from each
distribution described by the shape parameters given in rate.}]>:$shape,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{A tensor in which each scalar is a "rate" parameter describing the
associated poisson distribution.}]>:$rate,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{A tensor with shape `shape + shape(rate)`. Each slice
`[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
`rate[i0, i1, ...iN]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr R = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr S = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomShuffleOp : TF_Op<"RandomShuffle", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Randomly shuffles a tensor along its first dimension.";

  let description = [{
The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

```
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to be shuffled.}]>:$value,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_Tensor, [{A tensor of same shape and type as `value`, shuffled along its first
dimension.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RandomStandardNormalOp : TF_Op<"RandomStandardNormal", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Outputs random values from a normal distribution.";

  let description = [{
The generated values will have mean 0 and standard deviation 1.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with random normal values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_RandomUniformOp : TF_Op<"RandomUniform", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with uniform random values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_RandomUniformIntOp : TF_Op<"RandomUniformInt", [TF_CannotDuplicate, TF_RandomGeneratorSideEffect]> {
  let summary = "Outputs random integers from a uniform distribution.";

  let description = [{
The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{0-D.  Inclusive lower bound on the generated integers.}]>:$minval,
    Arg<TF_I32OrI64Tensor, [{0-D.  Exclusive upper bound on the generated integers.}]>:$maxval,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{A tensor of the specified shape filled with uniform random integers.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tout = TF_DerivedOperandTypeAttr<1>;
}

def TF_RangeOp : TF_Op<"Range", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
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
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32]>, [{0-D (scalar). First entry in the sequence.}]>:$start,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32]>, [{0-D (scalar). Upper limit of sequence, exclusive.}]>:$limit,
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32]>, [{0-D (scalar). Optional. Default is 1. Number that increments `start`.}]>:$delta
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32]>, [{1-D.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Value":$start, "Value":$limit, "Value":$delta)>
  ];

  let hasFolder = 1;

}

def TF_RangeDatasetOp : TF_Op<"RangeDataset", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Creates a dataset with a range of values. Corresponds to python's xrange.
  }];

  let arguments = (ins
    Arg<TF_Int64Tensor, [{corresponds to start in python's xrange().}]>:$start,
    Arg<TF_Int64Tensor, [{corresponds to stop in python's xrange().}]>:$stop,
    Arg<TF_Int64Tensor, [{corresponds to step in python's xrange().}]>:$step,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata,
    DefaultValuedAttr<BoolAttr, "false">:$replicate_on_split
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
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

def TF_RealDivOp : TF_Op<"RealDiv", [NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary]>,
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

def TF_ReciprocalOp : TF_Op<"Reciprocal", [Involution, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the reciprocal of x element-wise.";

  let description = [{
I.e., \\(y = 1 / x\\).
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReciprocalGradOp : TF_Op<"ReciprocalGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the gradient for the inverse of `x` wrt its input.";

  let description = [{
Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
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

def TF_RecvOp : TF_Op<"Recv", [TF_RecvSideEffect]> {
  let summary = "Receives the named tensor from send_device on recv_device.";

  let arguments = (ins
    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor to receive.}]>:$tensor
  );

  TF_DerivedResultTypeAttr tensor_type = TF_DerivedResultTypeAttr<0>;
}

def TF_RecvTPUEmbeddingActivationsOp : TF_Op<"RecvTPUEmbeddingActivations", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "An op that receives embedding activations on the TPU.";

  let description = [{
The TPU system performs the embedding lookups and aggregations specified by
the arguments to TPUEmbeddingEnqueue(Integer/Sparse/SparseTensor)Batch. The
results of these aggregations are visible to the Tensorflow Graph as the
outputs of a RecvTPUEmbeddingActivations op. This op returns a list containing
one Tensor of activations per table specified in the model. There can be at
most one RecvTPUEmbeddingActivations op in the TPU graph.
  }];

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    Res<Variadic<TF_Float32Tensor>, [{A TensorList of embedding activations containing one Tensor per
embedding table in the model.}]>:$outputs
  );

  TF_DerivedResultSizeAttr num_outputs = TF_DerivedResultSizeAttr<0>;
}

def TF_ReduceJoinOp : TF_Op<"ReduceJoin", [NoSideEffect]> {
  let summary = "Joins a string Tensor across the given dimensions.";

  let description = [{
Computes the string join across dimensions in the given string Tensor of shape
`[\\(d_0, d_1, ..., d_{n-1}\\)]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.  If
indices are not specified, joins across all dimensions beginning from `n - 1`
through `0`.

For example:

```python
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> "acbd"
tf.reduce_join(a, [1, 0]) ==> "abcd"
tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
```
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{The input to be joined.  All reduced indices must have non-zero size.}]>:$inputs,
    Arg<TF_Int32Tensor, [{The dimensions to reduce over.  Dimensions are reduced in the
order specified.  Omitting `reduction_indices` is equivalent to passing
`[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims,
    DefaultValuedAttr<StrAttr, "\"\"">:$separator
  );

  let results = (outs
    Res<TF_StrTensor, [{Has shape equal to that of the input with reduced dimensions removed or
set to `1` depending on `keep_dims`.}]>:$output
  );
}

def TF_ReluOp : TF_Op<"Relu", [Idempotent, NoSideEffect, SameOperandsAndResultType, TF_LayoutAgnostic]> {
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
}

def TF_Relu6Op : TF_Op<"Relu6", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes rectified linear 6: `min(max(features, 0), 6)`.";

  let arguments = (ins
    TF_IntOrFpTensor:$features
  );

  let results = (outs
    TF_IntOrFpTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_Relu6GradOp : TF_Op<"Relu6Grad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear 6 gradients for a Relu6 operation.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The backpropagated gradients to the corresponding Relu6 operation.}]>:$gradients,
    Arg<TF_IntOrFpTensor, [{The features passed as input to the corresponding Relu6 operation, or
its output; using either one produces the same result.}]>:$features
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{The gradients:
`gradients * (features > 0) * (features < 6)`.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ReluGradOp : TF_Op<"ReluGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes rectified linear gradients for a Relu operation.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{The backpropagated gradients to the corresponding Relu operation.}]>:$gradients,
    Arg<TF_IntOrFpTensor, [{The features passed as input to the corresponding Relu operation, OR
the outputs of that operation (both work equivalently).}]>:$features
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{`gradients * (features > 0)`.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RemoteCallOp : TF_Op<"RemoteCall", []> {
  let summary = "Runs function `f` on a remote device indicated by `target`.";

  let arguments = (ins
    Arg<TF_StrTensor, [{A fully specified device name where we want to run the function.}]>:$target,
    Arg<Variadic<TF_Tensor>, [{A list of arguments for the function.}]>:$args,

    SymbolRefAttr:$f
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of return values.}]>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
}

def TF_RepeatDatasetOp : TF_Op<"RepeatDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that emits the outputs of `input_dataset` `count` times.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Arg<TF_Int64Tensor, [{A scalar representing the number of times that `input_dataset` should
be repeated. A value of `-1` indicates that it should be repeated infinitely.}]>:$count,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
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

def TF_ResizeBilinearOp : TF_Op<"ResizeBilinear", [NoSideEffect]> {
  let summary = "Resize `images` to `size` using bilinear interpolation.";

  let description = [{
Input images can be of different types but output images are always float.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$images,
    Arg<TF_Int32Tensor, [{= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
new size for the images.}]>:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TF_Float32Tensor, [{4-D with shape
`[batch, new_height, new_width, channels]`.}]>:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResizeBilinearGradOp : TF_Op<"ResizeBilinearGrad", [NoSideEffect]> {
  let summary = "Computes the gradient of bilinear interpolation.";

  let arguments = (ins
    Arg<TF_Float32Tensor, [{4-D with shape `[batch, height, width, channels]`.}]>:$grads,
    Arg<TF_FloatTensor, [{4-D with shape `[batch, orig_height, orig_width, channels]`,
The image tensor that was resized.}]>:$original_image,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TF_FloatTensor, [{4-D with shape `[batch, orig_height, orig_width, channels]`.
Gradients with respect to the input image. Input image must have been
float or double.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResizeNearestNeighborOp : TF_Op<"ResizeNearestNeighbor", [NoSideEffect]> {
  let summary = [{
Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$images,
    Arg<TF_Int32Tensor, [{= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
new size for the images.}]>:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint8]>, [{4-D with shape
`[batch, new_height, new_width, channels]`.}]>:$resized_images
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResizeNearestNeighborGradOp : TF_Op<"ResizeNearestNeighborGrad", [NoSideEffect]> {
  let summary = "Computes the gradient of nearest neighbor interpolation.";

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int8, TF_Uint8]>, [{4-D with shape `[batch, height, width, channels]`.}]>:$grads,
    Arg<TF_Int32Tensor, [{= A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
original input size.}]>:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int8, TF_Uint8]>, [{4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
with respect to the input image.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ResourceApplyAdaMaxOp : TF_Op<"ResourceApplyAdaMax", []> {
  let summary = "Update '*var' according to the AdaMax algorithm.";

  let description = [{
m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- max(beta2 * v_{t-1}, abs(g))
variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$m,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$v,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$beta1_power,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum factor. Must be a scalar.}]>:$beta1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum factor. Must be a scalar.}]>:$beta2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Ridge term. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyAdadeltaOp : TF_Op<"ResourceApplyAdadelta", []> {
  let summary = "Update '*var' according to the adadelta scheme.";

  let description = [{
accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum_update,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Decay factor. Must be a scalar.}]>:$rho,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Constant factor. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyAdagradOp : TF_Op<"ResourceApplyAdagrad", []> {
  let summary = "Update '*var' according to the adagrad scheme.";

  let description = [{
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "true">:$update_slots
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyAdagradDAOp : TF_Op<"ResourceApplyAdagradDA", []> {
  let summary = "Update '*var' according to the proximal adagrad scheme.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$gradient_accumulator,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$gradient_squared_accumulator,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 regularization. Must be a scalar.}]>:$l2,
    Arg<TF_Int64Tensor, [{Training step number. Must be a scalar.}]>:$global_step,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyAdagradV2Op : TF_Op<"ResourceApplyAdagradV2", []> {
  let summary = "Update '*var' according to the adagrad scheme.";

  let description = [{
accum += grad * grad
var -= lr * grad * (1 / (sqrt(accum) + epsilon))
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Constant factor. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "true">:$update_slots
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyAdamOp : TF_Op<"ResourceApplyAdam", []> {
  let summary = "Update '*var' according to the Adam algorithm.";

  let description = [{
$$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
$$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
$$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
$$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$m,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$v,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$beta1_power,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$beta2_power,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum factor. Must be a scalar.}]>:$beta1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum factor. Must be a scalar.}]>:$beta2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Ridge term. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$use_nesterov
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyAddSignOp : TF_Op<"ResourceApplyAddSign", []> {
  let summary = "Update '*var' according to the AddSign update.";

  let description = [{
m_t <- beta1 * m_{t-1} + (1 - beta1) * g
update <- (alpha + sign_decay * sign(g) *sign(m)) * g
variable <- variable - lr_t * update
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$m,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$alpha,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$sign_decay,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$beta,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyCenteredRMSPropOp : TF_Op<"ResourceApplyCenteredRMSProp", []> {
  let summary = "Update '*var' according to the centered RMSProp algorithm.";

  let description = [{
The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.

Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient

Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

mg <- rho * mg_{t-1} + (1-rho) * grad
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
var <- var - mom
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$mg,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$ms,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$mom,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Decay rate. Must be a scalar.}]>:$rho,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum Scale. Must be a scalar.}]>:$momentum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Ridge term. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<4>;
}

def TF_ResourceApplyFtrlOp : TF_Op<"ResourceApplyFtrl", []> {
  let summary = "Update '*var' according to the Ftrl-proximal scheme.";

  let description = [{
accum_new = accum + grad * grad
linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$linear,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 regularization. Must be a scalar.}]>:$l2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr_power,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$multiply_linear_by_lr
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyFtrlV2Op : TF_Op<"ResourceApplyFtrlV2", []> {
  let summary = "Update '*var' according to the Ftrl-proximal scheme.";

  let description = [{
accum_new = accum + grad * grad
grad_with_shrinkage = grad + 2 * l2_shrinkage * var
linear += grad_with_shrinkage +
    (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$linear,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 shrinkage regularization. Must be a scalar.}]>:$l2,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$l2_shrinkage,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr_power,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$multiply_linear_by_lr
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceApplyGradientDescentOp : TF_Op<"ResourceApplyGradientDescent", []> {
  let summary = "Update '*var' by subtracting 'alpha' * 'delta' from it.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$alpha,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The change.}]>:$delta,

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
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum. Must be a scalar.}]>:$momentum,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$use_nesterov
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyMomentumOp : TF_Op<"ResourceApplyMomentum", []> {
  let summary = "Update '*var' according to the momentum scheme.";

  let description = [{
Set use_nesterov = True if you want to use Nesterov momentum.

accum = accum * momentum + grad
var -= lr * accum
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Momentum. Must be a scalar.}]>:$momentum,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$use_nesterov
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyPowerSignOp : TF_Op<"ResourceApplyPowerSign", []> {
  let summary = "Update '*var' according to the AddSign update.";

  let description = [{
m_t <- beta1 * m_{t-1} + (1 - beta1) * g
update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
variable <- variable - lr_t * update
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$m,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$logbase,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$sign_decay,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Must be a scalar.}]>:$beta,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyProximalAdagradOp : TF_Op<"ResourceApplyProximalAdagrad", []> {
  let summary = [{
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.
  }];

  let description = [{
accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 regularization. Must be a scalar.}]>:$l2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceApplyProximalGradientDescentOp : TF_Op<"ResourceApplyProximalGradientDescent", []> {
  let summary = "Update '*var' as FOBOS algorithm with fixed learning rate.";

  let description = [{
prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$alpha,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 regularization. Must be a scalar.}]>:$l2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The change.}]>:$delta,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResourceApplyRMSPropOp : TF_Op<"ResourceApplyRMSProp", []> {
  let summary = "Update '*var' according to the RMSProp algorithm.";

  let description = [{
Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$ms,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$mom,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Decay rate. Must be a scalar.}]>:$rho,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$momentum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Ridge term. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_ResourceGatherOp : TF_Op<"ResourceGather", []> {
  let summary = [{
Gather slices from the variable pointed to by `resource` according to `indices`.
  }];

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
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead]>:$resource,
    TF_I32OrI64Tensor:$indices,

    DefaultValuedAttr<I64Attr, "0">:$batch_dims,
    DefaultValuedAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_ResourceScatterAddOp : TF_Op<"ResourceScatterAdd", []> {
  let summary = "Adds sparse updates to the variable referenced by `resource`.";

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterDivOp : TF_Op<"ResourceScatterDiv", []> {
  let summary = [{
Divides sparse updates into the variable referenced by `resource`.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] /= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] /= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterMaxOp : TF_Op<"ResourceScatterMax", []> {
  let summary = [{
Reduces sparse updates into the variable referenced by `resource` using the `max` operation.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] = max(ref[indices, ...], updates[...])

    # Vector indices (for each i)
    ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions are combined.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterMinOp : TF_Op<"ResourceScatterMin", []> {
  let summary = [{
Reduces sparse updates into the variable referenced by `resource` using the `min` operation.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] = min(ref[indices, ...], updates[...])

    # Vector indices (for each i)
    ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions are combined.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterMulOp : TF_Op<"ResourceScatterMul", []> {
  let summary = [{
Multiplies sparse updates into the variable referenced by `resource`.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] *= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] *= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterNdAddOp : TF_Op<"ResourceScatterNdAdd", []> {
  let summary = [{
Applies sparse addition to individual values or slices in a Variable.
  }];

  let description = [{
`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
```

For example, say we want to add 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that addition would look like this:

```python
ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
add = tf.scatter_nd_add(ref, indices, updates)
with tf.Session() as sess:
  print sess.run(add)
```

The resulting update to ref would look like this:

    [1, 13, 3, 14, 14, 6, 7, 20]

See `tf.scatter_nd` for more details about how to make updates to
slices.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A resource handle. Must be from a VarHandleOp.}], [TF_VariableRead, TF_VariableWrite]>:$ref,
    Arg<TF_I32OrI64Tensor, [{A Tensor. Must be one of the following types: int32, int64.
A tensor of indices into ref.}]>:$indices,
    Arg<TF_Tensor, [{A Tensor. Must have the same type as ref. A tensor of
values to add to ref.}]>:$updates,

    DefaultValuedAttr<BoolAttr, "true">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResourceScatterNdSubOp : TF_Op<"ResourceScatterNdSub", []> {
  let summary = [{
Applies sparse subtraction to individual values or slices in a Variable.
  }];

  let description = [{
`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
```

For example, say we want to subtract 4 scattered elements from a rank-1 tensor
with 8 elements. In Python, that subtraction would look like this:

```python
ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
sub = tf.scatter_nd_sub(ref, indices, updates)
with tf.Session() as sess:
  print sess.run(sub)
```

The resulting update to ref would look like this:

    [1, -9, 3, -6, -4, 6, 7, -4]

See `tf.scatter_nd` for more details about how to make updates to
slices.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A resource handle. Must be from a VarHandleOp.}], [TF_VariableRead, TF_VariableWrite]>:$ref,
    Arg<TF_I32OrI64Tensor, [{A Tensor. Must be one of the following types: int32, int64.
A tensor of indices into ref.}]>:$indices,
    Arg<TF_Tensor, [{A Tensor. Must have the same type as ref. A tensor of
values to add to ref.}]>:$updates,

    DefaultValuedAttr<BoolAttr, "true">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResourceScatterNdUpdateOp : TF_Op<"ResourceScatterNdUpdate", []> {
  let summary = [{
Applies sparse `updates` to individual values or slices within a given
  }];

  let description = [{
variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to update 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that update would look like this:

```python
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    update = tf.scatter_nd_update(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(update)
```

The resulting update to ref would look like this:

    [1, 11, 3, 10, 9, 6, 7, 12]

See `tf.scatter_nd` for more details about how to make updates to
slices.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A resource handle. Must be from a VarHandleOp.}], [TF_VariableRead, TF_VariableWrite]>:$ref,
    Arg<TF_I32OrI64Tensor, [{A Tensor. Must be one of the following types: int32, int64.
A tensor of indices into ref.}]>:$indices,
    Arg<TF_Tensor, [{A Tensor. Must have the same type as ref. A tensor of updated
values to add to ref.}]>:$updates,

    DefaultValuedAttr<BoolAttr, "true">:$use_locking
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_ResourceScatterSubOp : TF_Op<"ResourceScatterSub", []> {
  let summary = [{
Subtracts sparse updates from the variable referenced by `resource`.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceScatterUpdateOp : TF_Op<"ResourceScatterUpdate", []> {
  let summary = [{
Assigns sparse updates to the variable referenced by `resource`.
  }];

  let description = [{
This operation computes

    # Scalar indices
    ref[indices, ...] = updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a `Variable` node.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_I32OrI64Tensor, [{A tensor of indices into the first dimension of `ref`.}]>:$indices,
    Arg<TF_Tensor, [{A tensor of updated values to add to `ref`.}]>:$updates
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_ResourceSparseApplyAdagradOp : TF_Op<"ResourceSparseApplyAdagrad", []> {
  let summary = [{
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
  }];

  let description = [{
That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Learning rate. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TF_I32OrI64Tensor, [{A vector of indices into the first dimension of var and accum.}]>:$indices,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "true">:$update_slots
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<4>;
}

def TF_ResourceSparseApplyAdagradV2Op : TF_Op<"ResourceSparseApplyAdagradV2", []> {
  let summary = [{
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
  }];

  let description = [{
That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Learning rate. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Constant factor. Must be a scalar.}]>:$epsilon,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TF_I32OrI64Tensor, [{A vector of indices into the first dimension of var and accum.}]>:$indices,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "true">:$update_slots
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<5>;
}

def TF_ResourceSparseApplyFtrlOp : TF_Op<"ResourceSparseApplyFtrl", []> {
  let summary = [{
Update relevant entries in '*var' according to the Ftrl-proximal scheme.
  }];

  let description = [{
That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$var,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$accum,
    Arg<TF_ResourceTensor, [{Should be from a Variable().}], [TF_VariableRead, TF_VariableWrite]>:$linear,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The gradient.}]>:$grad,
    Arg<TF_I32OrI64Tensor, [{A vector of indices into the first dimension of var and accum.}]>:$indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L1 regularization. Must be a scalar.}]>:$l1,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{L2 regularization. Must be a scalar.}]>:$l2,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Scaling factor. Must be a scalar.}]>:$lr_power,

    DefaultValuedAttr<BoolAttr, "false">:$use_locking,
    DefaultValuedAttr<BoolAttr, "false">:$multiply_linear_by_lr
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<4>;
}

def TF_ResourceStridedSliceAssignOp : TF_Op<"ResourceStridedSliceAssign", []> {
  let summary = "Assign `value` to the sliced l-value reference of `ref`.";

  let description = [{
The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.

NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead, TF_VariableWrite]>:$ref,
    TF_I32OrI64Tensor:$begin,
    TF_I32OrI64Tensor:$end,
    TF_I32OrI64Tensor:$strides,
    TF_Tensor:$value,

    DefaultValuedAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedAttr<I64Attr, "0">:$end_mask,
    DefaultValuedAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedAttr<I64Attr, "0">:$shrink_axis_mask
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<4>;
}

def TF_RestoreOp : TF_Op<"Restore", []> {
  let summary = "Restores a tensor from checkpoint files.";

  let description = [{
Reads a tensor stored in one or several files. If there are several files (for
instance because a tensor was saved as slices), `file_pattern` may contain
wildcard symbols (`*` and `?`) in the filename portion only, not in the
directory portion.

If a `file_pattern` matches several files, `preferred_shard` can be used to hint
in which file the requested tensor is likely to be found. This op will first
open the file at index `preferred_shard` in the list of matching files and try
to restore tensors from that file.  Only if some tensors or tensor slices are
not found in that first file, then the Op opens all the files. Setting
`preferred_shard` to match the value passed as the `shard` input
of a matching `Save` Op may speed up Restore.  This attribute only affects
performance, not correctness.  The default value -1 means files are processed in
order.

See also `RestoreSlice`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element. The pattern of the files from
which we read the tensor.}]>:$file_pattern,
    Arg<TF_StrTensor, [{Must have a single element. The name of the tensor to be
restored.}]>:$tensor_name,

    DefaultValuedAttr<I64Attr, "-1">:$preferred_shard
  );

  let results = (outs
    Res<TF_Tensor, [{The restored tensor.}]>:$tensor
  );

  TF_DerivedResultTypeAttr dt = TF_DerivedResultTypeAttr<0>;
}

def TF_RestoreV2Op : TF_Op<"RestoreV2", []> {
  let summary = "Restores tensors from a V2 checkpoint.";

  let description = [{
For backward compatibility with the V1 format, this Op currently allows
restoring from a V1 checkpoint as well:
  - This Op first attempts to find the V2 index file pointed to by "prefix", and
    if found proceed to read it as a V2 checkpoint;
  - Otherwise the V1 read path is invoked.
Relying on this behavior is not recommended, as the ability to fall back to read
V1 might be deprecated and eventually removed.

By default, restores the named tensors in full.  If the caller wishes to restore
specific slices of stored tensors, "shape_and_slices" should be non-empty
strings and correspondingly well-formed.

Callers must ensure all the named tensors are indeed stored in the checkpoint.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element.  The prefix of a V2 checkpoint.}]>:$prefix,
    Arg<TF_StrTensor, [{shape {N}.  The names of the tensors to be restored.}]>:$tensor_names,
    Arg<TF_StrTensor, [{shape {N}.  The slice specs of the tensors to be restored.
Empty strings indicate that they are non-partitioned tensors.}]>:$shape_and_slices
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{shape {N}.  The restored tensors, whose shapes are read from the
checkpoint directly.}]>:$tensors
  );

  TF_DerivedResultTypeListAttr dtypes = TF_DerivedResultTypeListAttr<0>;
}

def TF_RetrieveTPUEmbeddingADAMParametersOp : TF_Op<"RetrieveTPUEmbeddingADAMParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve ADAM embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the ADAM optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter momenta updated by the ADAM optimization algorithm.}]>:$momenta,
    Res<TF_Float32Tensor, [{Parameter velocities updated by the ADAM optimization algorithm.}]>:$velocities
  );
}

def TF_RetrieveTPUEmbeddingADAMParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingADAMParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$momenta,
    TF_Float32Tensor:$velocities,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingAdadeltaParametersOp : TF_Op<"RetrieveTPUEmbeddingAdadeltaParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve Adadelta embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the Adadelta optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter accumulators updated by the Adadelta optimization algorithm.}]>:$accumulators,
    Res<TF_Float32Tensor, [{Parameter updates updated by the Adadelta optimization algorithm.}]>:$updates
  );
}

def TF_RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$updates,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingAdagradParametersOp : TF_Op<"RetrieveTPUEmbeddingAdagradParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve Adagrad embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the Adagrad optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter accumulators updated by the Adagrad optimization algorithm.}]>:$accumulators
  );
}

def TF_RetrieveTPUEmbeddingAdagradParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingAdagradParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingCenteredRMSPropParametersOp : TF_Op<"RetrieveTPUEmbeddingCenteredRMSPropParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve centered RMSProp embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the centered RMSProp optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter ms updated by the centered RMSProp optimization algorithm.}]>:$ms,
    Res<TF_Float32Tensor, [{Parameter mom updated by the centered RMSProp optimization algorithm.}]>:$mom,
    Res<TF_Float32Tensor, [{Parameter mg updated by the centered RMSProp optimization algorithm.}]>:$mg
  );
}

def TF_RetrieveTPUEmbeddingFTRLParametersOp : TF_Op<"RetrieveTPUEmbeddingFTRLParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve FTRL embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the FTRL optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter accumulators updated by the FTRL optimization algorithm.}]>:$accumulators,
    Res<TF_Float32Tensor, [{Parameter linears updated by the FTRL optimization algorithm.}]>:$linears
  );
}

def TF_RetrieveTPUEmbeddingFTRLParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingFTRLParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$linears,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingMDLAdagradLightParametersOp : TF_Op<"RetrieveTPUEmbeddingMDLAdagradLightParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve MDL Adagrad Light embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the MDL Adagrad Light optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter accumulators updated by the MDL Adagrad Light optimization algorithm.}]>:$accumulators,
    Res<TF_Float32Tensor, [{Parameter weights updated by the MDL Adagrad Light optimization algorithm.}]>:$weights,
    Res<TF_Float32Tensor, [{Parameter benefits updated by the MDL Adagrad Light optimization algorithm.}]>:$benefits
  );
}

def TF_RetrieveTPUEmbeddingMomentumParametersOp : TF_Op<"RetrieveTPUEmbeddingMomentumParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve Momentum embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the Momentum optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter momenta updated by the Momentum optimization algorithm.}]>:$momenta
  );
}

def TF_RetrieveTPUEmbeddingMomentumParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingMomentumParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$momenta,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingProximalAdagradParametersOp : TF_Op<"RetrieveTPUEmbeddingProximalAdagradParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve proximal Adagrad embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the proximal Adagrad optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter accumulators updated by the proximal Adagrad optimization algorithm.}]>:$accumulators
  );
}

def TF_RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$accumulators,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingProximalYogiParametersOp : TF_Op<"RetrieveTPUEmbeddingProximalYogiParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$v,
    TF_Float32Tensor:$m
  );
}

def TF_RetrieveTPUEmbeddingProximalYogiParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingProximalYogiParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$v,
    TF_Float32Tensor:$m,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingRMSPropParametersOp : TF_Op<"RetrieveTPUEmbeddingRMSPropParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve RMSProp embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the RMSProp optimization algorithm.}]>:$parameters,
    Res<TF_Float32Tensor, [{Parameter ms updated by the RMSProp optimization algorithm.}]>:$ms,
    Res<TF_Float32Tensor, [{Parameter mom updated by the RMSProp optimization algorithm.}]>:$mom
  );
}

def TF_RetrieveTPUEmbeddingRMSPropParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingRMSPropParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$ms,
    TF_Float32Tensor:$mom,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_RetrieveTPUEmbeddingStochasticGradientDescentParametersOp : TF_Op<"RetrieveTPUEmbeddingStochasticGradientDescentParameters", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Retrieve SGD embedding parameters.";

  let description = [{
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
  }];

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    Res<TF_Float32Tensor, [{Parameter parameters updated by the stochastic gradient descent optimization algorithm.}]>:$parameters
  );
}

def TF_RetrieveTPUEmbeddingStochasticGradientDescentParametersGradAccumDebugOp : TF_Op<"RetrieveTPUEmbeddingStochasticGradientDescentParametersGradAccumDebug", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedAttr<I64Attr, "-1">:$table_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$table_name,
    I64Attr:$num_shards,
    I64Attr:$shard_id,
    DefaultValuedAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_Float32Tensor:$parameters,
    TF_Float32Tensor:$gradient_accumulators
  );
}

def TF_ReverseOp : TF_Op<"Reverse", [NoSideEffect]> {
  let summary = "Reverses specific dimensions of a tensor.";

  let description = [{
Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

```
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
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Str, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Up to 8-D.}]>:$tensor,
    Arg<TF_BoolTensor, [{1-D. The dimensions to reverse.}]>:$dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Str, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The same shape as `tensor`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    Arg<TF_Tensor, [{The input to reverse.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D with length `input.dims(batch_dim)` and
`max(seq_lengths) <= input.dims(seq_dim)`}]>:$seq_lengths,

    I64Attr:$seq_dim,
    DefaultValuedAttr<I64Attr, "0">:$batch_dim
  );

  let results = (outs
    Res<TF_Tensor, [{The partially reversed input. It has the same shape as `input`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
}

def TF_ReverseV2Op : TF_Op<"ReverseV2", [NoSideEffect]> {
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

def TF_RightShiftOp : TF_Op<"RightShift", [NoSideEffect, ResultsBroadcastableShape]>,
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

def TF_RintOp : TF_Op<"Rint", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns element-wise integer closest to x.";

  let description = [{
If the result is midway between two representable values,
the even representable is chosen.
For example:

```
rint(-1.5) ==> -2.0
rint(0.5000001) ==> 1.0
rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
```
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RiscAddOp : TF_Op<"RiscAdd", [Commutative, NoSideEffect]> {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `RiscAdd` does not supports broadcasting.

Given two input tensors, the `tf.risc_add` operation computes the sum for every element in the tensor.

Both input and output have a range `(-inf, inf)`.
  }];

  let arguments = (ins
    TF_FloatTensor:$x,
    TF_FloatTensor:$y
  );

  let results = (outs
    TF_FloatTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RiscDotOp : TF_Op<"RiscDot", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_FloatTensor:$a,
    TF_FloatTensor:$b,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    TF_FloatTensor:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RngReadAndSkipOp : TF_Op<"RngReadAndSkip", []> {
  let summary = "Advance the counter of a counter-based RNG.";

  let description = [{
The state of the RNG after
`rng_read_and_skip(n)` will be the same as that after `uniform([n])`
(or any other distribution). The actual increment added to the
counter is an unspecified implementation choice.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle of the resource variable that stores the state of the RNG.}]>:$resource,
    Arg<TF_Int32Tensor, [{The RNG algorithm.}]>:$alg,
    Arg<TF_Uint64Tensor, [{The amount of advancement.}]>:$delta
  );

  let results = (outs
    Res<TF_Int64Tensor, [{The old value of the resource variable, before incrementing. Since state size is algorithm-dependent, this output will be right-padded with zeros to reach shape int64[3] (the current maximal state size among algorithms).}]>:$value
  );
}

def TF_RollOp : TF_Op<"Roll", [NoSideEffect]> {
  let summary = "Rolls the elements of a tensor along an axis.";

  let description = [{
The elements are shifted positively (towards larger indices) by the offset of
`shift` along the dimension of `axis`. Negative `shift` values will shift
elements in the opposite direction. Elements that roll passed the last position
will wrap around to the first and vice versa. Multiple shifts along multiple
axes may be specified.

For example:

```
# 't' is [0, 1, 2, 3, 4]
roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]

# shifting along multiple dimensions
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]

# shifting along the same axis multiple times
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
```
  }];

  let arguments = (ins
    TF_Tensor:$input,
    Arg<TF_I32OrI64Tensor, [{Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
elements are shifted positively (towards larger indices) along the dimension
specified by `axis[i]`. Negative shifts will roll the elements in the opposite
direction.}]>:$shift,
    Arg<TF_I32OrI64Tensor, [{Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
`shift[i]` should occur. If the same axis is referenced more than once, the
total shift for that axis will be the sum of all the shifts that belong to that
axis.}]>:$axis
  );

  let results = (outs
    Res<TF_Tensor, [{Has the same shape and size as the input. The elements are shifted
positively (towards larger indices) by the offsets of `shift` along the
dimensions of `axis`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Taxis = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tshift = TF_DerivedOperandTypeAttr<1>;
}

def TF_RoundOp : TF_Op<"Round", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
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

def TF_RsqrtGradOp : TF_Op<"RsqrtGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

def TF_SaveOp : TF_Op<"Save", []> {
  let summary = "Saves the input tensors to disk.";

  let description = [{
The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
is written to `filename` with name `tensor_names[i]`.

See also `SaveSlices`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element. The name of the file to which we write
the tensor.}]>:$filename,
    Arg<TF_StrTensor, [{Shape `[N]`. The names of the tensors to be saved.}]>:$tensor_names,
    Arg<Variadic<TF_Tensor>, [{`N` tensors to save.}]>:$data
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<2>;
}

def TF_SaveSlicesOp : TF_Op<"SaveSlices", []> {
  let summary = "Saves input tensors slices to disk.";

  let description = [{
This is like `Save` except that tensors can be listed in the saved file as being
a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
larger tensor and the slice that this tensor covers. `shapes_and_slices` must
have as many elements as `tensor_names`.

Elements of the `shapes_and_slices` input must either be:

*  The empty string, in which case the corresponding tensor is
   saved normally.
*  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
   `dimI` are the dimensions of the larger tensor and `slice-spec`
   specifies what part is covered by the tensor to save.

`slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
where each `sliceI` is either:

*  The string `-` meaning that the slice covers all indices of this dimension
*  `start,length` where `start` and `length` are integers.  In that
   case the slice covers `length` indices starting at `start`.

See also `Save`.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element. The name of the file to which we write the
tensor.}]>:$filename,
    Arg<TF_StrTensor, [{Shape `[N]`. The names of the tensors to be saved.}]>:$tensor_names,
    Arg<TF_StrTensor, [{Shape `[N]`.  The shapes and slice specifications to use when
saving the tensors.}]>:$shapes_and_slices,
    Arg<Variadic<TF_Tensor>, [{`N` tensors to save.}]>:$data
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<3>;
}

def TF_SaveV2Op : TF_Op<"SaveV2", []> {
  let summary = "Saves tensors in V2 checkpoint format.";

  let description = [{
By default, saves the named tensors in full.  If the caller wishes to save
specific slices of full tensors, "shape_and_slices" should be non-empty strings
and correspondingly well-formed.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element. The prefix of the V2 checkpoint to which we
write the tensors.}]>:$prefix,
    Arg<TF_StrTensor, [{shape {N}. The names of the tensors to be saved.}]>:$tensor_names,
    Arg<TF_StrTensor, [{shape {N}.  The slice specs of the tensors to be saved.
Empty strings indicate that they are non-partitioned tensors.}]>:$shape_and_slices,
    Arg<Variadic<TF_Tensor>, [{`N` tensors to save.}]>:$tensors
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr dtypes = TF_DerivedOperandTypeListAttr<3>;
}

def TF_ScatterNdOp : TF_Op<"ScatterNd", [NoSideEffect]> {
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
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{Tensor of indices.}]>:$indices,
    Arg<TF_Tensor, [{Values to scatter into the output tensor.}]>:$updates,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64]>, [{1-D. The shape of the output tensor.}]>:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor with the given shape and updates applied according
to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
}

def TF_SegmentMaxOp : TF_Op<"SegmentMax", [NoSideEffect]> {
  let summary = "Computes the maximum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

If the max is empty for a given segment ID `i`, `output[i] = 0`.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as the same as a smaller following index.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
</div>

For example:

>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> tf.math.segment_max(c, tf.constant([0, 0, 1])).numpy()
array([[4, 3, 3, 4],
       [5, 6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_SegmentMeanOp : TF_Op<"SegmentMean", [NoSideEffect]> {
  let summary = "Computes the mean along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

If the mean is empty for a given segment ID `i`, `output[i] = 0`.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as a smaller following index when computing the numerator
of the mean.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
</div>

For example:

>>> c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> tf.math.segment_mean(c, tf.constant([0, 0, 1])).numpy()
array([[2.5, 2.5, 2.5, 2.5],
       [5., 6., 7., 8.]], dtype=float32)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_SegmentMinOp : TF_Op<"SegmentMin", [NoSideEffect]> {
  let summary = "Computes the minimum along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

If the min is empty for a given segment ID `i`, `output[i] = 0`.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as the same as a smaller following index.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
</div>

For example:

>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> tf.math.segment_min(c, tf.constant([0, 0, 1])).numpy()
array([[1, 2, 2, 1],
       [5, 6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_SegmentProdOp : TF_Op<"SegmentProd", [NoSideEffect]> {
  let summary = "Computes the product along segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

If the product is empty for a given segment ID `i`, `output[i] = 1`.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as the same as a smaller following index.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
</div>

For example:

>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> tf.math.segment_prod(c, tf.constant([0, 0, 1])).numpy()
array([[4, 6, 6, 4],
       [5, 6, 7, 8]], dtype=int32)
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_SegmentSumOp : TF_Op<"SegmentSum", [NoSideEffect]> {
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.}]>:$segment_ids
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
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
    TF_BoolTensor:$condition,
    Arg<TF_Tensor, [{= A `Tensor` which may have the same shape as `condition`.
If `condition` is rank 1, `x` may have higher rank,
but its first dimension must match the size of `condition`.}]>:$t,
    Arg<TF_Tensor, [{= A `Tensor` with the same type and shape as `x`.}]>:$e
  );

  let results = (outs
    Res<TF_Tensor, [{= A `Tensor` with the same type and shape as `x` and `y`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_SelectV2Op : TF_Op<"SelectV2", [NoSideEffect, ResultsBroadcastableShape]> {
  let summary = "";

  let arguments = (ins
    TF_BoolTensor:$condition,
    TF_Tensor:$t,
    TF_Tensor:$e
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$e, "Value":$t)>
  ];
}

def TF_SelfAdjointEigV2Op : TF_Op<"SelfAdjointEigV2", [NoSideEffect]> {
  let summary = [{
Computes the eigen decomposition of one or more square self-adjoint matrices.
  }];

  let description = [{
Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
are sorted in non-decreasing order.

```python
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = self_adjoint_eig(a)
e = self_adjoint_eig(a, compute_v=False)
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{`Tensor` input of shape `[N, N]`.}]>:$input,

    DefaultValuedAttr<BoolAttr, "true">:$compute_v
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Eigenvalues. Shape is `[N]`.}]>:$e,
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Eigenvectors. Shape is `[N, N]`.}]>:$v
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SeluOp : TF_Op<"Selu", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
  }];

  let description = [{
if < 0, `scale * features` otherwise.

To be used together with
`initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
For correct dropout, use `tf.contrib.nn.alpha_dropout`.

See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  }];

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SeluGradOp : TF_Op<"SeluGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
Computes gradients for the scaled exponential linear (Selu) operation.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{The backpropagated gradients to the corresponding Selu operation.}]>:$gradients,
    Arg<TF_FloatTensor, [{The outputs of the corresponding Selu operation.}]>:$outputs
  );

  let results = (outs
    Res<TF_FloatTensor, [{The gradients: `gradients * (outputs + scale * alpha)`
if outputs < 0, `scale * gradients` otherwise.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SendOp : TF_Op<"Send", [TF_SendSideEffect]> {
  let summary = "Sends the named tensor from send_device to recv_device.";

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to send.}]>:$tensor,

    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SendTPUEmbeddingGradientsOp : TF_Op<"SendTPUEmbeddingGradients", [AttrSizedOperandSegments, TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "Performs gradient updates of embedding tables.";

  let arguments = (ins
    Arg<Variadic<TF_Float32Tensor>, [{A TensorList of gradients with which to update embedding tables.
This argument has the same length and shapes as the return value of
RecvTPUEmbeddingActivations, but contains gradients of the model's loss
with respect to the embedding activations. The embedding tables are updated
from these gradients via the optimizer specified in the TPU embedding
configuration given to tpu.initialize_system.}]>:$inputs,
    Arg<Variadic<TF_Float32Tensor>, [{A TensorList of float32 scalars, one for each dynamic learning
rate tag: see the comments in
//third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
Multiple tables can share the same dynamic learning rate tag as specified
in the configuration. If the learning rates for all tables are constant,
this list should be empty.}]>:$learning_rates,

    StrAttr:$config
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandSizeAttr NN = TF_DerivedOperandSizeAttr<1>;
}

def TF_SerializeIteratorOp : TF_Op<"SerializeIterator", []> {
  let summary = [{
Converts the given `resource_handle` representing an iterator to a variant tensor.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{A handle to an iterator resource.}], [TF_DatasetIteratorRead]>:$resource_handle,

    DefaultValuedAttr<I64Attr, "0">:$external_state_policy
  );

  let results = (outs
    Res<TF_VariantTensor, [{A variant tensor storing the state of the iterator contained in the
resource.}]>:$serialized
  );
}

def TF_SerializeSparseOp : TF_Op<"SerializeSparse", [NoSideEffect]> {
  let summary = "Serialize a `SparseTensor` into a `[3]` `Tensor` object.";

  let arguments = (ins
    Arg<TF_Int64Tensor, [{2-D.  The `indices` of the `SparseTensor`.}]>:$sparse_indices,
    Arg<TF_Tensor, [{1-D.  The `values` of the `SparseTensor`.}]>:$sparse_values,
    Arg<TF_Int64Tensor, [{1-D.  The `shape` of the `SparseTensor`.}]>:$sparse_shape
  );

  let results = (outs
    TensorOf<[TF_Str, TF_Variant]>:$serialized_sparse
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;
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

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "BoolAttr":$use32Bit)>
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

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_ShardedFilenameOp : TF_Op<"ShardedFilename", [NoSideEffect]> {
  let summary = [{
Generate a sharded filename. The filename is printf formatted as
  }];

  let description = [{
%s-%05d-of-%05d, basename, shard, num_shards.
  }];

  let arguments = (ins
    TF_StrTensor:$basename,
    TF_Int32Tensor:$shard,
    TF_Int32Tensor:$num_shards
  );

  let results = (outs
    TF_StrTensor:$filename
  );
}

def TF_ShuffleAndRepeatDatasetV2Op : TF_Op<"ShuffleAndRepeatDatasetV2", []> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    TF_Int64Tensor:$buffer_size,
    TF_Int64Tensor:$seed,
    TF_Int64Tensor:$seed2,
    TF_Int64Tensor:$count,
    Arg<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorRead, TF_DatasetSeedGeneratorWrite]>:$seed_generator,

    DefaultValuedAttr<BoolAttr, "true">:$reshuffle_each_iteration,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_ShuffleDatasetV2Op : TF_Op<"ShuffleDatasetV2", []> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    TF_Int64Tensor:$buffer_size,
    Arg<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorRead, TF_DatasetSeedGeneratorWrite]>:$seed_generator,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_ShuffleDatasetV3Op : TF_Op<"ShuffleDatasetV3", []> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    TF_Int64Tensor:$buffer_size,
    TF_Int64Tensor:$seed,
    TF_Int64Tensor:$seed2,
    Arg<TF_ResourceTensor, "", [TF_DatasetSeedGeneratorRead, TF_DatasetSeedGeneratorWrite]>:$seed_generator,

    DefaultValuedAttr<BoolAttr, "true">:$reshuffle_each_iteration,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_ShutdownDistributedTPUOp : TF_Op<"ShutdownDistributedTPU", []> {
  let summary = "Shuts down a running distributed TPU system.";

  let description = [{
The op returns an error if no system is running.
  }];

  let arguments = (ins);

  let results = (outs);
}

def TF_SigmoidOp : TF_Op<"Sigmoid", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

def TF_SigmoidGradOp : TF_Op<"SigmoidGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_SignOp : TF_Op<"Sign", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
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

def TF_SinhOp : TF_Op<"Sinh", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes hyperbolic sine of x element-wise.";

  let description = [{
Given an input tensor, this function computes hyperbolic sine of every
  element in the tensor. Input range is `[-inf,inf]` and output range
  is `[-inf,inf]`.

  ```python
  x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
  tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
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

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_SliceOp : TF_Op<"Slice", [NoSideEffect, PredOpTrait<"input and output must have same element type", TCresVTEtIsSameAsOp<0, 0>>]> {
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

def TF_SnapshotOp : TF_Op<"Snapshot", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Returns a copy of the input tensor.";

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
    Arg<TF_FloatTensor, [{2-D with shape `[batch_size, num_classes]`.}]>:$logits
  );

  let results = (outs
    Res<TF_FloatTensor, [{Same shape as `logits`.}]>:$softmax
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_SoftmaxCrossEntropyWithLogitsOp : TF_Op<"SoftmaxCrossEntropyWithLogits", [NoSideEffect]> {
  let summary = [{
Computes softmax cross entropy cost and gradients to backpropagate.
  }];

  let description = [{
Inputs are the logits, not probabilities.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{batch_size x num_classes matrix}]>:$features,
    Arg<TF_FloatTensor, [{batch_size x num_classes matrix
The caller must ensure that each batch of labels represents a valid
probability distribution.}]>:$labels
  );

  let results = (outs
    Res<TF_FloatTensor, [{Per example loss (batch_size vector).}]>:$loss,
    Res<TF_FloatTensor, [{backpropagated gradients (batch_size x num_classes matrix).}]>:$backprop
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_SoftplusOp : TF_Op<"Softplus", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "";

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SoftplusGradOp : TF_Op<"SoftplusGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes softplus gradients for a softplus operation.";

  let arguments = (ins
    Arg<TF_FloatTensor, [{The backpropagated gradients to the corresponding softplus operation.}]>:$gradients,
    Arg<TF_FloatTensor, [{The features passed as input to the corresponding softplus operation.}]>:$features
  );

  let results = (outs
    Res<TF_FloatTensor, [{The gradients: `gradients / (1 + exp(-features))`.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SoftsignOp : TF_Op<"Softsign", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes softsign: `features / (abs(features) + 1)`.";

  let arguments = (ins
    TF_FloatTensor:$features
  );

  let results = (outs
    TF_FloatTensor:$activations
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SoftsignGradOp : TF_Op<"SoftsignGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes softsign gradients for a softsign operation.";

  let arguments = (ins
    Arg<TF_FloatTensor, [{The backpropagated gradients to the corresponding softsign operation.}]>:$gradients,
    Arg<TF_FloatTensor, [{The features passed as input to the corresponding softsign operation.}]>:$features
  );

  let results = (outs
    Res<TF_FloatTensor, [{The gradients: `gradients / (1 + abs(features)) ** 2`.}]>:$backprops
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SpaceToBatchOp : TF_Op<"SpaceToBatch", [NoSideEffect]> {
  let summary = "SpaceToBatch for 4-D tensors of type T.";

  let description = [{
This is a legacy version of the more general SpaceToBatchND.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

The attr `block_size` must be greater than one. It indicates the block size.

  * Non-overlapping blocks of size `block_size x block size` in the height and
    width dimensions are rearranged into the batch dimension at each location.
  * The batch of the output tensor is `batch * block_size * block_size`.
  * Both height_pad and width_pad must be divisible by block_size.

The shape of the output will be:

    [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
     depth]

Some examples:

(1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:

```
x = [[[[1], [2]], [[3], [4]]]]
```

The output tensor has shape `[4, 1, 1, 1]` and value:

```
[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
```

(2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:

```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

The output tensor has shape `[4, 1, 1, 3]` and value:

```
[[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
```

(3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:

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

(4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:

```
x = [[[[1],   [2],  [3],  [4]],
      [[5],   [6],  [7],  [8]]],
     [[[9],  [10], [11],  [12]],
      [[13], [14], [15],  [16]]]]
```

The output tensor has shape `[8, 1, 2, 1]` and value:

```
x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
     [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
```

Among others, this operation is useful for reducing atrous convolution into
regular convolution.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{4-D with shape `[batch, height, width, depth]`.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
  the padding of the input with zeros across the spatial dimensions as follows:

      paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

  The effective spatial dimensions of the zero-padded input tensor will be:

      height_pad = pad_top + height + pad_bottom
      width_pad = pad_left + width + pad_right}]>:$paddings,

    Confined<I64Attr, [IntMinValue<2>]>:$block_size
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tpaddings = TF_DerivedOperandTypeAttr<1>;
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
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "\"NHWC\"">:$data_format
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SparseAddOp : TF_Op<"SparseAdd", [NoSideEffect]> {
  let summary = [{
Adds two `SparseTensor` objects to produce another `SparseTensor`.
  }];

  let description = [{
The input `SparseTensor` objects' indices are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.

By default, if two values sum to zero at some index, the output `SparseTensor`
would still include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0` (default) means everything is kept and actual thresholding happens
only for a positive value.

In the following shapes, `nnz` is the count after taking `thresh` into account.
  }];

  let arguments = (ins
    Arg<TF_Int64Tensor, [{2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.}]>:$a_indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.}]>:$a_values,
    Arg<TF_Int64Tensor, [{1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.}]>:$a_shape,
    Arg<TF_Int64Tensor, [{2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.}]>:$b_indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.}]>:$b_values,
    Arg<TF_Int64Tensor, [{1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.}]>:$b_shape,
    Arg<TF_IntOrFpTensor, [{0-D.  The magnitude threshold that determines if an output value/index
pair takes space.}]>:$thresh
  );

  let results = (outs
    TF_Int64Tensor:$sum_indices,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$sum_values,
    TF_Int64Tensor:$sum_shape
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Treal = TF_DerivedOperandTypeAttr<6>;
}

def TF_SparseFillEmptyRowsOp : TF_Op<"SparseFillEmptyRows", [NoSideEffect]> {
  let summary = [{
Fills empty rows in the input 2-D `SparseTensor` with a default value.
  }];

  let description = [{
The input `SparseTensor` is represented via the tuple of inputs
(`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
same `dense_shape` but with indices `output_indices` and values
`output_values`.

This op inserts a single entry for every row that doesn't have any values.
The index is created as `[row, 0, ..., 0]` and the inserted value
is `default_value`.

For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

    [0, 1]: a
    [0, 3]: b
    [2, 0]: c
    [3, 1]: d

Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

    [0, 1]: a
    [0, 3]: b
    [1, 0]: default_value
    [2, 0]: c
    [3, 1]: d
    [4, 0]: default_value

The output `SparseTensor` will be in row-major order and will have the
same shape as the input.

This op also returns an indicator vector shaped `[dense_shape[0]]` such that

    empty_row_indicator[i] = True iff row i was an empty row.

And a reverse index map vector shaped `[indices.shape[0]]` that is used during
backpropagation,

    reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
  }];

  let arguments = (ins
    Arg<TF_Int64Tensor, [{2-D. the indices of the sparse tensor.}]>:$indices,
    Arg<TF_Tensor, [{1-D. the values of the sparse tensor.}]>:$values,
    Arg<TF_Int64Tensor, [{1-D. the shape of the sparse tensor.}]>:$dense_shape,
    Arg<TF_Tensor, [{0-D. default value to insert into location `[row, 0, ..., 0]`
  for rows missing from the input sparse tensor.
output indices: 2-D. the indices of the filled sparse tensor.}]>:$default_value
  );

  let results = (outs
    TF_Int64Tensor:$output_indices,
    Res<TF_Tensor, [{1-D. the values of the filled sparse tensor.}]>:$output_values,
    Res<TF_BoolTensor, [{1-D. whether the dense row was missing in the
input sparse tensor.}]>:$empty_row_indicator,
    Res<TF_Int64Tensor, [{1-D. a map from the input indices to the output indices.}]>:$reverse_index_map
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_SparseMatMulOp : TF_Op<"SparseMatMul", [NoSideEffect]> {
  let summary = [{
Multiply matrix "a" by matrix "b".
  }];

  let description = [{
The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
`SparseTensor`s.  This op is optimized for the case where at least one of "a" or
"b" is sparse, in the sense that they have a large proportion of zero values.
The breakeven for using this versus a dense matrix multiply on one platform was
30% zero values in the sparse matrix.

The gradient computation of this operation will only take advantage of sparsity
in the input gradient when that gradient comes from a Relu.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float32]>:$a,
    TensorOf<[TF_Bfloat16, TF_Float32]>:$b,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<BoolAttr, "false">:$a_is_sparse,
    DefaultValuedAttr<BoolAttr, "false">:$b_is_sparse
  );

  let results = (outs
    TF_Float32Tensor:$product
  );

  TF_DerivedOperandTypeAttr Ta = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tb = TF_DerivedOperandTypeAttr<1>;
}

def TF_SparseReduceSumOp : TF_Op<"SparseReduceSum", [NoSideEffect]> {
  let summary = [{
Computes the sum of elements across dimensions of a SparseTensor.
  }];

  let description = [{
This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.
  }];

  let arguments = (ins
    Arg<TF_Int64Tensor, [{2-D.  `N x R` matrix with the indices of non-empty values in a
SparseTensor, possibly not in canonical ordering.}]>:$input_indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{1-D.  `N` non-empty values corresponding to `input_indices`.}]>:$input_values,
    Arg<TF_Int64Tensor, [{1-D.  Shape of the input SparseTensor.}]>:$input_shape,
    Arg<TF_Int32Tensor, [{1-D.  Length-`K` vector containing the reduction axes.}]>:$reduction_axes,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{`R-K`-D.  The reduced Tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_SparseReshapeOp : TF_Op<"SparseReshape", [NoSideEffect]> {
  let summary = [{
Reshapes a SparseTensor to represent values in a new dense shape.
  }];

  let description = [{
This operation has the same semantics as reshape on the represented dense
tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

If one component of `new_shape` is the special value -1, the size of that
dimension is computed so that the total dense size remains constant.  At
most one component of `new_shape` can be -1.  The number of dense elements
implied by `new_shape` must be the same as the number of dense elements
originally implied by `input_shape`.

Reshaping does not affect the order of values in the SparseTensor.

If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
has length `R_out`, then `input_indices` has shape `[N, R_in]`,
`input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
`output_shape` has length `R_out`.
  }];

  let arguments = (ins
    Arg<TF_Int64Tensor, [{2-D.  `N x R_in` matrix with the indices of non-empty values in a
SparseTensor.}]>:$input_indices,
    Arg<TF_Int64Tensor, [{1-D.  `R_in` vector with the input SparseTensor's dense shape.}]>:$input_shape,
    Arg<TF_Int64Tensor, [{1-D.  `R_out` vector with the requested new dense shape.}]>:$new_shape
  );

  let results = (outs
    Res<TF_Int64Tensor, [{2-D.  `N x R_out` matrix with the updated indices of non-empty
values in the output SparseTensor.}]>:$output_indices,
    Res<TF_Int64Tensor, [{1-D.  `R_out` vector with the full dense shape of the output
SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
filled in.}]>:$output_shape
  );
}

def TF_SparseSegmentMeanOp : TF_Op<"SparseSegmentMean", [NoSideEffect]> {
  let summary = "Computes the mean along sparse segments of a tensor.";

  let description = [{
See `tf.sparse.segment_sum` for usage examples.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.
  }];

  let arguments = (ins
    TF_FloatTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Has same rank as `segment_ids`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Values should be sorted and can be repeated.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_FloatTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentMeanGradOp : TF_Op<"SparseSegmentMeanGrad", [NoSideEffect]> {
  let summary = "Computes gradients for SparseSegmentMean.";

  let description = [{
Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{gradient propagated to the SparseSegmentMean op.}]>:$grad,
    Arg<TF_I32OrI64Tensor, [{indices passed to the corresponding SparseSegmentMean op.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{segment_ids passed to the corresponding SparseSegmentMean op.}]>:$segment_ids,
    Arg<TF_Int32Tensor, [{dimension 0 of "data" passed to SparseSegmentMean op.}]>:$output_dim0
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentMeanWithNumSegmentsOp : TF_Op<"SparseSegmentMeanWithNumSegments", [NoSideEffect]> {
  let summary = "Computes the mean along sparse segments of a tensor.";

  let description = [{
Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
missing, the `output` tensor at that position will be zeroed.

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.
  }];

  let arguments = (ins
    TF_FloatTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Has same rank as `segment_ids`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Values should be sorted and can be repeated.}]>:$segment_ids,
    Arg<TF_I32OrI64Tensor, [{Should equal the number of distinct segment IDs.}]>:$num_segments
  );

  let results = (outs
    Res<TF_FloatTensor, [{Has same shape as data, except for dimension 0 which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<3>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentSqrtNOp : TF_Op<"SparseSegmentSqrtN", [NoSideEffect]> {
  let summary = [{
Computes the sum along sparse segments of a tensor divided by the sqrt of N.
  }];

  let description = [{
N is the size of the segment being reduced.

See `tf.sparse.segment_sum` for usage examples.
  }];

  let arguments = (ins
    TF_FloatTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Has same rank as `segment_ids`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Values should be sorted and can be repeated.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_FloatTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentSqrtNGradOp : TF_Op<"SparseSegmentSqrtNGrad", [NoSideEffect]> {
  let summary = "Computes gradients for SparseSegmentSqrtN.";

  let description = [{
Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{gradient propagated to the SparseSegmentSqrtN op.}]>:$grad,
    Arg<TF_I32OrI64Tensor, [{indices passed to the corresponding SparseSegmentSqrtN op.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{segment_ids passed to the corresponding SparseSegmentSqrtN op.}]>:$segment_ids,
    Arg<TF_Int32Tensor, [{dimension 0 of "data" passed to SparseSegmentSqrtN op.}]>:$output_dim0
  );

  let results = (outs
    TF_FloatTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentSqrtNWithNumSegmentsOp : TF_Op<"SparseSegmentSqrtNWithNumSegments", [NoSideEffect]> {
  let summary = [{
Computes the sum along sparse segments of a tensor divided by the sqrt of N.
  }];

  let description = [{
N is the size of the segment being reduced.

Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
missing, the `output` tensor at that position will be zeroed.

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.
  }];

  let arguments = (ins
    TF_FloatTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Has same rank as `segment_ids`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Values should be sorted and can be repeated.}]>:$segment_ids,
    Arg<TF_I32OrI64Tensor, [{Should equal the number of distinct segment IDs.}]>:$num_segments
  );

  let results = (outs
    Res<TF_FloatTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<3>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
}

def TF_SparseSegmentSumOp : TF_Op<"SparseSegmentSum", [NoSideEffect]> {
  let summary = "Computes the sum along sparse segments of a tensor.";

  let description = [{
Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

For example:

```python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
# => [[0 0 0 0]]

# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
# => [[ 1  2  3  4]
#     [-1 -2 -3 -4]]

# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
# => [[0 0 0 0]
#     [5 6 7 8]]

# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
```
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$data,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Has same rank as `segment_ids`.}]>:$indices,
    Arg<TF_I32OrI64Tensor, [{A 1-D tensor. Values should be sorted and can be repeated.}]>:$segment_ids
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{Has same shape as data, except for dimension 0 which
has size `k`, the number of segments.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tsegmentids = TF_DerivedOperandTypeAttr<2>;
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
    Arg<TF_FloatTensor, [{batch_size x num_classes matrix}]>:$features,
    Arg<TF_I32OrI64Tensor, [{batch_size vector with values in [0, num_classes).
This is the label for the given minibatch entry.}]>:$labels
  );

  let results = (outs
    Res<TF_FloatTensor, [{Per example loss (batch_size vector).}]>:$loss,
    Res<TF_FloatTensor, [{backpropagated gradients (batch_size x num_classes matrix).}]>:$backprop
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tlabels = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_SparseTensorDenseMatMulOp : TF_Op<"SparseTensorDenseMatMul", [NoSideEffect]> {
  let summary = [{
Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
  }];

  let description = [{
No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:

if adjoint_a == false:
  A should be sorted in lexicographically increasing order.  Use SparseReorder
  if you're not sure.
if adjoint_a == true:
  A should be sorted in order of increasing dimension 1 (i.e., "column major"
  order instead of "row major" order).
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.}]>:$a_indices,
    Arg<TF_Tensor, [{1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.}]>:$a_values,
    Arg<TF_Int64Tensor, [{1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.}]>:$a_shape,
    Arg<TF_Tensor, [{2-D.  A dense Matrix.}]>:$b,

    DefaultValuedAttr<BoolAttr, "false">:$adjoint_a,
    DefaultValuedAttr<BoolAttr, "false">:$adjoint_b
  );

  let results = (outs
    TF_Tensor:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
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
    Arg<TF_I32OrI64Tensor, [{0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
index where `sparse_values[i]` will be placed.}]>:$sparse_indices,
    Arg<TF_I32OrI64Tensor, [{1-D.  Shape of the dense output tensor.}]>:$output_shape,
    Arg<TF_Tensor, [{1-D.  Values corresponding to each row of `sparse_indices`,
or a scalar value to be used for all sparse indices.}]>:$sparse_values,
    Arg<TF_Tensor, [{Scalar value to set for indices not specified in
`sparse_indices`.}]>:$default_value,

    DefaultValuedAttr<BoolAttr, "true">:$validate_indices
  );

  let results = (outs
    Res<TF_Tensor, [{Dense output tensor of shape `output_shape`.}]>:$dense
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<0>;
}

def TF_SplitOp : TF_Op<"Split", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let arguments = (ins
    Arg<TF_Int32Tensor, [{0-D.  The dimension along which to split.  Must be in the range
`[-rank(value), rank(value))`.}]>:$split_dim,
    Arg<TF_Tensor, [{The tensor to split.}]>:$value
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{They are identically shaped tensors, whose shape matches that of `value`
except along `axis`, where their sizes are
`values.shape[split_dim] / num_split`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultSizeAttr num_split = TF_DerivedResultSizeAttr<0>;

  let hasVerifier = 1;
}

def TF_SplitVOp : TF_Op<"SplitV", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to split.}]>:$value,
    Arg<TF_I32OrI64Tensor, [{list containing the sizes of each output tensor along the split
dimension. Must sum to the dimension of value along split_dim.
Can contain one -1 indicating that dimension is to be inferred.}]>:$size_splits,
    Arg<TF_Int32Tensor, [{0-D.  The dimension along which to split.  Must be in the range
`[-rank(value), rank(value))`.}]>:$split_dim
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{Tensors whose shape matches that of `value`
except along `axis`, where their sizes are
`size_splits[i]`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultSizeAttr num_split = TF_DerivedResultSizeAttr<0>;

  let hasVerifier = 1;
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

def TF_SqrtGradOp : TF_Op<"SqrtGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the gradient for the sqrt of `x` wrt its input.";

  let description = [{
Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
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

def TF_SquareOp : TF_Op<"Square", [NoSideEffect, SameOperandsAndResultType]> {
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
}

def TF_SquaredDifferenceOp : TF_Op<"SquaredDifference", [Commutative, NoSideEffect, ResultsBroadcastableShape]>,
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
    Arg<TF_Tensor, [{The `input` to squeeze.}]>:$input,

    DefaultValuedAttr<I64ArrayAttr, "{}">:$squeeze_dims
  );

  let results = (outs
    Res<TF_Tensor, [{Contains the same data as `input`, but has one or more dimensions of
size 1 removed.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_StackCloseV2Op : TF_Op<"StackCloseV2", []> {
  let summary = "Delete the stack from its resource container.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a stack.}], [TF_StackFree]>:$handle
  );

  let results = (outs);
}

def TF_StackPopV2Op : TF_Op<"StackPopV2", []> {
  let summary = "Pop the element at the top of the stack.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a stack.}], [TF_StackRead, TF_StackWrite]>:$handle
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor that is popped from the top of the stack.}]>:$elem
  );

  TF_DerivedResultTypeAttr elem_type = TF_DerivedResultTypeAttr<0>;
}

def TF_StackPushV2Op : TF_Op<"StackPushV2", []> {
  let summary = "Push an element onto the stack.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a stack.}], [TF_StackRead, TF_StackWrite]>:$handle,
    Arg<TF_Tensor, [{The tensor to be pushed onto the stack.}]>:$elem,

    DefaultValuedAttr<BoolAttr, "false">:$swap_memory
  );

  let results = (outs
    Res<TF_Tensor, [{The same tensor as the input 'elem'.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_StackV2Op : TF_Op<"StackV2", [TF_UniqueResourceAllocation]> {
  let summary = "A stack that produces elements in first-in last-out order.";

  let arguments = (ins
    Arg<TF_Int32Tensor, [{The maximum size of the stack if non-negative. If negative, the stack
size is unlimited.}]>:$max_size,

    TypeAttr:$elem_type,
    DefaultValuedAttr<StrAttr, "\"\"">:$stack_name
  );

  let results = (outs
    Res<TF_ResourceTensor, [{The handle to the stack.}], [TF_StackAlloc]>:$handle
  );
}

def TF_StatefulStandardNormalV2Op : TF_Op<"StatefulStandardNormalV2", []> {
  let summary = "Outputs random values from a normal distribution.";

  let description = [{
The generated values will have mean 0 and standard deviation 1.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle of the resource variable that stores the state of the RNG.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_Int64Tensor, [{The RNG algorithm.}]>:$algorithm,
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with random normal values.}]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatefulTruncatedNormalOp : TF_Op<"StatefulTruncatedNormal", []> {
  let summary = "Outputs random values from a truncated normal distribution.";

  let description = [{
The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle of the resource variable that stores the state of the RNG.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_Int64Tensor, [{The RNG algorithm.}]>:$algorithm,
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatefulUniformOp : TF_Op<"StatefulUniform", []> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle of the resource variable that stores the state of the RNG.}], [TF_VariableRead, TF_VariableWrite]>:$resource,
    Arg<TF_Int64Tensor, [{The RNG algorithm.}]>:$algorithm,
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessMultinomialOp : TF_Op<"StatelessMultinomial", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Draws samples from a multinomial distribution.";

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
represents the unnormalized log probabilities for all classes.}]>:$logits,
    Arg<TF_Int32Tensor, [{0-D.  Number of independent samples to draw for each row slice.}]>:$num_samples,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
contains the drawn class labels with range `[0, num_classes)`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr output_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessParameterizedTruncatedNormalOp : TF_Op<"StatelessParameterizedTruncatedNormal", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "";

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The mean parameter of each batch.}]>:$means,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The standard deviation parameter of each batch. Must be greater than 0.}]>:$stddevs,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The minimum cutoff. May be -infinity.}]>:$minvals,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The maximum cutoff. May be +infinity, and must be more than the minval
for each batch.}]>:$maxvals
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The outputs are truncated normal samples and are a deterministic function of
`shape`, `seed`, `minvals`, `maxvals`, `means` and `stddevs`.}]>:$output
  );

  TF_DerivedOperandTypeAttr S = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_StatelessRandomBinomialOp : TF_Op<"StatelessRandomBinomial", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random numbers from a binomial distribution.
  }];

  let description = [{
Outputs random values from a binomial distribution.

The outputs are a deterministic function of `shape`, `seed`, `counts`, and `probs`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The counts of the binomial distribution. Must be broadcastable with `probs`,
and broadcastable with the rightmost dimensions of `shape`.}]>:$counts,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The probability of success for the binomial distribution. Must be broadcastable
with `counts` and broadcastable with the rightmost dimensions of `shape`.}]>:$probs
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr S = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomGammaV2Op : TF_Op<"StatelessRandomGammaV2", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random numbers from a gamma distribution.
  }];

  let description = [{
Outputs random values from a gamma distribution.

The outputs are a deterministic function of `shape`, `seed`, and `alpha`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{The concentration of the gamma distribution. Shape must match the rightmost
dimensions of `shape`.}]>:$alpha
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_StatelessRandomGetAlgOp : TF_Op<"StatelessRandomGetAlg", []> {
  let summary = "Picks the best counter-based RNG algorithm based on device.";

  let description = [{
This op picks the best counter-based RNG algorithm based on device.
  }];

  let arguments = (ins);

  let results = (outs
    Res<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );
}

def TF_StatelessRandomGetKeyCounterOp : TF_Op<"StatelessRandomGetKeyCounter", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Scrambles seed into key and counter, using the best algorithm based on device.
  }];

  let description = [{
This op scrambles a shape-[2] seed into a key and a counter, both needed by counter-based RNG algorithms. The scrambing uses the best algorithm based on device. The scrambling is opaque but approximately satisfies the property that different seed results in different key/counter pair (which will in turn result in different random numbers).
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Res<TF_Uint64Tensor, [{Counter for the counter-based RNG algorithm. Since counter size is algorithm-dependent, this output will be right-padded with zeros to reach shape uint64[2] (the current maximal counter size among algorithms).}]>:$counter
  );

  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<0>;
}

def TF_StatelessRandomGetKeyCounterAlgOp : TF_Op<"StatelessRandomGetKeyCounterAlg", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Picks the best algorithm based on device, and scrambles seed into key and counter.
  }];

  let description = [{
This op picks the best counter-based RNG algorithm based on device, and scrambles a shape-[2] seed into a key and a counter, both needed by the counter-based algorithm. The scrambling is opaque but approximately satisfies the property that different seed results in different key/counter pair (which will in turn result in different random numbers).
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Res<TF_Uint64Tensor, [{Counter for the counter-based RNG algorithm. Since counter size is algorithm-dependent, this output will be right-padded with zeros to reach shape uint64[2] (the current maximal counter size among algorithms).}]>:$counter,
    Res<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );

  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<0>;
}

def TF_StatelessRandomNormalOp : TF_Op<"StatelessRandomNormal", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom values from a normal distribution.
  }];

  let description = [{
The generated values will have mean 0 and standard deviation 1.

The outputs are a deterministic function of `shape` and `seed`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomNormalV2Op : TF_Op<"StatelessRandomNormalV2", [NoSideEffect]> {
  let summary = [{
Outputs deterministic pseudorandom values from a normal distribution.
  }];

  let description = [{
The generated values will have mean 0 and standard deviation 1.

The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Arg<TF_Uint64Tensor, [{Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.}]>:$counter,
    Arg<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomPoissonOp : TF_Op<"StatelessRandomPoisson", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random numbers from a Poisson distribution.
  }];

  let description = [{
Outputs random values from a Poisson distribution.

The outputs are a deterministic function of `shape`, `seed`, and `lam`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed,
    Arg<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{The rate of the Poisson distribution. Shape must match the rightmost dimensions
of `shape`.}]>:$lam
  );

  let results = (outs
    Res<TensorOf<[TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Int64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Rtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomUniformOp : TF_Op<"StatelessRandomUniform", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random values from a uniform distribution.
  }];

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

The outputs are a deterministic function of `shape` and `seed`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomUniformFullIntOp : TF_Op<"StatelessRandomUniformFullInt", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random integers from a uniform distribution.
  }];

  let description = [{
The generated values are uniform integers covering the whole range of `dtype`.

The outputs are a deterministic function of `shape` and `seed`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomUniformFullIntV2Op : TF_Op<"StatelessRandomUniformFullIntV2", [NoSideEffect]> {
  let summary = [{
Outputs deterministic pseudorandom random integers from a uniform distribution.
  }];

  let description = [{
The generated values are uniform integers covering the whole range of `dtype`.

The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Arg<TF_Uint64Tensor, [{Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.}]>:$counter,
    Arg<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );

  let results = (outs
    Res<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessRandomUniformIntOp : TF_Op<"StatelessRandomUniformInt", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom random integers from a uniform distribution.
  }];

  let description = [{
The generated values follow a uniform distribution in the range `[minval, maxval)`.

The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed,
    Arg<TF_I32OrI64Tensor, [{Minimum value (inclusive, scalar).}]>:$minval,
    Arg<TF_I32OrI64Tensor, [{Maximum value (exclusive, scalar).}]>:$maxval
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<2>;
}

def TF_StatelessRandomUniformIntV2Op : TF_Op<"StatelessRandomUniformIntV2", [NoSideEffect]> {
  let summary = [{
Outputs deterministic pseudorandom random integers from a uniform distribution.
  }];

  let description = [{
The generated values follow a uniform distribution in the range `[minval, maxval)`.

The outputs are a deterministic function of `shape`, `key`, `counter`, `alg`, `minval` and `maxval`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Arg<TF_Uint64Tensor, [{Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.}]>:$counter,
    Arg<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg,
    Arg<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{Minimum value (inclusive, scalar).}]>:$minval,
    Arg<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{Maximum value (exclusive, scalar).}]>:$maxval
  );

  let results = (outs
    Res<TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<4>;
}

def TF_StatelessRandomUniformV2Op : TF_Op<"StatelessRandomUniformV2", [NoSideEffect]> {
  let summary = [{
Outputs deterministic pseudorandom random values from a uniform distribution.
  }];

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Arg<TF_Uint64Tensor, [{Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.}]>:$counter,
    Arg<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessTruncatedNormalOp : TF_Op<"StatelessTruncatedNormal", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Outputs deterministic pseudorandom values from a truncated normal distribution.
  }];

  let description = [{
The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

The outputs are a deterministic function of `shape` and `seed`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_I32OrI64Tensor, [{2 seeds (shape [2]).}]>:$seed
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tseed = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatelessTruncatedNormalV2Op : TF_Op<"StatelessTruncatedNormalV2", [NoSideEffect]> {
  let summary = [{
Outputs deterministic pseudorandom values from a truncated normal distribution.
  }];

  let description = [{
The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,
    Arg<TF_Uint64Tensor, [{Key for the counter-based RNG algorithm (shape uint64[1]).}]>:$key,
    Arg<TF_Uint64Tensor, [{Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.}]>:$counter,
    Arg<TF_Int32Tensor, [{The RNG algorithm (shape int32[]).}]>:$alg
  );

  let results = (outs
    Res<TF_FloatTensor, [{Random values with specified shape.}]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StaticRegexFullMatchOp : TF_Op<"StaticRegexFullMatch", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Check if the input matches the regex pattern.";

  let description = [{
The input is a string tensor of any shape. The pattern is the
regular expression to be matched with every element of the input tensor.
The boolean values (True or False) of the output tensor indicate
if the input matches the regex pattern provided.

The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{A string tensor of the text to be processed.}]>:$input,

    StrAttr:$pattern
  );

  let results = (outs
    Res<TF_BoolTensor, [{A bool tensor with the same shape as `input`.}]>:$output
  );
}

def TF_StopGradientOp : TF_Op<"StopGradient", [NoSideEffect, TF_AllTypesMatch<["input", "output"]>]> {
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

    DefaultValuedAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedAttr<I64Attr, "0">:$end_mask,
    DefaultValuedAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedAttr<I64Attr, "0">:$shrink_axis_mask
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

  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<4>;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // If sliced shape is able to be deduced, returns true, updates `shape`
    // with the final shape after performing StridedSlice, and updates
    // `begin_indices`, `end_indices`, and `strides` with their canonical
    // values, respectively.
    bool GetSlicedShapeAndBoundRanges(
      ::llvm::SmallVectorImpl<int64_t> *input_shape,
      ::llvm::SmallVectorImpl<int64_t> *slice_begin,
      ::llvm::SmallVectorImpl<int64_t> *slice_end,
      ::llvm::SmallVectorImpl<int64_t> *slice_stride);
  }];
}

def TF_StringJoinOp : TF_Op<"StringJoin", [NoSideEffect]> {
  let summary = [{
Joins the strings in the given list of string tensors into one tensor;
  }];

  let description = [{
with the given separator (default is an empty separator).

Examples:

>>> s = ["hello", "world", "tensorflow"]
>>> tf.strings.join(s, " ")
<tf.Tensor: shape=(), dtype=string, numpy=b'hello world tensorflow'>
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{A list of string tensors.  The tensors must all have the same shape,
or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
of non-scalar inputs.}]>:$inputs,

    DefaultValuedAttr<StrAttr, "\"\"">:$separator
  );

  let results = (outs
    TF_StrTensor:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_StringStripOp : TF_Op<"StringStrip", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Strip leading and trailing whitespaces from the Tensor.";

  let description = [{
Examples:

>>> tf.strings.strip(["\nTensorFlow", "     The python library    "]).numpy()
array([b'TensorFlow', b'The python library'], dtype=object)
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{A string `Tensor` of any shape.}]>:$input
  );

  let results = (outs
    Res<TF_StrTensor, [{A string `Tensor` of the same shape as the input.}]>:$output
  );
}

def TF_StringToHashBucketFastOp : TF_Op<"StringToHashBucketFast", [NoSideEffect]> {
  let summary = [{
Converts each string in the input Tensor to its hash mod by a number of buckets.
  }];

  let description = [{
The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

Examples:

>>> tf.strings.to_hash_bucket_fast(["Hello", "TensorFlow", "2.x"], 3).numpy()
array([0, 2, 2])
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{The strings to assign a hash bucket.}]>:$input,

    Confined<I64Attr, [IntMinValue<1>]>:$num_buckets
  );

  let results = (outs
    Res<TF_Int64Tensor, [{A Tensor of the same shape as the input `string_tensor`.}]>:$output
  );
}

def TF_SubOp : TF_Op<"Sub", [NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary, TF_SameOperandsAndResultElementTypeResolveRef]>,
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

def TF_SumOp : TF_Op<"Sum", [NoSideEffect]> {
  let summary = "Computes the sum of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The tensor to reduce.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{The dimensions to reduce. Must be in the range
`[-rank(input), rank(input))`.}]>:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The reduced tensor.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$reduction_indices,
      "BoolAttr":$keep_dims)>
  ];

  let hasFolder = 1;
}

def TF_SvdOp : TF_Op<"Svd", [NoSideEffect]> {
  let summary = [{
Computes the singular value decompositions of one or more matrices.
  }];

  let description = [{
Computes the SVD of each inner matrix in `input` such that
`input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`

```python
# a is a tensor containing a batch of matrices.
# s is a tensor of singular values for each matrix.
# u is the tensor containing the left singular vectors for each matrix.
# v is the tensor containing the right singular vectors for each matrix.
s, u, v = svd(a)
s, _, _ = svd(a, compute_uv=False)
```
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.}]>:$input,

    DefaultValuedAttr<BoolAttr, "true">:$compute_uv,
    DefaultValuedAttr<BoolAttr, "false">:$full_matrices
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Singular values. Shape is `[..., P]`.}]>:$s,
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Left singular vectors. If `full_matrices` is `False` then shape is
`[..., M, P]`; if `full_matrices` is `True` then shape is
`[..., M, M]`. Undefined if `compute_uv` is `False`.}]>:$u,
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>, [{Left singular vectors. If `full_matrices` is `False` then shape is
`[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
Undefined if `compute_uv` is false.}]>:$v
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SymbolicGradientOp : TF_Op<"SymbolicGradient", [NoSideEffect]> {
  let summary = [{
Computes the gradient function for function f via backpropagation.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{a list of input tensors of size N + M;}]>:$input,

    SymbolRefAttr:$f
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{a list of output tensors of size N;}]>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
}

def TF_TPUCompilationResultOp : TF_Op<"TPUCompilationResult", [TF_MustExecute]> {
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

def TF_TPUCompileSucceededAssertOp : TF_Op<"TPUCompileSucceededAssert", [TF_MustExecute]> {
  let summary = "Asserts that compilation succeeded.";

  let description = [{
This op produces no output and closes the device during failure to ensure all
pending device interactions fail.

'compilation_status' is a serialized CompilationResultProto.
  }];

  let arguments = (ins
    TF_StrTensor:$compilation_status
  );

  let results = (outs);
}

def TF_TPUCopyWithLayoutOp : TF_Op<"TPUCopyWithLayout", [NoSideEffect]> {
  let summary = "Op that copies host tensor to device with specified layout.";

  let description = [{
For internal use only.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_Int64Tensor:$layout
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TPUEmbeddingActivationsOp : TF_Op<"TPUEmbeddingActivations", [NoSideEffect]> {
  let summary = "An op enabling differentiation of TPU Embeddings.";

  let description = [{
This op simply returns its first input, which is assumed to have been sliced
from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of
this op, and its first argument being a trainable Variable, enables automatic
differentiation of graphs containing embeddings via the TPU Embedding Python
libraries.
  }];

  let arguments = (ins
    Arg<TF_Float32Tensor, [{A trainable variable, enabling optimizers to find this op.}]>:$embedding_variable,
    Arg<TF_Float32Tensor, [{The embedding activations Tensor to return.}]>:$sliced_activations,

    Confined<I64Attr, [IntMinValue<0>]>:$table_id,
    Confined<I64Attr, [IntMinValue<0>]>:$lookup_id
  );

  let results = (outs
    TF_Float32Tensor:$output
  );
}

def TF_TPUExecuteOp : TF_Op<"TPUExecute", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def TF_TPUExecuteAndUpdateVariablesOp : TF_Op<"TPUExecuteAndUpdateVariables", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

  let hasVerifier = 1;
}

def TF_TPUGetLayoutOp : TF_Op<"TPUGetLayoutOp", [NoSideEffect]> {
  let summary = [{
Op that retrieves the layout of an input or output determined by TPUCompile.
  }];

  let description = [{
For internal use only.
  }];

  let arguments = (ins
    TF_StrTensor:$cache_key,

    I64Attr:$index,
    BoolAttr:$is_output
  );

  let results = (outs
    TF_Int64Tensor:$layout
  );
}

def TF_TPUOrdinalSelectorOp : TF_Op<"TPUOrdinalSelector", []> {
  let summary = "A TPU core selector Op.";

  let description = [{
This Op produces a set of TPU cores (for warm-up) or a single TPU core
(for regular inference) to execute the TPU program on. The output is
consumed by TPUPartitionedCall.
  }];

  let arguments = (ins);

  let results = (outs
    Res<TF_Int32Tensor, [{A vector 1 or more TPU cores.}]>:$device_ordinals
  );
}

def TF_TPUReplicateMetadataOp : TF_Op<"TPUReplicateMetadata", []> {
  let summary = [{
Metadata indicating how the TPU computation should be replicated.
  }];

  let description = [{
This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.
  }];

  let arguments = (ins
    Confined<I64Attr, [IntMinValue<0>]>:$num_replicas,
    DefaultValuedAttr<I64Attr, "1">:$num_cores_per_replica,
    DefaultValuedAttr<StrAttr, "\"\"">:$topology,
    DefaultValuedAttr<BoolAttr, "true">:$use_tpu,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$device_assignment,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$computation_shape,
    DefaultValuedAttr<StrArrayAttr, "{}">:$host_compute_core,
    DefaultValuedAttr<StrArrayAttr, "{}">:$padding_map,
    DefaultValuedAttr<StrAttr, "\"STEP_MARK_AT_ENTRY\"">:$step_marker_location,
    DefaultValuedAttr<BoolAttr, "false">:$allow_soft_placement,
    DefaultValuedAttr<BoolAttr, "false">:$use_spmd_for_xla_partitioning,
    DefaultValuedAttr<StrAttr, "\"\"">:$tpu_compile_options_proto
  );

  let results = (outs);
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
    DefaultValuedAttr<I64Attr, "-1">:$index,
    DefaultValuedAttr<BoolAttr, "false">:$is_packed
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultSizeAttr num_replicas = TF_DerivedResultSizeAttr<0>;
}

def TF_TPUReshardVariablesOp : TF_Op<"TPUReshardVariables", []> {
  let summary = "Op that reshards on-device TPU variables to specified state.";

  let description = [{
Op that reshards on-device TPU variables to specified state. Internal use only.

The sharding state is represented as the key of the compilation that generated
the sharding/unsharding programs along with the main program. new_format_key
specifies the desired state, and format_state_var is the current state of the
variables.
  }];

  let arguments = (ins
    Arg<Variadic<TF_ResourceTensor>, "", [TF_VariableRead, TF_VariableWrite]>:$vars,
    TF_StrTensor:$new_format_key,
    Arg<TF_ResourceTensor, "", [TF_VariableRead, TF_VariableWrite]>:$format_state_var
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_TPURoundRobinOp : TF_Op<"TPURoundRobin", []> {
  let summary = "Round-robin load balancing on TPU cores.";

  let description = [{
A load balancing op that round-robins among TPU cores.

This op round-robins between the integers in [0, NumTPUCoresVisiblePerHost]. It
is useful for interfacing with TensorFlow ops that take as input a TPU core on
which to execute computations, such as `TPUPartitionedCall`.

device_ordinal: An integer in [0, NumTPUCoresVisiblePerHost].
  }];

  let arguments = (ins);

  let results = (outs
    TF_Int32Tensor:$device_ordinal
  );
}

def TF_TakeDatasetOp : TF_Op<"TakeDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that contains `count` elements from the `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Arg<TF_Int64Tensor, [{A scalar representing the number of elements from the `input_dataset`
that should be taken. A value of `-1` indicates that all of `input_dataset`
is taken.}]>:$count,

    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_TakeWhileDatasetOp : TF_Op<"TakeWhileDataset", [NoSideEffect]> {
  let summary = [{
Creates a dataset that stops iteration when predicate` is false.
  }];

  let description = [{
The `predicate` function must return a scalar boolean and accept the
following arguments:

* One tensor for each component of an element of `input_dataset`.
* One tensor for each value in `other_arguments`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Arg<Variadic<TF_Tensor>, [{A list of tensors, typically values that were captured when
building a closure for `predicate`.}]>:$other_arguments,

    SymbolRefAttr:$predicate,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_TanOp : TF_Op<"Tan", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes tan of x element-wise.";

  let description = [{
Given an input tensor, this function computes tangent of every
  element in the tensor. Input range is `(-inf, inf)` and
  output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
  is returned.

  ```python
  x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
  tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
  ```
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$x
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TanhOp : TF_Op<"Tanh", [NoSideEffect, TF_LayoutAgnostic, TF_SameOperandsAndResultTypeResolveRef]> {
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
}

def TF_TanhGradOp : TF_Op<"TanhGrad", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
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

def TF_TensorArrayCloseV3Op : TF_Op<"TensorArrayCloseV3", []> {
  let summary = "Delete the TensorArray from its resource container.";

  let description = [{
This enables the user to close and release the resource in the middle
of a step/run.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray (output of TensorArray or TensorArrayGrad).}], [TF_TensorArrayFree]>:$handle
  );

  let results = (outs);
}

def TF_TensorArrayConcatV3Op : TF_Op<"TensorArrayConcatV3", []> {
  let summary = "Concat the elements from the TensorArray into value `value`.";

  let description = [{
Takes `T` elements of shapes

  ```
  (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
  ```

and concatenates them into a Tensor of shape:

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```

All elements must have the same shape (excepting the first dimension).
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead]>:$handle,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in,

    DefaultValuedAttr<TF_ShapeAttr, "llvm::None">:$element_shape_except0
  );

  let results = (outs
    Res<TF_Tensor, [{All of the elements in the TensorArray, concatenated along the first
axis.}]>:$value,
    Res<TF_Int64Tensor, [{A vector of the row sizes of the original T elements in the
value output.  In the example above, this would be the values:
`(n1, n2, ..., n(T-1))`.}]>:$lengths
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorArrayGatherV3Op : TF_Op<"TensorArrayGatherV3", []> {
  let summary = [{
Gather specific elements from the TensorArray into output `value`.
  }];

  let description = [{
All elements selected by `indices` must have the same shape.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead]>:$handle,
    Arg<TF_Int32Tensor, [{The locations in the TensorArray from which to read tensor elements.}]>:$indices,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in,

    DefaultValuedAttr<TF_ShapeAttr, "llvm::None">:$element_shape
  );

  let results = (outs
    Res<TF_Tensor, [{All of the elements in the TensorArray, concatenated along a new
axis (the new dimension 0).}]>:$value
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorArrayGradV3Op : TF_Op<"TensorArrayGradV3", []> {
  let summary = [{
Creates a TensorArray for storing the gradients of values in the given handle.
  }];

  let description = [{
If the given TensorArray gradient already exists, returns a reference to it.

Locks the size of the original TensorArray by disabling its dynamic size flag.

**A note about the input flow_in:**

The handle flow_in forces the execution of the gradient lookup to occur
only after certain other operations have occurred.  For example, when
the forward TensorArray is dynamically sized, writes to this TensorArray
may resize the object.  The gradient TensorArray is statically sized based
on the size of the forward TensorArray when this operation executes.
Furthermore, the size of the forward TensorArray is frozen by this call.
As a result, the flow is used to ensure that the call to generate the gradient
TensorArray only happens after all writes are executed.

In the case of dynamically sized TensorArrays, gradient computation should
only be performed on read operations that have themselves been chained via
flow to occur only after all writes have executed. That way the final size
of the forward TensorArray is known when this operation is called.

**A note about the source attribute:**

TensorArray gradient calls use an accumulator TensorArray object.  If
multiple gradients are calculated and run in the same session, the multiple
gradient nodes may accidentally flow through the same accumulator TensorArray.
This double counts and generally breaks the TensorArray gradient flow.

The solution is to identify which gradient call this particular
TensorArray gradient is being called in.  This is performed by identifying
a unique string (e.g. "gradients", "gradients_1", ...) from the input
gradient Tensor's name.  This string is used as a suffix when creating
the TensorArray gradient object here (the attribute `source`).

The attribute `source` is added as a suffix to the forward TensorArray's
name when performing the creation / lookup, so that each separate gradient
calculation gets its own TensorArray accumulator.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to the forward TensorArray.}], [TF_TensorArrayRead, TF_TensorArrayWrite]>:$handle,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in,

    StrAttr:$source
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_TensorArrayAlloc]>:$grad_handle,
    TF_Float32Tensor:$flow_out
  );
}

def TF_TensorArrayReadV3Op : TF_Op<"TensorArrayReadV3", []> {
  let summary = "Read an element from the TensorArray into output `value`.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead]>:$handle,
    TF_Int32Tensor:$index,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor that is read from the TensorArray.}]>:$value
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorArrayScatterV3Op : TF_Op<"TensorArrayScatterV3", []> {
  let summary = [{
Scatter the data from the input value into specific TensorArray elements.
  }];

  let description = [{
`indices` must be a vector, its length must match the first dim of `value`.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead, TF_TensorArrayWrite]>:$handle,
    Arg<TF_Int32Tensor, [{The locations at which to write the tensor elements.}]>:$indices,
    Arg<TF_Tensor, [{The concatenated tensor to write to the TensorArray.}]>:$value,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in
  );

  let results = (outs
    Res<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_TensorArraySizeV3Op : TF_Op<"TensorArraySizeV3", []> {
  let summary = "Get the current size of the TensorArray.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray (output of TensorArray or TensorArrayGrad).}], [TF_TensorArrayRead]>:$handle,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in
  );

  let results = (outs
    Res<TF_Int32Tensor, [{The current size of the TensorArray.}]>:$size
  );
}

def TF_TensorArraySplitV3Op : TF_Op<"TensorArraySplitV3", []> {
  let summary = [{
Split the data from the input value into TensorArray elements.
  }];

  let description = [{
Assuming that `lengths` takes on values

  ```(n0, n1, ..., n(T-1))```

and that `value` has shape

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,

this splits values into a TensorArray with T tensors.

TensorArray index t will be the subtensor of values with starting position

  ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```

and having size

  ```nt x d0 x d1 x ...```
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead, TF_TensorArrayWrite]>:$handle,
    Arg<TF_Tensor, [{The concatenated tensor to write to the TensorArray.}]>:$value,
    Arg<TF_Int64Tensor, [{The vector of lengths, how to split the rows of value into the
TensorArray.}]>:$lengths,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in
  );

  let results = (outs
    Res<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorArrayV3Op : TF_Op<"TensorArrayV3", []> {
  let summary = "An array of Tensors of given size.";

  let description = [{
Write data via Write and read via Read or Pack.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{The size of the array.}]>:$size,

    TypeAttr:$dtype,
    DefaultValuedAttr<TF_ShapeAttr, "llvm::None">:$element_shape,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_size,
    DefaultValuedAttr<BoolAttr, "true">:$clear_after_read,
    DefaultValuedAttr<BoolAttr, "false">:$identical_element_shapes,
    DefaultValuedAttr<StrAttr, "\"\"">:$tensor_array_name
  );

  let results = (outs
    Res<TF_ResourceTensor, [{The handle to the TensorArray.}], [TF_TensorArrayAlloc]>:$handle,
    Res<TF_Float32Tensor, [{A scalar used to control gradient flow.}]>:$flow
  );
}

def TF_TensorArrayWriteV3Op : TF_Op<"TensorArrayWriteV3", []> {
  let summary = "Push an element onto the tensor_array.";

  let arguments = (ins
    Arg<TF_ResourceTensor, [{The handle to a TensorArray.}], [TF_TensorArrayRead, TF_TensorArrayWrite]>:$handle,
    Arg<TF_Int32Tensor, [{The position to write to inside the TensorArray.}]>:$index,
    Arg<TF_Tensor, [{The tensor to write to the TensorArray.}]>:$value,
    Arg<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_in
  );

  let results = (outs
    Res<TF_Float32Tensor, [{A float scalar that enforces proper chaining of operations.}]>:$flow_out
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF_TensorListConcatV2Op : TF_Op<"TensorListConcatV2", [NoSideEffect]> {
  let summary = "Concats all tensors in the list along the 0th dimension.";

  let description = [{
Requires that all tensors have the same shape except the first dimension.

input_handle: The input list.
element_shape: The shape of the uninitialized elements in the list. If the first
  dimension is not -1, it is assumed that all list elements have the same
  leading dim.
leading_dims: The list of leading dims of uninitialized list elements. Used if
  the leading dim of input_handle.element_shape or the element_shape input arg
  is not already set.
tensor: The concated result.
lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_I32OrI64Tensor:$element_shape,
    TF_Int64Tensor:$leading_dims
  );

  let results = (outs
    TF_Tensor:$tensor,
    TF_Int64Tensor:$lengths
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorListElementShapeOp : TF_Op<"TensorListElementShape", [NoSideEffect]> {
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

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorListGatherOp : TF_Op<"TensorListGather", [NoSideEffect]> {
  let summary = "Creates a Tensor by indexing into the TensorList.";

  let description = [{
Each row in the produced Tensor corresponds to the element in the TensorList
specified by the given index (see `tf.gather`).

input_handle: The input tensor list.
indices: The indices used to index into the list.
values: The tensor.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$indices,
    TF_Int32Tensor:$element_shape
  );

  let results = (outs
    TF_Tensor:$values
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorListGetItemOp : TF_Op<"TensorListGetItem", [NoSideEffect]> {
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
    TF_Int32Tensor:$length
  );
}

def TF_TensorListPopBackOp : TF_Op<"TensorListPopBack", [NoSideEffect]> {
  let summary = [{
Returns the last element of the input list as well as a list with all but that element.
  }];

  let description = [{
Fails if the list is empty.

input_handle: the input list
tensor: the withdrawn last element of the list
element_dtype: the type of elements in the list
element_shape: the shape of the output tensor
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$element_shape
  );

  let results = (outs
    TF_VariantTensor:$output_handle,
    TF_Tensor:$tensor
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<1>;
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
    TF_Int32Tensor:$size
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );
}

def TF_TensorListScatterIntoExistingListOp : TF_Op<"TensorListScatterIntoExistingList", [NoSideEffect]> {
  let summary = "Scatters tensor at indices in an input list.";

  let description = [{
Each member of the TensorList corresponds to one row of the input tensor,
specified by the given index (see `tf.gather`).

input_handle: The list to scatter into.
tensor: The input tensor.
indices: The indices used to index into the list.
output_handle: The TensorList.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Tensor:$tensor,
    TF_Int32Tensor:$indices
  );

  let results = (outs
    TF_VariantTensor:$output_handle
  );

  TF_DerivedOperandTypeAttr element_dtype = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorListSetItemOp : TF_Op<"TensorListSetItem", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_VariantTensor:$input_handle,
    TF_Int32Tensor:$index,
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
    TF_Int32Tensor:$element_shape,

    DefaultValuedAttr<I64Attr, "-1">:$num_elements
  );

  let results = (outs
    TF_Tensor:$tensor
  );

  TF_DerivedResultTypeAttr element_dtype = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_TensorScatterAddOp : TF_Op<"TensorScatterAdd", [NoSideEffect]> {
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

Note: on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to copy/update.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates
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

def TF_TensorScatterMaxOp : TF_Op<"TensorScatterMax", [NoSideEffect]> {
  let summary = [{
Apply a sparse update to a tensor taking the element-wise maximum.
  }];

  let description = [{
Returns a new tensor copied from `tensor` whose values are element-wise maximum between
tensor and updates according to the indices.

>>> tensor = [0, 0, 0, 0, 0, 0, 0, 0] 
>>> indices = [[1], [4], [5]]
>>> updates = [1, -1, 1]
>>> tf.tensor_scatter_nd_max(tensor, indices, updates).numpy()
array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int32)

Refer to `tf.tensor_scatter_nd_update` for more details.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to update.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor copied from tensor whose values are element-wise maximum between tensor and updates according to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorScatterMinOp : TF_Op<"TensorScatterMin", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to update.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor copied from tensor whose values are element-wise minimum between tensor and updates according to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorScatterSubOp : TF_Op<"TensorScatterSub", [NoSideEffect]> {
  let summary = [{
Subtracts sparse `updates` from an existing tensor according to `indices`.
  }];

  let description = [{
This operation creates a new tensor by subtracting sparse `updates` from the
passed in `tensor`.
This operation is very similar to `tf.scatter_nd_sub`, except that the updates
are subtracted from an existing tensor (as opposed to a variable). If the memory
for the existing tensor cannot be re-used, a copy is made and updated.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:

    indices.shape[-1] <= shape.rank

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape

    indices.shape[:-1] + shape[indices.shape[-1]:]

The simplest form of tensor_scatter_sub is to subtract individual elements
from a tensor by index. For example, say we want to insert 4 scattered elements
in a rank-1 tensor with 8 elements.

In Python, this scatter subtract operation would look like this:

```python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    tensor = tf.ones([8], dtype=tf.int32)
    updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

    [1, -10, 1, -9, -8, 1, 1, -11]

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter add operation would look like this:

```python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    tensor = tf.ones([4, 4, 4],dtype=tf.int32)
    updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

    [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
     [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to copy/update.}]>:$tensor,
    Arg<TF_I32OrI64Tensor, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates
  );

  let results = (outs
    Res<TF_Tensor, [{A new tensor copied from tensor and updates subtracted according to the indices.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_TensorScatterUpdateOp : TF_Op<"TensorScatterUpdate", [NoSideEffect]> {
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

If an out of bound index is found on CPU, an error is returned.

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

For usage examples see the python [tf.tensor_scatter_nd_update](
https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{Tensor to copy/update.}]>:$tensor,
    Arg<TensorOf<[TF_Int16, TF_Int32, TF_Int64, TF_Uint16]>, [{Index tensor.}]>:$indices,
    Arg<TF_Tensor, [{Updates to scatter into output.}]>:$updates
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

def TF_TensorSliceDatasetOp : TF_Op<"TensorSliceDataset", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Creates a dataset that emits each dim-0 slice of `components` once.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$components,

    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "false">:$is_files,
    DefaultValuedAttr<StrAttr, "\"\"">:$metadata,
    DefaultValuedAttr<BoolAttr, "false">:$replicate_on_split
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Toutput_types = TF_DerivedOperandTypeListAttr<0>;
}

def TF_TensorStridedSliceUpdateOp : TF_Op<"TensorStridedSliceUpdate", [NoSideEffect]> {
  let summary = "Assign `value` to the sliced l-value reference of `input`.";

  let description = [{
The values of `value` are assigned to the positions in the tensor `input` that
are selected by the slice parameters. The slice parameters `begin` `end`
`strides` etc. work exactly as in `StridedSlice`.

NOTE this op currently does not support broadcasting and so `value`'s shape
must be exactly the shape produced by the slice of `input`.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$begin,
    TF_I32OrI64Tensor:$end,
    TF_I32OrI64Tensor:$strides,
    TF_Tensor:$value,

    DefaultValuedAttr<I64Attr, "0">:$begin_mask,
    DefaultValuedAttr<I64Attr, "0">:$end_mask,
    DefaultValuedAttr<I64Attr, "0">:$ellipsis_mask,
    DefaultValuedAttr<I64Attr, "0">:$new_axis_mask,
    DefaultValuedAttr<I64Attr, "0">:$shrink_axis_mask
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Index = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    Arg<TF_Tensor, [{1-D or higher.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{1-D. Length must be the same as the number of dimensions in `input`}]>:$multiples
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tmultiples = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_TimestampOp : TF_Op<"Timestamp", []> {
  let summary = "Provides the time since epoch in seconds.";

  let description = [{
Returns the timestamp as a `float64` for seconds since the Unix epoch.

Note: the timestamp is computed when the op is executed, not when it is added
to the graph.
  }];

  let arguments = (ins);

  let results = (outs
    TF_Float64Tensor:$ts
  );
}

def TF_TopKUniqueOp : TF_Op<"TopKUnique", [NoSideEffect]> {
  let summary = "Returns the TopK unique values in the array in sorted order.";

  let description = [{
The running time is proportional to the product of K and the input
size. Sorting the whole array is more efficient for sufficiently large
values of K. The median-of-medians algorithm is probably faster, but
difficult to implement efficiently in XLA. If there are fewer than K
unique numbers (not NANs), the results are padded with negative
infinity. NaNs are never returned. Subnormal numbers are flushed to
zero. If an element appears at multiple indices, the highest index is
returned. If a TopK element never appears in the input due to padding
values, the indices are padded with negative one. If a padding value
appears in the input and padding is needed, the highest index of the
padding value will be returned. The semantics are not the same as
kth_order_statistic.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input,

    I64Attr:$k
  );

  let results = (outs
    TF_Float32Tensor:$topk,
    TF_Int32Tensor:$topk_indices
  );
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
    Arg<TF_IntOrFpTensor, [{1-D or higher with last dimension at least `k`.}]>:$input,
    Arg<TF_Int32Tensor, [{0-D.  Number of top elements to look for along the last dimension (along each
row for matrices).}]>:$k,

    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{The `k` largest elements along each last dimensional slice.}]>:$values,
    Res<TF_Int32Tensor, [{The indices of `values` within the last dimension of `input`.}]>:$indices
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_TopKWithUniqueOp : TF_Op<"TopKWithUnique", [NoSideEffect]> {
  let summary = "Returns the TopK values in the array in sorted order.";

  let description = [{
This is a combination of MakeUnique and TopKUnique. The returned top-K will
have its lower bits replaced by iota, thus it will be close to the original
value but not exactly the same. The running time is proportional to the product
of K and the input size. NaNs are never returned. Subnormal numbers are flushed
to zero.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input,

    I64Attr:$k
  );

  let results = (outs
    TF_Float32Tensor:$topk,
    TF_Int32Tensor:$topk_indices
  );
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
    OpBuilder<(ins "Value":$x, "Value":$perm)>
  ];

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_TridiagonalSolveOp : TF_Op<"TridiagonalSolve", [NoSideEffect]> {
  let summary = "Solves tridiagonal systems of equations.";

  let description = [{
Solves tridiagonal systems of equations.
  Supports batch dimensions and multiple right-hand sides per each left-hand
  side.
  On CPU, solution is computed via Gaussian elimination with or without partial
  pivoting, depending on `partial_pivoting` attribute. On GPU, Nvidia's cuSPARSE
  library is used: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv
  Partial pivoting is not yet supported by XLA backends.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float32, TF_Float64]>, [{Tensor of shape `[..., 3, M]` whose innermost 2 dimensions represent the
tridiagonal matrices with three rows being the superdiagonal, diagonals, and
subdiagonals, in order. The last element of the superdiagonal and the first
element of the subdiagonal is ignored.}]>:$diagonals,
    Arg<TensorOf<[TF_Complex128, TF_Complex64, TF_Float32, TF_Float64]>, [{Tensor of shape `[..., M, K]`, representing K right-hand sides per each
left-hand side.}]>:$rhs,

    DefaultValuedAttr<BoolAttr, "true">:$partial_pivoting,
    DefaultValuedAttr<BoolAttr, "false">:$perturb_singular
  );

  let results = (outs
    Res<TensorOf<[TF_Complex128, TF_Complex64, TF_Float32, TF_Float64]>, [{Tensor of shape `[..., M, K]` containing the solutions}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TruncateDivOp : TF_Op<"TruncateDiv", [NoSideEffect, ResultsBroadcastableShape]>,
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$x,
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$y
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_TruncateModOp : TF_Op<"TruncateMod", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                       WithBroadcastableBinOpBuilder {
  let summary = [{
Returns element-wise remainder of division. This emulates C semantics in that
  }];

  let description = [{
the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
y + truncate_mod(x, y) = x`.

*NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
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

def TF_TruncatedNormalOp : TF_Op<"TruncatedNormal", [TF_CannotDuplicate]> {
  let summary = "Outputs random values from a truncated normal distribution.";

  let description = [{
The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.
  }];

  let arguments = (ins
    Arg<TF_I32OrI64Tensor, [{The shape of the output tensor.}]>:$shape,

    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    Res<TF_FloatTensor, [{A tensor of the specified shape filled with random truncated normal
values.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_UncompressElementOp : TF_Op<"UncompressElement", [NoSideEffect]> {
  let summary = "Uncompresses a compressed dataset element.";

  let arguments = (ins
    TF_VariantTensor:$compressed
  );

  let results = (outs
    Variadic<TF_Tensor>:$components
  );

  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr output_types = TF_DerivedResultTypeListAttr<0>;
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
    Arg<TF_Tensor, [{1-D.}]>:$x
  );

  let results = (outs
    Res<TF_Tensor, [{1-D.}]>:$y,
    Res<TF_I32OrI64Tensor, [{1-D.}]>:$idx
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
    Arg<TF_Tensor, [{1-D or higher, with `axis` dimension size equal to `num`.}]>:$value,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{The list of tensors unpacked from `value`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultSizeAttr num = TF_DerivedResultSizeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_UnsortedSegmentMaxOp : TF_Op<"UnsortedSegmentMax", [NoSideEffect]> {
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

def TF_UnsortedSegmentMinOp : TF_Op<"UnsortedSegmentMin", [NoSideEffect]> {
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

def TF_UnsortedSegmentProdOp : TF_Op<"UnsortedSegmentProd", [NoSideEffect]> {
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    Arg<TF_I32OrI64Tensor, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
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
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$data,
    Arg<TF_I32OrI64Tensor, [{A tensor whose shape is a prefix of `data.shape`.
The values must be less than `num_segments`.

Caution: The values are always validated to be in range on CPU, never validated
on GPU.}]>:$segment_ids,
    TF_I32OrI64Tensor:$num_segments
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Has same shape as data, except for the first `segment_ids.rank`
dimensions, which are replaced with a single dimension which has size
`num_segments`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tnumsegments = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_UpperBoundOp : TF_Op<"UpperBound", [NoSideEffect]> {
  let summary = [{
Applies upper_bound(sorted_search_values, values) along each row.
  }];

  let description = [{
Each set of rows with the same index in (sorted_inputs, values) is treated
independently.  The resulting row is the equivalent of calling
`np.searchsorted(sorted_inputs, values, side='right')`.

The result is not a global index to the entire
`Tensor`, but rather just the index in the last dimension.

A 2-D example:
  sorted_sequence = [[0, 3, 9, 9, 10],
                     [1, 2, 3, 4, 5]]
  values = [[2, 4, 9],
            [0, 2, 6]]

  result = UpperBound(sorted_sequence, values)

  result == [[1, 2, 4],
             [0, 2, 5]]
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{2-D Tensor where each row is ordered.}]>:$sorted_inputs,
    Arg<TF_Tensor, [{2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
the values that will be searched for in `sorted_search_values`.}]>:$values
  );

  let results = (outs
    Res<TF_I32OrI64Tensor, [{A `Tensor` with the same shape as `values`.  It contains the last scalar index
into the last dimension where values can be inserted without changing the
ordered property.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;
}

def TF_VarIsInitializedOp : TF_Op<"VarIsInitializedOp", []> {
  let summary = [{
Checks whether a resource handle-based variable has been initialized.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, [{the input resource handle.}], [TF_VariableRead]>:$resource
  );

  let results = (outs
    Res<TF_BoolTensor, [{a scalar boolean which is true if the variable has been
initialized.}]>:$is_initialized
  );

  let hasCanonicalizer = 1;
}

def TF_VariableOp : TF_Op<"Variable", []> {
  let summary = "Use VariableV2 instead.";

  let arguments = (ins
    TF_ShapeAttr:$shape,
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name
  );

  let results = (outs
    TF_Tensor:$ref
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let hasCanonicalizer = 1;
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
    Arg<TF_ResourceTensor, "", [TF_VariableRead]>:$input
  );

  let results = (outs
    TF_I32OrI64Tensor:$output
  );

  TF_DerivedResultTypeAttr out_type = TF_DerivedResultTypeAttr<0>;

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TF_VariableV2Op : TF_Op<"VariableV2", []> {
  let summary = [{
Holds state in the form of a tensor that persists across steps.
  }];

  let description = [{
Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow.
  }];

  let arguments = (ins
    TF_ShapeAttr:$shape,
    DefaultValuedAttr<StrAttr, "\"\"">:$container,
    DefaultValuedAttr<StrAttr, "\"\"">:$shared_name
  );

  let results = (outs
    Res<TF_Tensor, [{A reference to the variable tensor.}]>:$ref
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
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
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input
  );

  let results = (outs
    TF_Int64Tensor:$index
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XdivyOp : TF_Op<"Xdivy", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if x == 0, and x / y otherwise, elementwise.";

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$x,
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$y
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_XlaAllReduceOp : TF_Op<"XlaAllReduce", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA AllReduce operator";

  let description = [{
documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32]>, [{Array or a non-empty tuple of arrays to reduce across replicas.}]>:$input,
    Arg<TF_Int32Tensor, [{Groups between which the reductions are performed.}]>:$group_assignment,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean"]>:$reduce_op,
    TF_AnyStrAttrOf<["CrossReplica", "CrossReplicaAndPartition"]>:$mode
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaBroadcastHelperOp : TF_Op<"XlaBroadcastHelper", [InferTensorType, NoSideEffect, TF_NoConstantFold]> {
  let summary = "Helper operator for performing XLA-style broadcasts";

  let description = [{
Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
for binary operators.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the LHS input tensor}]>:$lhs,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the RHS input tensor}]>:$rhs,
    Arg<TF_I32OrI64Tensor, [{an XLA-style broadcast dimension specification}]>:$broadcast_dims
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the broadcasted LHS tensor}]>:$lhs_output,
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the broadcasted RHS tensor}]>:$rhs_output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];
}

def TF_XlaClusterOutputOp : TF_Op<"XlaClusterOutput", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
Operator that connects the output of an XLA computation to other consumer graph nodes.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$outputs
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaConvOp : TF_Op<"XlaConv", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA ConvGeneralDilated operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor}]>:$lhs,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the kernel tensor}]>:$rhs,
    Arg<TF_I32OrI64Tensor, [{the inter-window strides}]>:$window_strides,
    Arg<TF_I32OrI64Tensor, [{the padding to apply at the start and end of each input dimensions}]>:$padding,
    Arg<TF_I32OrI64Tensor, [{dilation to apply between input elements}]>:$lhs_dilation,
    Arg<TF_I32OrI64Tensor, [{dilation to apply between kernel elements}]>:$rhs_dilation,
    Arg<TF_I32OrI64Tensor, [{number of feature groups for grouped convolution.}]>:$feature_group_count,

    StrAttr:$dimension_numbers,
    StrAttr:$precision_config
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;
}

def TF_XlaConvV2Op : TF_Op<"XlaConvV2", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA ConvGeneralDilated operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{input tensor}]>:$lhs,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{kernel tensor}]>:$rhs,
    Arg<TF_I32OrI64Tensor, [{inter-window strides}]>:$window_strides,
    Arg<TF_I32OrI64Tensor, [{padding to apply at the start and end of each input dimensions}]>:$padding,
    Arg<TF_I32OrI64Tensor, [{dilation to apply between input elements}]>:$lhs_dilation,
    Arg<TF_I32OrI64Tensor, [{dilation to apply between kernel elements}]>:$rhs_dilation,
    Arg<TF_I32OrI64Tensor, [{number of feature groups for grouped convolution.}]>:$feature_group_count,

    StrAttr:$dimension_numbers,
    StrAttr:$precision_config,
    DefaultValuedAttr<I64Attr, "1">:$batch_group_count
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr preferred_element_type = TF_DerivedResultTypeAttr<0>;
}

def TF_XlaDotOp : TF_Op<"XlaDot", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA DotGeneral operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the LHS tensor}]>:$lhs,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the RHS tensor}]>:$rhs,

    StrAttr:$dimension_numbers,
    StrAttr:$precision_config
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaDotV2Op : TF_Op<"XlaDotV2", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA DotGeneral operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the LHS tensor}]>:$lhs,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the RHS tensor}]>:$rhs,

    StrAttr:$dimension_numbers,
    StrAttr:$precision_config
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr LhsT = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr RhsT = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr preferred_element_type = TF_DerivedResultTypeAttr<0>;
}

def TF_XlaDynamicSliceOp : TF_Op<"XlaDynamicSlice", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA DynamicSlice operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
.

DynamicSlice extracts a sub-array from the input array at dynamic
start_indices. The size of the slice in each dimension is passed in
size_indices, which specify the end point of exclusive slice intervals in each
dimension -- [start, start + size). The shape of start_indices must have rank 1,
with dimension size equal to the rank of operand.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A `Tensor` of type T.}]>:$input,
    Arg<TF_I32OrI64Tensor, [{List of N integers containing the slice size for each
dimension. Each value must be strictly greater than zero, and start + size
must be less than or equal to the size of the dimension to avoid
implementation defined behavior.}]>:$start_indices,
    TF_I32OrI64Tensor:$size_indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_XlaDynamicUpdateSliceOp : TF_Op<"XlaDynamicUpdateSlice", [NoSideEffect, TF_NoConstantFold]> {
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

def TF_XlaEinsumOp : TF_Op<"XlaEinsum", [NoSideEffect, TF_NoConstantFold]> {
  let summary = [{
An op which supports basic einsum op with 2 inputs and 1 output.
  }];

  let description = [{
This op has better TPU performance since it doesn't have explicitly reshape and
transpose operations as tf.einsum does.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Complex64, TF_Float32]>:$a,
    TensorOf<[TF_Bfloat16, TF_Complex64, TF_Float32]>:$b,

    StrAttr:$equation
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex64, TF_Float32]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaGatherOp : TF_Op<"XlaGather", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA Gather operator documented at";

  let description = [{
https://www.tensorflow.org/xla/operation_semantics#gather
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The array we're gathering from.}]>:$operand,
    Arg<TF_I32OrI64Tensor, [{Array containing the starting indices of the slices we gather.}]>:$start_indices,
    Arg<TF_I32OrI64Tensor, [{slice_sizes[i] is the bounds for the slice on dimension i.}]>:$slice_sizes,

    StrAttr:$dimension_numbers,
    BoolAttr:$indices_are_sorted
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_XlaKeyValueSortOp : TF_Op<"XlaKeyValueSort", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA Sort operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.
  }];

  let arguments = (ins
    Arg<TF_IntOrFpTensor, [{A `Tensor` of type K.}]>:$keys,
    Arg<TF_Tensor, [{A `Tensor` of type V.}]>:$values
  );

  let results = (outs
    Res<TF_IntOrFpTensor, [{A `Tensor` of type K.}]>:$sorted_keys,
    Res<TF_Tensor, [{A `Tensor` of type V.}]>:$sorted_values
  );

  TF_DerivedOperandTypeAttr K = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr V = TF_DerivedOperandTypeAttr<1>;
}

def TF_XlaOptimizationBarrierOp : TF_Op<"XlaOptimizationBarrier", [NoSideEffect]> {
  let summary = "Wraps the XLA OptimizationBarrier operator.";

  let description = [{
Documented at https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A Tuple of Arrays of any type.}]>:$input
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

def TF_XlaPadOp : TF_Op<"XlaPad", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Wraps the XLA Pad operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#pad
.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A `Tensor` of type T.}]>:$input,
    Arg<TF_Tensor, [{A scalar `Tensor` of type T.}]>:$padding_value,
    Arg<TF_I32OrI64Tensor, [{the padding to apply at the start of each input dimensions. Must
be a compile-time constant 1D tensor of length equal to rank of input.}]>:$padding_low,
    Arg<TF_I32OrI64Tensor, [{the padding to apply at the end of each input dimension. Must
be a compile-time constant 1D tensor of length equal to rank of input.}]>:$padding_high,
    Arg<TF_I32OrI64Tensor, [{the padding to apply between each input element. Must
be a compile-time constant 1D tensor of length equal to rank of input,
containing only non-negative values.}]>:$padding_interior
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type T.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;
}

def TF_XlaRecvOp : TF_Op<"XlaRecv", [TF_RecvSideEffect]> {
  let summary = [{
Receives the named tensor from another XLA computation. Wraps the XLA Recv
  }];

  let description = [{
operator documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#recv .
  }];

  let arguments = (ins
    StrAttr:$tensor_name,
    TF_ShapeAttr:$shape
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor to receive.}]>:$tensor
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_XlaRecvFromHostOp : TF_Op<"XlaRecvFromHost", [TF_RecvSideEffect]> {
  let summary = "An op to receive a tensor from the host.";

  let description = [{
output: the tensor that will be received from the host.
Toutput: element type for output.
shape: shape for output.
key: A unique identifier for this region used to match up host transfers.
  }];

  let arguments = (ins
    TF_ShapeAttr:$shape,
    StrAttr:$key
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr Toutput = TF_DerivedResultTypeAttr<0>;
}

def TF_XlaRecvTPUEmbeddingActivationsOp : TF_Op<"XlaRecvTPUEmbeddingActivations", [TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "An op that receives embeddng activations on the TPU.";

  let description = [{
The TPU system performs the embedding lookups and aggregations. The results of
these aggregations are visible to the Tensorflow Graph as the outputs of a
XlaRecvTPUEmbeddingActivations Op. This op returns a list containing one
Tensor of activations per table specified in the model.
  }];

  let arguments = (ins
    Arg<TF_VariantTensor, [{A Tensor with type=DT_VARIANT containing the deduplication
data. The tensor is an XLA nested tuple containing N elements (where N is
the ratio of the number of embedding to tensor cores per TPU chip). Each
element of the nested tuple is a tuple of rank 1 tensors. Each tensor either
contains indices (DT_UINT32) for embedding lookup on the TensorCore or
weights (DT_FLOAT) to apply to the output of the embedding lookup operation.}]>:$deduplication_data,

    StrAttr:$config
  );

  let results = (outs
    Res<Variadic<TF_Float32Tensor>, [{A TensorList of embedding activations containing one Tensor per
embedding table in the model.}]>:$outputs
  );

  TF_DerivedResultSizeAttr num_tables = TF_DerivedResultSizeAttr<0>;
}

def TF_XlaRecvTPUEmbeddingDeduplicationDataOp : TF_Op<"XlaRecvTPUEmbeddingDeduplicationData", []> {
  let summary = [{
Receives deduplication data (indices and weights) from the embedding core.
  }];

  let description = [{
The deduplication data is a Tensor with type=DT_VARIANT. The tensor itself is an
XLA nested tuple containing N elements (where N is the ratio of the number of
embedding to tensor cores per TPU chip). Each element of the nested tuple is a
tuple of rank 1 tensors. Each tensor either contains indices (DT_UINT32) for
embedding lookup on the TensorCore or weights (DT_FLOAT) to apply to the output
of the embedding lookup operation.
  }];

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    TF_VariantTensor:$output
  );
}

def TF_XlaReduceOp : TF_Op<"XlaReduce", [NoSideEffect]> {
  let summary = "Wraps the XLA Reduce operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#reduce .
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor}]>:$input,
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a scalar representing the initial value for the reduction}]>:$init_value,

    I64ArrayAttr:$dimensions_to_reduce,
    SymbolRefAttr:$reducer
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_XlaReduceScatterOp : TF_Op<"XlaReduceScatter", [NoSideEffect]> {
  let summary = "Wraps the XLA ReduceScatter operator";

  let description = [{
documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32]>, [{Array or a non-empty tuple of arrays to reduce across replicas.}]>:$input,
    Arg<TF_Int32Tensor, [{Groups between which the reductions are performed.}]>:$group_assignment,
    Arg<TF_Int32Tensor, [{Dimension to scatter.}]>:$scatter_dimension,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean"]>:$reduce_op
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaReduceWindowOp : TF_Op<"XlaReduceWindow", [NoSideEffect]> {
  let summary = "Wraps the XLA ReduceWindow operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor}]>:$input,
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a scalar representing the initial value for the reduction}]>:$init_value,
    Arg<TF_I32OrI64Tensor, [{the shape of the window}]>:$window_dimensions,
    Arg<TF_I32OrI64Tensor, [{the inter-window strides}]>:$window_strides,
    TF_I32OrI64Tensor:$base_dilations,
    TF_I32OrI64Tensor:$window_dilations,
    Arg<TF_I32OrI64Tensor, [{the padding to apply at the start and end of each input dimensions}]>:$padding,

    SymbolRefAttr:$computation
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<2>;

  let hasVerifier = 1;
}

def TF_XlaRemoveDynamicDimensionSizeOp : TF_Op<"XlaRemoveDynamicDimensionSize", [NoSideEffect]> {
  let summary = "Inverse of XlaSetDynamicDimensionSize.";

  let description = [{
Make an xla bounded dynamic dimension into a static dimension. The bound of the
size of dimension `dim_index` becomes the static dimension size.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_Int32Tensor:$dim_index
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaReplicaIdOp : TF_Op<"XlaReplicaId", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Replica ID.";

  let arguments = (ins);

  let results = (outs
    TF_Int32Tensor:$id
  );

  // Constant folding is disabled for this op as it is a runtime op and can't
  // constant folded at the compile time.
}

def TF_XlaRngBitGeneratorOp : TF_Op<"XlaRngBitGenerator", [NoSideEffect]> {
  let summary = "Stateless PRNG bit generator.";

  let description = [{
Wraps the XLA RngBitGenerator operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.
  }];

  let arguments = (ins
    Arg<TF_Int32Tensor, [{The PRNG algorithm to use, one of
tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.}]>:$algorithm,
    Arg<TF_Uint64Tensor, [{Initial state for the PRNG algorithm. For THREEFRY, it should be
a u64[2] and for PHILOX a u64[3].}]>:$initial_state,
    Arg<TF_I32OrI64Tensor, [{The output shape of the generated data.}]>:$shape
  );

  let results = (outs
    TF_Uint64Tensor:$output_key,
    TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>:$output
  );

  TF_DerivedOperandTypeAttr Tshape = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<1>;
}

def TF_XlaScatterOp : TF_Op<"XlaScatter", [NoSideEffect]> {
  let summary = "Wraps the XLA Scatter operator documented at";

  let description = [{
https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Array to be scattered into.}]>:$operand,
    Arg<TF_I32OrI64Tensor, [{Array containing the starting indices of the slices that must
be scattered to.}]>:$scatter_indices,
    Arg<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Array containing the values that must be used for scattering.}]>:$updates,

    SymbolRefAttr:$update_computation,
    StrAttr:$dimension_numbers,
    BoolAttr:$indices_are_sorted
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
}

def TF_XlaSelectAndScatterOp : TF_Op<"XlaSelectAndScatter", [NoSideEffect]> {
  let summary = "Wraps the XLA SelectAndScatter operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor}]>:$operand,
    Arg<TF_I32OrI64Tensor, [{the shape of the window}]>:$window_dimensions,
    Arg<TF_I32OrI64Tensor, [{the inter-window strides}]>:$window_strides,
    Arg<TF_I32OrI64Tensor, [{the padding to apply at the start and end of each input dimensions}]>:$padding,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a tensor of values to scatter}]>:$source,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{a scalar representing the initial value for the output tensor}]>:$init_value,

    SymbolRefAttr:$select,
    SymbolRefAttr:$scatter
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_XlaSelfAdjointEigOp : TF_Op<"XlaSelfAdjointEig", [NoSideEffect]> {
  let summary = [{
Computes the eigen decomposition of a batch of self-adjoint matrices
  }];

  let description = [{
(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
i=0...N-1.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor.}]>:$a,

    BoolAttr:$lower,
    I64Attr:$max_iter,
    F32Attr:$epsilon
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The eigenvalues in ascending order, each repeated according to its
multiplicity.}]>:$w,
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{The column v[..., :, i] is the normalized eigenvector corresponding to the
eigenvalue w[..., i].}]>:$v
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaSendOp : TF_Op<"XlaSend", [TF_SendSideEffect]> {
  let summary = [{
Sends the named tensor to another XLA computation. Wraps the XLA Send operator
  }];

  let description = [{
documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#send .
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to send.}]>:$tensor,

    StrAttr:$tensor_name
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaSendTPUEmbeddingGradientsOp : TF_Op<"XlaSendTPUEmbeddingGradients", [AttrSizedOperandSegments, TF_MustExecute, TF_TPUEmbeddingReadEffect]> {
  let summary = "An op that performs gradient updates of embedding tables.";

  let description = [{
The gradients argument is a TensorList having the same length and shapes as the
return value of XlaRecvTPUEmbeddingActivations, but contains gradients of the
model's loss with respect to the embedding activations. The embedding tables are
updated from these gradients via the optimizer specified in the
TPUEmbeddingConfiguration proto given to tpu.initialize_system.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Float32Tensor>, [{A TensorList of gradients with which to update embedding tables.}]>:$gradients,
    Arg<Variadic<TF_Float32Tensor>, [{A TensorList of learning rates used for updating the embedding
tables via the optimizer. The length of the TensorList must be equal to the
number of dynamic learning rate tags specified in the
TPUEmbeddingConfiguration proto.}]>:$learning_rates,
    Arg<TF_VariantTensor, [{A Tensor with type=DT_VARIANT containing the deduplication
data. The tensor is an XLA nested tuple containing N elements (where N is
the ratio of the number of embedding to tensor cores per TPU chip). Each
element of the nested tuple is a tuple of rank 1 tensors. Each tensor either
contains indices (DT_UINT32) for embedding lookup on the TensorCore or
weights (DT_FLOAT) to apply to the output of the embedding lookup operation.}]>:$deduplication_data,

    StrAttr:$config
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr NumLearningRateTags = TF_DerivedOperandSizeAttr<1>;
  TF_DerivedOperandSizeAttr NumTables = TF_DerivedOperandSizeAttr<0>;
}

def TF_XlaSendToHostOp : TF_Op<"XlaSendToHost", [TF_SendSideEffect]> {
  let summary = "An op to send a tensor to the host.";

  let description = [{
input: the tensor that will be sent to the host.
Tinput: element type for input.
key: A unique identifier for this region used to match up host transfers.
  }];

  let arguments = (ins
    TF_Tensor:$input,

    StrAttr:$key
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tinput = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaSetDynamicDimensionSizeOp : TF_Op<"XlaSetDynamicDimensionSize", [InferTensorType, NoSideEffect, TF_NoConstantFold]> {
  let summary = "Make a static dimension into a xla bounded dynamic dimension.";

  let description = [{
The current static dimension size will become the bound and the second
        operand becomes the dynamic size of the dimension.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    TF_Int32Tensor:$dim_index,
    TF_Int32Tensor:$size
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];
}

def TF_XlaSortOp : TF_Op<"XlaSort", [NoSideEffect]> {
  let summary = "Wraps the XLA Sort operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{A `Tensor` of type T.}]>:$input
  );

  let results = (outs
    Res<TF_Tensor, [{A `Tensor` of type T.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaSvdOp : TF_Op<"XlaSvd", [NoSideEffect]> {
  let summary = [{
Computes the eigen decomposition of a batch of self-adjoint matrices
  }];

  let description = [{
(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{the input tensor.}]>:$a,

    I64Attr:$max_iter,
    F32Attr:$epsilon,
    StrAttr:$precision_config
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Singular values. The values are sorted in reverse order of magnitude, so
s[..., 0] is the largest value, s[..., 1] is the second largest, etc.}]>:$s,
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Left singular vectors.}]>:$u,
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{Right singular vectors.}]>:$v
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlaVariadicReduceOp : TF_Op<"XlaVariadicReduce", [NoSideEffect, SameVariadicOperandSize]> {
  let summary = "Wraps the variadic XLA Reduce operator.";

  let description = [{
Semantics are documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

This version is limited to operands of the same dtype.
XlaVariadicReduceV2 is a version that supports heterogeneous operands.
  }];

  let arguments = (ins
    Arg<Variadic<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>>, [{the input tensor(s)}]>:$input,
    Arg<Variadic<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>>, [{scalar initial value(s) for the reduction}]>:$init_value,

    I64ArrayAttr:$dimensions_to_reduce,
    SymbolRefAttr:$reducer
  );

  let results = (outs
    Variadic<TensorOf<[TF_Bfloat16, TF_Bool, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TF_XlaVariadicReduceV2Op : TF_Op<"XlaVariadicReduceV2", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = "Wraps the variadic XLA Reduce operator.";

  let description = [{
Semantics are documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

This is an expanded version of XlaVariadicReduce, with support for
operands of different dtypes, and improved shape inference.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{the input tensor(s)}]>:$inputs,
    Arg<Variadic<TF_Tensor>, [{scalar initial value(s) for the reduction}]>:$init_values,

    I64ArrayAttr:$dimensions_to_reduce,
    SymbolRefAttr:$reducer
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;

  let hasVerifier = 1;
}

def TF_XlaVariadicSortOp : TF_Op<"XlaVariadicSort", [NoSideEffect]> {
  let summary = "Wraps the XLA Sort operator, documented at";

  let description = [{
https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts one or more tensors, with support for custom comparator, dimension, and
is_stable attributes.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of `Tensor` of identical shape but possibly different types.}]>:$inputs,
    Arg<TF_Int32Tensor, [{The dimension along which to sort. Must be a compile-time constant.}]>:$dimension,

    SymbolRefAttr:$comparator,
    BoolAttr:$is_stable
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of `Tensor` of same shape and types as the `input`.}]>:$outputs
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;


  let hasVerifier = 1;
}

def TF_Xlog1pyOp : TF_Op<"Xlog1py", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
  let summary = "Returns 0 if x == 0, and x * log1p(y) otherwise, elementwise.";

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$x,
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$y
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_XlogyOp : TF_Op<"Xlogy", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if x == 0, and x * log(y) otherwise, elementwise.";

  let arguments = (ins
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$x,
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$y
  );

  let results = (outs
    TensorOf<[TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ZerosLikeOp : TF_Op<"ZerosLike", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns a tensor of zeros with the same shape and type as x.";

  let arguments = (ins
    Arg<TF_Tensor, [{a tensor of type T.}]>:$x
  );

  let results = (outs
    Res<TF_Tensor, [{a tensor of the same shape and type as x but filled with zeros.}]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ZetaOp : TF_Op<"Zeta", [NoSideEffect, ResultsBroadcastableShape]>,
                WithBroadcastableBinOpBuilder {
  let summary = [{
Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
  }];

  let description = [{
The Hurwitz zeta function is defined as:


\\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$x,
    TF_F32OrF64Tensor:$q
  );

  let results = (outs
    TF_F32OrF64Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__ArrayToListOp : TF_Op<"_ArrayToList", [NoSideEffect]> {
  let summary = "Converts an array of tensors to a list of tensors.";

  let arguments = (ins
    Variadic<TF_Tensor>:$input
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultTypeListAttr out_types = TF_DerivedResultTypeListAttr<0>;
}

def TF__CollateTPUEmbeddingMemoryOp : TF_Op<"_CollateTPUEmbeddingMemory", []> {
  let summary = [{
An op that merges the string-encoded HbmBufferConfig protos from all hosts.
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{String-encoded HbmBuffersConfig protos containing metadata about
the memory allocations reserved for TPUEmbedding across all hosts.}]>:$memory_configs
  );

  let results = (outs
    TF_StrTensor:$merged_memory_config
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF__ConfigureTPUEmbeddingHostOp : TF_Op<"_ConfigureTPUEmbeddingHost", []> {
  let summary = "An op that configures the TPUEmbedding software on a host.";

  let arguments = (ins
    Arg<TF_StrTensor, [{A string-encoded CommonConfiguration proto containing metadata
about the TPUEmbedding partitioner output.}]>:$common_config,
    Arg<TF_StrTensor, [{A string-encoded HbmBuffersConfig proto containing metadata about
the memory allocations reserved for TPUEmbedding.}]>:$memory_config,

    StrAttr:$config
  );

  let results = (outs
    Res<TF_StrTensor, [{A string containing metadata about the hostname and RPC port
used for communication with this host.}]>:$network_config
  );
}

def TF__ConfigureTPUEmbeddingMemoryOp : TF_Op<"_ConfigureTPUEmbeddingMemory", []> {
  let summary = "An op that configures the TPUEmbedding software on a host.";

  let arguments = (ins
    Arg<TF_StrTensor, [{A string-encoded CommonConfiguration proto containing metadata
about the TPUEmbedding partitioner output and the HBM size (in bytes) required
for operation.}]>:$common_config
  );

  let results = (outs
    Res<TF_StrTensor, [{A string-encoded HbmBuffersConfig proto containing metadata about
the memory allocations reserved for TPUEmbedding.}]>:$memory_config
  );
}

def TF__ConnectTPUEmbeddingHostsOp : TF_Op<"_ConnectTPUEmbeddingHosts", []> {
  let summary = [{
An op that sets up communication between TPUEmbedding host software instances
  }];

  let description = [{
after ConfigureTPUEmbeddingHost has been called on each host.
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{Strings containing metadata about the hostname and RPC port
used for communication with all hosts.}]>:$network_configs
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF__EagerConstOp : TF_Op<"_EagerConst", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__ExecuteTPUEmbeddingPartitionerOp : TF_Op<"_ExecuteTPUEmbeddingPartitioner", []> {
  let summary = [{
An op that executes the TPUEmbedding partitioner on the central configuration
  }];

  let description = [{
device and computes the HBM size (in bytes) required for TPUEmbedding operation.
  }];

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    Res<TF_StrTensor, [{A string-encoded tpu_embedding::CommonConfiguration proto
containing metadata about the TPUEmbedding partitioner output and
the HBM size (in bytes) required for operation.}]>:$common_config
  );
}

def TF__FinalizeTPUEmbeddingOp : TF_Op<"_FinalizeTPUEmbedding", []> {
  let summary = "An op that finalizes the TPUEmbedding configuration.";

  let arguments = (ins
    Arg<TF_StrTensor, [{A string-encoded CommonConfiguration proto containing metadata
about the TPUEmbedding partitioner output and the HBM size (in bytes) required
for operation.}]>:$common_config,
    Arg<TF_StrTensor, [{A string-encoded HbmBuffersConfig proto containing metadata about
the memory allocations reserved for TPUEmbedding.}]>:$memory_config
  );

  let results = (outs);
}

def TF__FusedBatchNormExOp : TF_Op<"_FusedBatchNormEx", [NoSideEffect]> {
  let summary = "Internal FusedBatchNorm operation: reserved for internal use.";

  let description = [{
Do not invoke this operator directly in Python. A fusion optimization is
expected to create these operators.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$x,
    TF_Float32Tensor:$scale,
    TF_Float32Tensor:$offset,
    TF_Float32Tensor:$mean,
    TF_Float32Tensor:$variance,
    Variadic<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>>:$side_input,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0f">:$exponential_avg_factor,
    DefaultValuedAttr<StrAttr, "\"Identity\"">:$activation_mode,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$y,
    TF_Float32Tensor:$batch_mean,
    TF_Float32Tensor:$batch_variance,
    TF_Float32Tensor:$reserve_space_1,
    TF_Float32Tensor:$reserve_space_2,
    TF_Float32Tensor:$reserve_space_3
  );

  TF_DerivedOperandSizeAttr num_side_inputs = TF_DerivedOperandSizeAttr<5>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;
}

def TF__FusedConv2DOp : TF_Op<"_FusedConv2D", [NoSideEffect]> {
  let summary = [{
Performs a convolution followed by a specified series of operations.
  }];

  let description = [{
The inputs to the convolution are `input` and `filter`. The series of operations
that follows is specified by the `fused_ops` attribute, which is a list of TF op
names specified as strings (e.g. "Relu"). They are performed in order, where the
(first) input to each op is the output of the preceding op. The first input and
the output of each fused_op must be of type T.

Currently supported fused_op combinations are: [X] and [X,A], where X is one of
{"BiasAdd","FusedBatchNorm"} and A is one of {"Elu","Relu","Relu6"}.

* The first input to op X is the Conv2D result, and the additional input(s) to X
are specified by `args`.
* If there is an op A specified, the output of op X is the input to op A, and op
A produces the _FusedConv2D output. Otherwise, op X produces the _FusedConv2D
output.

*NOTE*: Do not invoke this operator directly in Python. Grappler is expected to
create these operators.
  }];

  let arguments = (ins
    TF_F32OrF64Tensor:$input,
    TF_F32OrF64Tensor:$filter,
    Variadic<TF_F32OrF64Tensor>:$args,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<BoolAttr, "true">:$use_cudnn_on_gpu,
    DefaultValuedAttr<StrArrayAttr, "{}">:$fused_ops,
    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.2f">:$leakyrelu_alpha
  );

  let results = (outs
    TF_F32OrF64Tensor:$output
  );

  TF_DerivedOperandSizeAttr num_args = TF_DerivedOperandSizeAttr<2>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__FusedMatMulOp : TF_Op<"_FusedMatMul", [NoSideEffect, TF_SameOperandsAndResultElementTypeResolveRef]> {
  let summary = [{
Performs a MatMul followed by a specified series of operations.
  }];

  let description = [{
The inputs to the MatMul are specified by `a` and `b`. The series of operations
that follows is specified by the `fused_ops` attribute, which is a list of TF op
names specified as strings (e.g. "Relu"). They are performed in order, where the
(first) input to each op is the output of the preceding op. The first input and
the output of each fused_op must be of type T.

Currently supported fused_op combinations are: ["BiasAdd"] and ["BiasAdd",A],
where A is one of {"Elu","Relu","Relu6"}.

* The first input to BiasAdd is the MatMul result, and the additional BiasAdd
input is specified by `args`.
* If there is an op A specified, the output of the BiasAdd is the input to op A,
and op A produces the _FusedConv2D output. Otherwise, the BiasAdd produces the
_FusedConv2D output.

*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$a,
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$b,
    Variadic<TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>>:$args,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<StrArrayAttr, "{}">:$fused_ops,
    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.2f">:$leakyrelu_alpha
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32]>:$product
  );

  TF_DerivedOperandSizeAttr num_args = TF_DerivedOperandSizeAttr<2>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__HostRecvOp : TF_Op<"_HostRecv", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_RecvSideEffect]> {
  let summary = "Receives the named tensor from send_device on recv_device.";

  let description = [{
_HostRecv produces its output on host memory whereas _Recv produces its
output on device memory.
  }];

  let arguments = (ins
    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor to receive.}]>:$tensor
  );

  TF_DerivedResultTypeAttr tensor_type = TF_DerivedResultTypeAttr<0>;
}

def TF__HostSendOp : TF_Op<"_HostSend", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_SendSideEffect]> {
  let summary = "Sends the named tensor from send_device to recv_device.";

  let description = [{
_HostSend requires its input on host memory whereas _Send requires its
input on device memory.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to send.}]>:$tensor,

    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__ListToArrayOp : TF_Op<"_ListToArray", [NoSideEffect]> {
  let summary = "Converts a list of tensors to an array of tensors.";

  let arguments = (ins
    Variadic<TF_Tensor>:$input
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultSizeAttr N = TF_DerivedResultSizeAttr<0>;
  TF_DerivedResultTypeAttr T = TF_DerivedResultTypeAttr<0>;
}

def TF__RecvOp : TF_Op<"_Recv", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_RecvSideEffect]> {
  let summary = "Receives the named tensor from send_device on recv_device.";

  let arguments = (ins
    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs
    Res<TF_Tensor, [{The tensor to receive.}]>:$tensor
  );

  TF_DerivedResultTypeAttr tensor_type = TF_DerivedResultTypeAttr<0>;
}

def TF__SendOp : TF_Op<"_Send", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_SendSideEffect]> {
  let summary = "Sends the named tensor from send_device to recv_device.";

  let arguments = (ins
    Arg<TF_Tensor, [{The tensor to send.}]>:$tensor,

    StrAttr:$tensor_name,
    StrAttr:$send_device,
    I64Attr:$send_device_incarnation,
    StrAttr:$recv_device,
    DefaultValuedAttr<BoolAttr, "false">:$client_terminated
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__TPUCompileMlirOp : TF_Op<"_TPUCompileMlir", [TF_MustExecute]> {
  let summary = [{
Compiles a computations for execution on one or more TPU devices.
  }];

  let description = [{
For the internal use of the distributed TPU compiler.

'mlir_module' is a serialized MLIR module with a `main` function that contains
target computation.
'dynamic_shapes' contains dynamic shapes of arguments whose shapes were not
known statically at TPUReplication rewrite time.
'metadata' is a serialized TPUCompileMetadataProto describing the shapes and
types of the inputs to the computation, as well as a mapping onto the TPU pod
topology.
'program' output is a string key that is passed to the TPUExecute op and used to
look up the program in the compilation cache.
  }];

  let arguments = (ins
    Variadic<TF_Int64Tensor>:$dynamic_shapes,

    DefaultValuedAttr<StrAttr, "\"\"">:$mlir_module,
    StrAttr:$metadata
  );

  let results = (outs
    TF_StrTensor:$compilation_status,
    Variadic<TF_StrTensor>:$program
  );

  TF_DerivedOperandSizeAttr NumDynamicShapes = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedResultSizeAttr num_computations = TF_DerivedResultSizeAttr<1>;
}

def TF__TPUCompileMlirPlaceholderProgramKeyOp : TF_Op<"_TPUCompileMlirPlaceholderProgramKey", [TF_MustExecute]> {
  let summary = [{
Placeholder program key (compilation cache key) of a _TPUCompileMlir `program`.
  }];

  let description = [{
This op can be used when certain rewrite passes materialize ops that require a
program key but the _TPUCompileMlir op has not been added yet. Subsequent
rewrite passes must replace this op with a _TPUCompileMlir op `program` output.
  }];

  let arguments = (ins);

  let results = (outs
    TF_StrTensor:$program
  );
}

def TF__UnaryOpsCompositionOp : TF_Op<"_UnaryOpsComposition", [NoSideEffect, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = [{
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
  }];

  let description = [{
expected to create these operators.
  }];

  let arguments = (ins
    TensorOf<[TF_Float16, TF_Float32, TF_Float64]>:$x,

    StrArrayAttr:$op_names
  );

  let results = (outs
    TensorOf<[TF_Float16, TF_Float32, TF_Float64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF__XlaHostComputeMlirOp : TF_Op<"_XlaHostComputeMlir", [TF_RecvSideEffect, TF_SendSideEffect, TF_XlaHostComputeSideEffect]> {
  let summary = [{
A pseudo-op to represent host-side computation in an XLA program.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be sent to the host.}]>:$inputs,

    StrAttr:$send_key,
    StrAttr:$recv_key,
    DefaultValuedAttr<StrAttr, "\"\"">:$host_mlir_module
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of tensors that will be returned to the device.}]>:$outputs
  );

  TF_DerivedOperandTypeListAttr Tinputs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    func::FuncOp GetHostFunc(mlir::OwningOpRef<mlir::ModuleOp>* mlir_module);
  }];

  let hasVerifier = 1;
}

def TF__XlaRecvAtHostOp : TF_Op<"_XlaRecvAtHost", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_RecvSideEffect]> {
  let summary = [{
A placeholder op to receive values from a running XLA computation.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{The key sent at runtime by the compile node to identify which
execution the transfer corresponds to.}]>:$dynamic_key,

    StrAttr:$key,
    I64Attr:$device_ordinal
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of tensors that will be received from the XLA computation.}]>:$outputs
  );

  TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF__XlaRecvAtHostV2Op : TF_Op<"_XlaRecvAtHostV2", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_RecvSideEffect]> {
  let summary = [{
A placeholder op to receive values from a running XLA computation with support for a runtime device ordinal.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{The key sent at runtime by the compile node to identify which
execution the transfer corresponds to.}]>:$dynamic_key,
    Arg<TF_Int64Tensor, [{The device id relative to the associated host device.}]>:$device_ordinal,

    StrAttr:$key
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of tensors that will be received from the XLA computation.}]>:$outputs
  );

  TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF__XlaSendFromHostOp : TF_Op<"_XlaSendFromHost", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_SendSideEffect]> {
  let summary = "A placeholder op to send values to a running XLA computation.";

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be sent to the XLA computation.}]>:$inputs,
    Arg<TF_StrTensor, [{The key sent at runtime by the compile node to identify which
execution the transfer corresponds to.}]>:$dynamic_key,

    StrAttr:$key,
    I64Attr:$device_ordinal
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr Tinputs = TF_DerivedOperandTypeListAttr<0>;
}

def TF__XlaSendFromHostV2Op : TF_Op<"_XlaSendFromHostV2", [DeclareOpInterfaceMethods<TF_GetResourceInstanceInterface>, TF_SendSideEffect]> {
  let summary = [{
A placeholder op to send values to a running XLA computation with support for a runtime device ordinal.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be sent to the XLA computation.}]>:$inputs,
    Arg<TF_StrTensor, [{The key sent at runtime by the compile node to identify which
execution the transfer corresponds to.}]>:$dynamic_key,
    Arg<TF_Int64Tensor, [{The device id relative to the associated host device.}]>:$device_ordinal,

    StrAttr:$key
  );

  let results = (outs);

  TF_DerivedOperandTypeListAttr Tinputs = TF_DerivedOperandTypeListAttr<0>;
}
