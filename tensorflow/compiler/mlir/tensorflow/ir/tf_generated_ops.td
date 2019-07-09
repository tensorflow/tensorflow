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

#ifdef TF_OP_BASE
#else
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
#endif // TF_OP_BASE

def TF_AbsOp : TF_Op<"Abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the absolute value of a tensor.";

  let description = [{
Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
  }];

  let arguments = (ins
    TF_FpOrI32OrI64Tensor:$x
  );

  let results = (outs
    TF_FpOrI32OrI64Tensor:$y
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
  }];

  let arguments = (ins
    Variadic<TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Variant]>>:$inputs,

    Confined<I64Attr, [IntMinValue<1>]>:$N
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Variant]>:$sum
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_AddV2Op : TF_Op<"AddV2", [Broadcastable, Commutative, NoSideEffect]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
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
    TF_NumberTensor:$value,
    TF_NumberTensor:$bias,

    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
```python
>>> a = [1., 2., 3.]
>>> equality_bitcast = tf.bitcast(a,tf.complex128)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Cannot bitcast from float to complex128: shape [3] [Op:Bitcast]
>>> equality_cast = tf.cast(a,tf.complex128)
>>> print(equality_cast)
tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)
```
Example 2:
```python
>>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
<tf.Tensor: ... shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>
```
Example 3:
```python
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
[[ 0 0 0 0]
 [ 0 0 128 63]
 [ 0 0 128 63]], shape=(3, 4), dtype=uint8)
```

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

  let hasCanonicalizer = 1;
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

```python
>>> x = tf.constant([1, 2, 3])
>>> y = tf.broadcast_to(x, [3, 3])
>>> sess.run(y)
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]], dtype=int32)
```

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

def TF_ConcatOp : TF_Op<"Concat", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    I32Tensor:$concat_dim,
    Variadic<TF_Tensor>:$values,

    Confined<I64Attr, [IntMinValue<2>]>:$N
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_ConcatV2Op : TF_Op<"ConcatV2", [NoSideEffect]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$values,
    TF_I32OrI64Tensor:$axis,

    Confined<I64Attr, [IntMinValue<2>]>:$N
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_ConjOp : TF_Op<"Conj", [NoSideEffect]> {
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
    TF_FpTensor:$input,
    TF_FpTensor:$filter,

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

def TF_CosOp : TF_Op<"Cos", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes cos of x element-wise.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
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
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
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

def TF_EqualOp : TF_Op<"Equal", [Broadcastable, Commutative, NoSideEffect]>,
                 WithBroadcastableCmpOpBuilder {
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
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$x,
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$y
  );

  let results = (outs
    I1Tensor:$z
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
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
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
}

def TF_GreaterOp : TF_Op<"Greater", [Broadcastable, NoSideEffect]>,
                   WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x > y) element-wise.";

  let description = [{
*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
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

def TF_InvertOp : TF_Op<"Invert", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
Invert (flip) each bit of supported types; for example, type `uint8` value 01010101 becomes 10101010.
  }];

  let description = [{
Flip each bit of supported types.  For example, type `int8` (decimal 2) binary 00000010 becomes (decimal -3) binary 11111101.
This operation is performed on each element of the tensor argument `x`.
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

def TF_LessOp : TF_Op<"Less", [Broadcastable, NoSideEffect]>,
                WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x < y) element-wise.";

  let description = [{
*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
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

def TF_LogSoftmaxOp : TF_Op<"LogSoftmax", [NoSideEffect]> {
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
  let summary = "Returns the truth value of NOT x element-wise.";

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
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

def TF_MaxPoolOp : TF_Op<"MaxPool", [NoSideEffect]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$input,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_AnyStrAttrOf<["NHWC", "NCHW", "NCHW_VECT_C"]>, "NHWC">:$data_format
  );

  let results = (outs
    TF_IntOrFpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
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

def TF_MulOp : TF_Op<"Mul", [Broadcastable, Commutative, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x * y element-wise.";

  let description = [{
*NOTE*: `Multiply` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
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

def TF_NotEqualOp : TF_Op<"NotEqual", [Broadcastable, Commutative, NoSideEffect]>,
                    WithBroadcastableCmpOpBuilder {
  let summary = "Returns the truth value of (x != y) element-wise.";

  let description = [{
*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$x,
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$y
  );

  let results = (outs
    I1Tensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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

    Confined<I64Attr, [IntMinValue<1>]>:$N,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
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
other one to maximize the respresentable range.

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
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
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
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
    TensorOf<[BF16, F32, F64, I32, I64]>:$start,
    TensorOf<[BF16, F32, F64, I32, I64]>:$limit,
    TensorOf<[BF16, F32, F64, I32, I64]>:$delta
  );

  let results = (outs
    TensorOf<[BF16, F32, F64, I32, I64]>:$output
  );

  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<0>;

  let builders = [
    OpBuilder<"Builder* builder, OperationState* result, Value* start, "
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
    OpBuilder<"Builder* builder, OperationState* result, Value* input">
  ];
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
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
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
  }];

  let arguments = (ins
    TF_IntOrFpTensor:$features
  );

  let results = (outs
    TF_IntOrFpTensor:$activations
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

def TF_ReshapeOp : TF_Op<"Reshape", [NoSideEffect]> {
  let summary = "Reshapes a tensor.";

  let description = [{
Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of `shape` is the special value -1, the size of that dimension
is computed so that the total size remains constant.  In particular, a `shape`
of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

If `shape` is 1-D or higher, then the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

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
      "Builder* builder, OperationState* result, Value* tensor, Value* shape">
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
    TF_IntOrFpTensor:$images,
    I32Tensor:$size,

    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    F32Tensor:$resized_images
  );

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
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$tensor,
    TF_I32OrI64Tensor:$axis
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I1, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Str]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
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

def TF_SinOp : TF_Op<"Sin", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes sin of x element-wise.";

  let description = [{
  }];

  let arguments = (ins
    TF_FpOrComplexTensor:$x
  );

  let results = (outs
    TF_FpOrComplexTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
}

def TF_SoftmaxOp : TF_Op<"Softmax", [NoSideEffect]> {
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

def TF_SplitOp : TF_Op<"Split", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    I32Tensor:$split_dim,
    TF_Tensor:$value,

    Confined<I64Attr, [IntMinValue<1>]>:$num_split
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;
}

def TF_SplitVOp : TF_Op<"SplitV", [NoSideEffect]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$value,
    TF_I32OrI64Tensor:$size_splits,
    I32Tensor:$split_dim,

    Confined<I64Attr, [IntMinValue<1>]>:$num_split
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr Tlen = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
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
}

def TF_SubOp : TF_Op<"Sub", [Broadcastable, NoSideEffect]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x - y element-wise.";

  let description = [{
*NOTE*: `Subtract` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
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
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
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
      "Builder* builder, OperationState* result, Value* x, Value* perm">
  ];

  let verifier = [{
    return Verify(*this);
  }];
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
    TF_NumberTensor:$x,
    TF_NumberTensor:$y
  );

  let results = (outs
    TF_NumberTensor:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
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

    Confined<I64Attr, [IntMinValue<0>]>:$num,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
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
