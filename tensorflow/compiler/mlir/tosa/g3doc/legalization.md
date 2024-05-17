# TOSA Lowerings

## Introduction

### Overview

This document provides pseudo-code lowerings from TensorFlow and TensorFlow Lite
MLIR Dialects (https://www.tensorflow.org/mlir/dialects) to the TOSA Dialect
(https://mlir.llvm.org/docs/Dialects/TOSA/).

The documentation is a work-in-progress: sections with missing legalizations are
in the process of being written.

## Syntax

The pseudo-code syntax used in this document is described below.

### Primitive Datatypes

int8: signed 8-bit integer uint8: unsigned 8-bit integer int16: signed 16-bit
integer int32: signed 32-bit integer int64: signed 32-bit integer uint32:
unsigned 32-bit integer float32: IEEE-754 32-bit floating point format float64:
IEEE-754 64-bit floating point format bool: boolean

### Value

In pseudo-code, symbol starting with "%" indicates it’s a value. A value is
evaluated by an operator at run time, and operator can consume and can only
consume a list of values as operands. Note value’s tensor type is determined at
compile time. Only the evaluation happens at run time One can easily construct a
data flow subgraph by looking at the producer/consumer.

### Tensor Type

Tensor type is an attribute determined by legalization at compile time,
describing the shape and element data type. It’s noted as tensor&lt;shape,
dtype&gt;, or shorthanded as tensor&lt;%t.type&gt;

### Operator Prototype

In pseudocode an TOSA operator is prototyped as following format.

%&lt;output\_value&gt; = tosa.&lt;OPERATOR&gt;(%&lt;input\_value&gt;)
{&lt;attribute = …​}

### Value Attributes

For the purposes of brevity and clarity in this document, the pseudocode allows
the following notation on value attribute.

Shorthand         | Description
----------------- | ---------------------------------------------------
`%t.shape`        | Shape vector for the tensor
`%t.shape[i]`     | Size of dimension i for the tensor
`%t.rank`         | Rank of the tensor
`%t.dtype`        | Datatype of the tensor
`%t.scale`        | Quantized scaling parameter (float64)
`%t.zp`           | Quantized zero-point (int64)
`%t.signed`       | Boolean indicating the type is signed
`%t.num_bits`     | Number of bits in the datatype
`%t.num_elements` | Number of elements in the tensor
`%t.type`         | Tuple of `tensor<%t.shape, %t.dtype>`
`%t.size`         | For tensor lists: the number of tensors in the list

### Tensor Dimension Shorthand

Where the TOSA Specification allows the use of named dimensions, the following
names may be used.

Name | Description
---- | --------------------
`N`  | Batch dimension
`H`  | Height dimension
`W`  | Width dimension
`C`  | Channel dimension
`M`  | Depthwise multiplier

Each of these may be prefixed with `I` for the input dimension or `O` for the
output dimension or `K` for kernel dimensions.

## Common Legalization Functions

The following pseudocode helper functions are used to cannonicalize arguments
from different frameworks to the TOSA dialect.

### .as_constant(): Matched as Constant

Wherever %tensor.as_constant() is specified, a constant vector will be created
to hold the value in the %tensor at compile time. This only succeeds if %tensor
is fed by a constant type operator. If constant matching fails, the lowering
will fail and be terminated.

## Common Legalization Functions

The following pseudo-code helper functions are used to cannonicalize arguments
from different frameworks to the TOSA dialect.

### get_padding_values_from_explicit_pad_attr()

```
vector<int64> get_padding_values_from_explict_pad_attr(vector<int64> explicit_pad,
                                                         tensorflow::TensorFormat data_format_tf)
{
    int64 pad_before, pad_after
    vector<int64> computed_paddings

    for (int32 i = 0; i < 2; i++) {
        int64 dim = GetTensorSpatialDimIndex(4, data_format_tf, i)
        pad_before = explicit_pad[dim * 2]
        pad_after  = explicit_pad[dim * 2 + 1]
        computed_paddings.push_back(pad_before)
        computed_paddings.push_back(pad_after)
    }

    return computed_paddings
}
```

### get_padding_values_from_pad_type()

Calculate explicit padding array based on pad type

```
vector<int64> get_padding_values_from_pad_type(tensorflow::Padding padding, tensorflow::TensorFormat data_format,
                                        uint32 first_filter_spatial_dim, type input_type, type filter_type
                                        vector strides, vector dilations)
{
    assert(padding != tensorflow::Padding::EXPLICIT);

    vector<int64> computed_padding;

    // Padding over H and W dimensions
    for (int32 i = 0; i < 2; i++) {
        int32 ifm_dim = get_tensor_spatial_dim_index(4, data_format, i);

        int32 filter_dim = first_filter_spatial_dim + i;

        int32 dim_dilation = dilations[ifm_dim];
        int32 dim_stride   = strides[ifm_dim];

        int64 op_size, pad_before_tf, pad_after_tf;

        tensorflow::GetWindowedOutputSizeVerbose(input_type.shape[ifm_dim], filter_type.shape[filter_dim],
                                                   dim_dilation, dim_stride, padding,
                                                   // Outputs
                                                   &op_size, &pad_before_tf, &pad_after_tf);
        computed_paddings.push_back(pad_before_tf);
        computed_paddings.push_back(pad_after_tf);
    }

    return computed_paddings;
}
```

### positive_axis()

```
// Cannonicalize scalar axis attributes to a scalar positive axis attribute
int32 positive_axis(int32 axis, int32 rank)
{
   if (axis < 0)
       axis += rank;

   return axis;
}
```

### compute_scale_32()

```
void compute_scale_32(float64 scale, int32& multiplier, int32& shift)
{
    /* Generates mantissa and shift values where mantissa is in [-1.0,-0.5] or
    [0.5, 1.0] such that
    multiplier = mantissa*2^shift */

    const float64 mantissa = std::frexp(scale, &shift);
    auto shifted_m = std::round(mantissa * (int64(1) << 31));

    assert(shifted_m <= (int64(1) << 31)); // can't be greater that 1.0
    if (shifted_m == (int64(1) << 31)) {
        shifted_m /= 2;
        shift++;
    }
    // TOSA expect right shift to be positive, and embed (1 << 31) into right
    // shift bits
    shift = (-shift) + 31;

    assert(shifted_m <= std::numeric_limits<int32>::max());

    multiplier = static_cast<int32>(shifted_m);

}
```

### lower_batch_to_space_nd_op()

```
Value lower_batch_to_space_nd_op(Value %input, Value %block_shape, Value %crops, shape_t output_shape)
{

    vector <size_t> block_shape(%block_shape.rank)
    vector std::pair<size_t, size_t> crops_arr

    size_t remaining_shape_rank = %input.rank - %block.rank - 1
    size_t crops_dim = %crops.shape[0]

    for (int32 i = 0; i < crops_dim; i++) {
        crops[i] = std::make_pair(%crops.as_constant()[i * crops_dim + 0],
                                  %crops.as_constant()[i * crops_dim + 1])
    }

    // Step 1: Reshape input to
    // [block_shape[0],
    // ...
    // [block_shape[M-1],
    // [batch / prod(block_shape)]
    // [input_shape[1],
    // ...
    // [input_shape[N-1]

    vector <size_t> a1_shape(%block.rank + %input.rank)

    for (int32 i = 0; i < %block.rank; i++) {
        a1_shape[i] = %block.shape[i]
    }

    a1_shape[%block.rank] = %input.shape.[0] / %block.num_elements

    for (int32 i = 1; i < %input.rank; i++) {
        a1_shape[i + %block.rank] = %input.shape[i]
    }

    // Step 2. Permute to shape:
    // [ batch / prod(block_shape) ],
    // [ input_shape[1] ], [ block_shape[0] ]
    //  ...
    // [ input_shape[M] ], [ block_shape[M-1]
    // + remaining_input_shapes input_shape[M+1 .. N-1]
    vector <size_t> a2_perm(%block.rank + %input.rank)

    a2_perm[0] = %block.rank
    for (int32 i = 0; i < %block.rank; i++) {
        a2_perm[1 + i * 2 + 0] = %block.rank + 1 + i
        a2_perm[1 + i * 2 + 1] = i
    }

    // Step 3. Reshape to
    // [ batch / prod(block_shape) ],
    // [input_shape[1] * block_shape[0] ],
    //    ..
    // [input_shape[M * block_shape[M-1],
    // + remaining input shapes [input_shape[M+1.. N-1]]
    vector <size_t> a3_shape(%input.rank)

    %a3_shape[0] = %input.shape[0] / %block.num_elements
    for (int32 i = 0; i < %block.rank; i++) {
        a3_shape[i + 1] = %input.shape[i + 1] * %block.shape[i]
    }

    for (int32 i = 0; remaining_block_shape; i++) {
        a3_shape[1 + %block.rank + 1] = %input.shape[%block.rank + 1 + i]
    }

    // Step 4 Crop the start/end dimensions using slice
    vector <size_t> a4_begin(%input.rank), a4_size(%input.rank)

    for (int32 i = 0; i < %input.rank; i++) {
        if (i == 0 || i > crop_dims) {
           a4_begin[i] = 0
           a4_size[i] = output_shape[i]
        } else {
          a4_begin[i] = %crops[i-1].first
          a4_size[i] = crops[i - 1].first - crops[i - 1].second
        }
    }

    %a1_reshape = tosa.RESHAPE(%input) {new_shape=a1_shape}
    %a2_transpose = tosa.TRANSPOSE(%a1_reshape) {perms=a2_perm}
    %a3_reshape = tosa.RESHAPE(%a2_transpose) {new_shape=a3_shape}
    %output = tosa.SLICE(%a3_reshape) {begin=a4_begin, size=a4_size}

    return %output
}
```

### lower_concatv2_op()

```
Value lower_concatv2_op(Type output_type, Value %values, int32 axis)
{
    int32 tosa_axis = positive_axis(axis)

    assert(%values.size >= 2)

    // Convert scalar inputs to a tensor
    if (%values:0.size == 0) {
       for (int32 i = 0; i < %values.size; i++) {
          %values:i = tosa.RESHAPE(%values:i) {new_shape=1}
       }
    }

    for (int32 i=0; i < %values.size(); i++) {
        %val = %values:i
        if (%val.zp != output_type.zp || %val.scale != output_type.scale) {
            float64 rescale_scale = %val.scale / output_type.scale
            %values:i = tosa.RESCALE(%val) {scale=rescale_scale, input_zp=%values:0.zp, output_zp=output_type.zp}
        }
    }

    %concat_op = tosa.CONCAT(%values:0, %values:1) {axis=tosa_axis}

    for (int32 i = 2; i < %values.size; i++) {
        %concat_op = tosa.CONCAT(%concat_op, %values:i) {axis=tosa_axis}
    }

    return %concat_op
}
```

### lower_depth_to_space_op()

```
Value lower_depth_to_space_op(Value %input, size_t block_size[], Format_t data_format)
{
    assert(data_format == 'NHWC')

    vector <size_t> a2_shape = {%input.shape[0],
                                %input.shape[1],
                                %input.shape[2],
                                block_size[0],
                                block_size[1],
                                %input.shape[3] / (block_size[0] * block_size[1])}

    vector <size_t> a4_shape = {%input.shape[0],
                                %input.shape[1] * block_size[0],
                                %input.shape[2] * block_size[1],
                                %input.shape[3] / (block_size[0] * block_size[1])}

    %a2_reshape = tosa.RESHAPE(%input) {new_shape=a2_shape}
    %a3_transpose = tosa.TRANSPOSE(%a2_reshape) {perms={0, 1, 3, 2, 4, 5}}
    %output = tosa.RESHAPE(%a3_transpose) {new_shape=a4_shape}

    return %output
}
```

### lower_elu_op()

```
Value lower_elu_op(Value %value)
{
    // elu(x) = x < 0 ? (exp(x) - 1) : x
    // Create constants for 0/1 and reshape to match the rank
    // of %value
    %one_const = tosa.CONST() {value={1}}
    %zero_const = tosa.CONST() {value={0}}

    vector bcast_shape
    for (int32 i = 0; i < %value.rank; i++) {
        bcast_shape.push_back(1)
    }

    %one_reshape = tosa.RESHAPE(%one_const) {new_shape=bcast_shape}
    %zero_reshape = tosa.RESHAPE(%zero_const) {new_shape=bcast_shape}

    %exp_in = tosa.EXP(%value)
    %sub = tosa.SUB(%exp_in, %one_reshape)
    %ge  = tosa.GREATER_EQUAL(%value, %zero_reshape)
    %output = tosa.SELECT(%ge, %value, %sub)
    return %output
}
```

### lower_expand_dims()

```
Value lower_expand_dims(Value %input, int32 axis)
{
    vector<size_t> reshape_dims

    if (axis < 0 || axis >= %input.rank) {
        // Insert at the end of the tensor
        axis += %input.rank
        for (int32 i = 0; i < input.rank; i++) {
           reshape_dims.push_back(%input.shape[i])
        }
    } else {
        for (int32 i= 0 ; i < %input.rank; i++) {
            if (i == axis) {
                reshape_dims.push_back(1)
            }
            reshape_dims.push_back(%input.shape[i])
        }
    }

    %output = tosa.RESHAPE(%input) {new_shape=reshape_dims}
    return %output
}
```

### lower_fake_quant_op()

```
Value lower_fake_quant_op(Value %inputs, type output_type, float64 min, float64 max,
                            int64 num_bits, bool narrow_range)
{
    assert(num_bits == 8 || num_bits == 16)

    int64 qmax = (1L << (num_bits - 1)) - 1;
    int64 qmin = -(1L << (num_bits - 1))

    if (narrow_range) {
       qmin = qmin + 1
    }

    float64 scale = (max - min) / float64(qmax - qmin)

    int64 zeropoint = (int64)std::round((-min) / scale + float64(qmin))

    %quantized = lower_quantize_op(%inputs.type, %inputs, 1.0 / scale, zeropoint)

    %dequantized = lower_dequantize_op(output_type, %quantized_op, scale, zeropoint)

    return %dequantized
}
```

### lower_floor_div()

```
Value lower_floor_div(Value %lhs, Value %rhs)
{
    %recip = tosa.RECIPROCAL(%rhs)
    %mul = tosa.MUL(%lhs, %recip)
    %output = tosa.FLOOR(%mul)

    return %output
}
```

### lower_floor_mod()

```
Value lower_floor_mod(Value %lhs, Value %rhs)
{
    %recip = tosa.RECIPROCAL(%rhs)
    %mul = tosa.MUL(%lhs, %recip)
    %floor = tosa.FLOOR(%mul)
    %output = tosa.SUB(%mul, %floor)
    return %output
}
```

### lower_quantize_op()

```
Value lower_quantize_op(Type output_type, Value %input, float64 scale, int64 zeropoint)
{
    %const_scale = tosa.CONST() {value={scale}}
    %const_zp = tosa.CONST() {value={zeropoint}}
    %op1_mul_in_scale = tosa.MUL(%input, %const_scale)
    %op2_add_op1_zp = tosa.ADD(%op1_mul_in_scale, %const_zp)
    %op3_cast_op2 = tosa.CAST(%op2_add_op1_zp) // f32->%output.dtype
}
```

### lower_dequantize_op()

```
Value lower_dequantize_op(Value %input, float64 scale, int64 zeropoint)
{
    %const_scale = tosa.CONST() {value={scale}}
    %const_zp = tosa.CONST() {value={(float64)zeropoint}}
    %op1_cast_in = tosa.CAST(%input) // %input.dtype->f32
    %op2_sub_op1_zp = tosa.SUB(%op1_cast_in, %const_zp)
    %op3_mul_op2_scale = tosa.MUL(%op2_sub_op1_zp, %const_scale)
}
```

### lower_log_softmax_op()

```
Value lower_log_softmax_op(Value %logits)
{
    %op1 = tosa.EXP(%logits)
    %op2 = tosa.REDUCE_SUM(%op1) {axis=(%logits.rank-1)}
    %op3 = tosa.RECIPROCAL(%op2)
    %op4 = tosa.MUL(%op1, %op3)
    %op5 = tosa.LOG(%op4)

    return %op5
}
```

### lower_pack_op()

```
Value lower_pack_op(Value %input[], size_t axis)
{
    size_t concat_axis = positive_axis(axis)

    size_t input_tensor_rank = %input[0].rank

    // Convert any rank 0 to rank 1 with reshape
    if (input_tensor_rank == 0) {
       for (int32 i = 0; i < %input.size; i++) {
           %input[i] = tosa.RESHAPE(%input[i], {1})
       }
   }

   vector<size_t> output_shape
   for (int32 i = 0; i < input_tensor_rank; i++) {
       output_shape.push_back(%input[0].shape[i]
   }

   output_shape[concat_axis] = output_shape[concat_axis] * %input.size

   // First pair of tensors
   %concat = tosa.CONCAT(%input[0], %input[1]) {axis=concat_axis}

   // Remaining tensors
   for (int32 i = 2; i < %input.size; i++) {
      %concat = tosa.CONCAT(%concat, %input[i]) {axis=concat_axis}
   }

   if (input_tensor_rank == 0) {
      // No reshape needed for rank 0, already done
      %output = %concat
   } else

      %reshape = tosa.RESHAPE(%concat) {new_shape=output_shape}

      if (concat_axis == input_tensor_rank) {
         // Output shape is [A, B, C, .. n] in this case,
         // need to reshape to [N, A, B, C, ..] with perm [1, 2, 3, .. 0]
         concat_axis = 0

         vector <size_t> perms
         for (int32 i = 0; i < %input[0].rank; i++)
            perms.push_back(i + 1)
         perms.push_back(0)

         %output = tosa.TRANSPOSE(%reshape) {perms=perms}
     } else {
         %output = %reshape
     }

     return %output
}
```

### lower_reduce_op()

```
Value lower_reduce_op<tosa_op_t OP>(Value %input, shape_t output_shape, Value %axes, bool keep_dims, float64 input_scale=1.0f, int32 input_zp=0, float64 output_scale=1.0f, int32 output_zp=0)
{

    vector axes_vec = %axes.as_constant();

    // Special case of no axes means no transformation
    if (axes_vec.size() == 0) {
       return tosa.IDENTITY(%input)
    }

    bool is_quantized = isa<QuantizedType>(%input.dtype) ? true : false

    shape_t shape = %input.shape;
    %output = %input;

    if (is_quantized) {
        %output = tosa.RESCALE(%output) {scale=input_scale, input_zp=input_zp, output_zp=0}
    }

    for (int32 i = 0; i < axes_vec.size(); i++) {
        int32 axis = positive_axis(axes_vec[i], %input.rank);

        shape[axis] = 1;
        %output = tosa.OP(%output) {axis=axis}
    }

    if (!keep_dims) {
       %output = tosa.RESHAPE(%output) {new_shape=output_shape}
    }

    if (is_quantized) {
        %output = tosa.RESCALE(%output) {scale=output_scale, input_zp=0, output_zp=output_zp}
    }

    return %output;
}
```

### lower_resize_op()

```
Value lower_resize_op(Value %images, Value %size, shape output_shape, dtype output_dtype, mode_t mode)
{
    int32 input_height  = %input.shape[1]
    int32 input_width   = %input.shape[2]
    int32 output_height = %output.shape[1]
    int32 output_width  = %output.shape[2]

    float64 in_center_h  = static_cast<float64>(input_height - 1) / 2.0
    float64 in_center_w  = static_cast<float64>(input_width - 1) / 2.0
    float64 out_center_h = static_cast<float64>(output_height - 1) / 2.0
    float64 out_center_w = static_cast<float64>(output_width - 1) / 2.0

    float64 fp_stride_y, fp_stride_x
    if (align_corner && output_height > 1)
        fp_stride_y = static_cast<float64>(input_height - 1) / static_cast<float64>(output_height - 1)
    else
        fp_stride_y = static_cast<float64>(input_height) / static_cast<float64>(output_height)
    if (align_corner && output_width > 1)
        fp_stride_x = static_cast<float64>(input_width - 1) / static_cast<float64>(output_width - 1)
    else
        fp_stride_x = static_cast<float64>(input_width) / static_cast<float64>(output_width)

    float64 fp_offset_y = fp_offset_y = 0.0f
    if (half_pixel_centers) {
        fp_offset_y = fp_stride_y * 0.5f - 0.5f
        fp_offset_x = fp_stride_x * 0.5f - 0.5f
    }

    if (dtype == float)
        %op1_resize_in = tosa.RESIZE(%input) {stride={fp_stride_y, fp_stride_x}, offset={fp_offset_y, fp_offset_x}, shift=0, resize_mode=mode}
    else {
        int32 shift = 10
        float64 unit = static_cast<float64>(1 << shift)
        int32 stride_y = fp_stride_y * unit
        int32 stride_x = fp_stride_x * unit
        int32 offset_y = fp_offset_y * unit
        int32 offset_x = fp_offset_x * unit

        %op1_resize_in = tosa.RESIZE(%input) {stride={stride_y, stride_x}, offset={offset_y, offset_x}, shift=shift, resize_mode=mode}

        if (mode == "BILINEAR") {
            %const_zero = tosa.CONST() {value={0}}
            %const_twenty = tosa.CONST() {value={20}}
            %op2_ge_op1 = tosa.GREATER_EQUAL(%op1_resize_in, %const_zero)
            %op3_abs_op1 = tosa.ABS(%op1_resize_in)
            %op4_rshift_op3 = tosa.ARITHMETIC_RIGHT_SHIFT(%op3_abs_op1, %const_twenty)
            %op5_negate_op4 = tosa.NEGATE(%op4_rshift_op3)
            %op6_select_op2_op4_op5 = tosa.SELECT(%op2_ge_op1, %op4_rshift_op3, %op5_negate_op4)
            %op7_cast_op6 = tosa.CAST(%op6_select_op2_op4_op5) // i32/i48->%output.dtype
        }
    }
}
```

### lower_reversev2_op()

```
Value lower_reverse_v2_op(Value %tensor, Value %axis)
{
    Value %output = %tensor

    if (%axis.num_elements == 0) {
       %output = tosa.IDENTITY(%tensor)
    } else {
        for (int32 i = 0; i < %axis.shape[0]; i++) {
            size_t axis_val = positive_axis(%axis.as_constant()[i])
            %output = tosa.REVERSE(%output) {axis=%axis_val}
        }
    }

    return %output
}
```

### lower_round_op()

```
Value lower_round_op(Value %x)
{
    %half = tosa.CONST() {value={0.5}}
    %add = tosa.ADD(%x, %half)
    %output = tosa.FLOOR(%add)

    return %output
}
```

### lower_selectv2_op()

```
Value lower_selectv2_op(Value %condition, Value %t, Value %e, shape output_shape)
{
    // Reshape condition so that ranks match to support
    // broadcasting (if necessary)

    if (%condition.rank != output_shape.size) {
       vector <size_t> cond_shape = %condition.shape
       for (int32 i = 0; i < (output_shape.size - %condition.rank); i++) {
           cond_shape.push_front(1)
       }

       %condition = tosa.RESHAPE(%condition) {new_shape=cond_shape}
    }

    %output = tosa.SELECT(%condition, %t, %e)

    return %output
}
```

### lower_shape_op()

```
Value lower_shape_op(Value %input)
{
    vector <size_t> input_shape = %input.shape

    %shape = tosa.CONST() {value={input_shape}}
    return %shape
}
```

### lower_space_to_batch_nd_op()

```
Value lower_space_to_batch_nd_op(Value %input, Value %block_shape, Value %padding)
{

    size_t block_rank = %block.shape[0]
    size_t remaining_shape_rank = %input.rank - block_rank - 1;

    // Step 1. Pad based on paddings operand (flattened representation of [input.rank][2]-shaped array)
    vector <size_t> a1_padding
    a1_padding[0] = 0
    a1_padding[1] = 0

    for (int32 i = 0; i < %padding.shape[0]; i++) {
        a1_padding[i + 2] = %padding.as_constant()[i]
    }

    %a1_pad = tosa.PAD(%input) {padding=a1_padding}

    // Step 2. Reshape to
    // [batch + padded_shape[1] / block_shape[0], block_shape[0], ...
    //    padded_shape[M] / block_shape[M-1], block_shape[M-1]] +
    //    remaining_shape

    vector <size_t> a2_shape(1 + block_rank * 2 + remaining_shape_rank)
    a2_shape[0] = %input.shape[0]
    for (int32 i = 0; i < block_rank; i++) {
        a2_shape[1 + i * 2 + 0] = %a1_pad.shape[1 + i] / block_shape.as_constant()[i]
        a2_shape[1 + i * 2 + 1] = block_shape.as_constant()[i]
    }

    for (int32 i = 0; i < remaining_shape_rank; i++) {
        a2_shape[1 + block_rank * 2 + i] = %input.shape[1 + block_rank + i]
    }

    %a2_reshape = tosa.RESHAPE(%a1_pad) {new_shape=a2_shape}

    // Step 3 transpose to
    //  block-shape +
    //  [batch] +
    //  [padded_shape[1] / block_shape[0],
    // ...
    //  [padded_shape[M] / block_shape[M-1]] +
    //  remaining_shape
    vector <size_t> a3_perm(%a2_reshape.rank)
    size_t block_num_elems = 1

    for (int32 i = 0; i < block_rank; i++) {
        a3_perm[i] = 1 + 2 * i + 1
        a3_perm[block_rank + 1 + i] = 2 * i + 1
        block_num_elems *= %block.as_constant()[i]
    }

    a3_perm[block_rank] = 0
    for (int32 i = (1 + block_rank * 2); i < %a2_reshape.rank; i++) {
        a3_perm[i] = i
    }

    %a3_reshape = tosa.RESHAPE(%a2_reshape) {perm=a3_perm}

    // Step 4. Reshape transposed tensor to
    // [ batch * prod(block_shape)] +
    // [ padded_shape[1] / block_shape[0],
    //   ...,
    // padded_shape[M] / block_shape[M-1]] +
    // remaining_shape

    vector <size_t> a4_shape(%input.rank)
    a4_shape[0] = batch_size * block_num_elements

    for (int32 i = 0; i < block_rank; i++) {
        a4_shape[i + 1] = %a1_pad.shape[i + 1] / %block.as_constant()[i]
    }

    for (int32 i = 0; i < remaining_block_shape; i++) {
        a4_shape[1 + block_rank + i] = %input.shape[1 + block_rank + i]
    }

    %output = tosa.RESHAPE(%a3_reshape) {new_shape=a4_shape}

    return %output
}
```

### lower_space_to_depth_op()

```
Value lower_space_to_depth_op(Value %input, size_t block_size[], Format_t data_format)
{
    assert(data_format == 'NHWC')

    vector <size_t> a2_shape = {%input.shape[0],
                                %input.shape[1] / block_size[0],
                                %block_size[0],
                                %input_shape[2] / block_size[1],
                                %block_size[1],
                                %input_shape[3]}
    %a2_reshape = tosa.RESHAPE(%input) {new_shape=a2_shape}
    %a3_transpose = tosa.TRANSPOSE(%a2_reshape) {perm={0, 1, 3, 2, 4, 5}}

    vector <size_t> a4_shape = {%input.shape[0],
                                %input_shape[1] / block_size[0],
                                %input_shape[2] / block_size[1],
                                %input_shape[3] * block_size[0] * block_size[1]}
    %output = tosa.RESHAPE(%a3_transpose) {new_shape=%a4_shape}
    return %output
}
```

### lower_split_op()

```
Value lower_split_op(Value %value, size_t axis, size_t num_split)
{
    Value %output[]

    size_t slice_size = %value.shape[axis] / num_split

    for (int32 i = 0; i < num_split; i++) {
        vector <size_t> begin_vals, size_vals

        for (int32 j = 0; j < %value.rank; j++) {
            if (j == axis) {
               begin_vals.push_back(slice_size * i)
               size_vals.push_back(slice_size)
            } else {
               begin_vals.push_back(0)
               size_vals.push_bac(%value.shape[j])
            }

            %output[i] = tosa.SLICE(%value) {start=begin_vals, size=size_vals}
        }

    }

    %output_list = tosa.IDENTITYN(%output)
    return %output_list
}
```

### lower_splitv_op()

```
Value lower_splitv_op(Value %value, vector <size_t> size_split, size_t axis)
{
   Value %output[]

   size_t curr_split_start = 0

   for (int32 i = 0; i < size_split.size(); i++) {
       vector <size_t> begin_vals, size_vals

       for (int32 j = 0; j < %value.rank; j++) {
           if (j == axis) {
              begin_vals.push_back(curr_split_start)
              size_vals.push_back(size_split[i])
           } else {
              begin_vals.push_back(0)
              size_vals.push_back(input.shape[j])
           }
       }

       %output[i] = tosa.SLICE(%value) {start=begin_vals, size=size_vals}

       curr_split_start += size_split[i]
   }

    %output_list = tosa.IDENTITYN(%output)
    return %output_list
}
```

### lower_squeeze_op()

```
Value lower_squeeze_op(Value %input, vector<size_t> squeeze_dims)
{
    vector <size_t> reshape_dims

    if (squeeze_dims.size() == 0) {
       // Remove all 1-dims
       for (int32 i = 0; i < %input.rank; i++) {
           if (%input.shape[i] != 1) {
              reshape_dims.push_back(%input_shape[i])
           }
       }
    } else {
      // Remove the specified dimensions
      for (int32 i = 0; i < %input.rank; i++) {
          if (!squeeze_dims.find(i) || %input.shape[i] != -1) {
              reshape_dims.push_back(%input_shape[i])
          }
      }
    }

    %output = tosa.RESHAPE(%input) {new_shape=reshape_dims}

    return %output
}
```

### lower_strided_slice_op()

```
Value lower_strided_slice_op(Value %input, Value %begin_val, Value %end_val, Value %strides_val,
                               size_t begin_mask, size_t end_mask, size_t ellipsis_mask,
                               size_t new_axis_mask, size_t shrink_axis_mask)
{
    // Note: does not implement ellipsis_mask or reverse stride at this time
    assert(ellipsis_mask == 0)

    vector <size_t> begin(%begin_val.as_constant()), end(%end_val.as_constant()), strides(%strides_val.as_constant())
    vector <size_t> a1_start, a1_size, a2_shape, a3_start, a3_size, a4_shape

    for (int32 i = 0; i < %input.rank; i++) {
        if (begin_mask & (1 << i)) {
           begin[i] = 0
        }

        if (end_mask & (1 << i)) {
           end[i] = %input.shape[i]
        }

        // Wrap around index if begin and end are negative
        if (begin[i] < 0) {
           begin[i] += %input.shape[i]
        }

        if (end[i] < 0) {
           end[i] += %input.shape[i]
        }

        a1_start[i] = begin[i]
        a1_size[i] = end[i] - begin[i]

        a2_shape[i*2 + 0] = a1_size[i] / strides[i]
        a2_shape[i*2 + 1] = strides[i]

        a3_start[i*2 + 0] = 0
        a3_start[i*2 + 1] = 0

        if (shrink_axis_mask & (1 << i)) {
           a3_size[i*2 + 0] = 1
        } else {
           a3_size[i*2 + 0] = a1_size[i] / strides[i]
        }
        a3_size[i*2 + 1] = 1

        if (!(shrink_axis_mask & (1 << i))) {
           if (new_axis_mask & (1 << i)) {
              a4_shape.push_back(1)
           a4_shape.push_back((a1_size[i] / strides[i]))
        }
    }

    // Step 1: Slice the input array
    %a1_slice = tosa.SLICE(%input) {start=a1_start, size=a1_size}

    // Step 2: Reshape the sliced array: 2x as many dimensions as %input
    %a2_reshape = tosa.RESHAPE(%a1_slice) {new_shape=a2_shape}

    // Step 3: Take a slice of the [0] index along each of the strided dimensions (even dimensions)
    %a3_slice = tosa.SLICE(%a2_reshape) {start=a3_start, size=a3_size}

    // Step 4: Reshape the now-strided tensor back down to the desired number of dimensions
    %output = tosa.RESHAPE(%a3_slice) {new_shape=a4_shape}

    return %output
}
```

### lower_unpack_op()

```
Value lower_unpack_op(Value %value, size_t axis, uint64_t num)
{
    axis = positive_axis(axis)

    Value %output_arr[]

    // Step 1: transpose 'axis' to left-most dimension, if necessary
    Value %transposed_value

    if (axis != 0) {
       vector <size_t> perms

       perms.push_back(axis)
       for (int32 i = 0; i < %input.rank; i++) {
           if (i != axis)
              perms.push_back(i)
       }

       %transposed_value = tosa.TRANSPOSE(%value) {perms=perms}

   } else {
      %transposed_value = %value
   }

   // Step 2: Slice [N, A, B, C] into [N] [A, B, C]
   for (int32 i = 0; i < %transposed_value.rank; i++) {
       vector <size_t> begin_vals, size_vals, shape_vals

       begin_vals.push_back(i)
       size_vals.push_back(1)

       for (int32 j = 1; j < %transposed_value.rank; j++) {
           begin_vals.push_back(0)
           size_vals.push_back(transposed_value.shape[j])
           shape_vals.push_back(transposed_value.shape[j])
       }

       %slice = %tosa.SLICE(%transposed_value) {begin=begin_vals, size=size_vals}
       %output_arr[i] = %tosa.RESHAPE(%slice) {new_shape=shape_vals} {begin=begin_vals, size=size_vals}
   }

   // Combine array of sliced tensors into a list of tensors
   %output = tosa.IDENTITYN(%output_arr)
   return %output
}
```

### get_transpose_conv2d_padding_values_from_pad_type()

```
vector<int64> get_transpose_conv2d_padding_values_from_pad_type(tensorflow::Padding padding, tensorflow::TensorFormat data_format,
                                                         uint32 first_filter_spatial_dim, type input_type, type filter_type
                                                         vector strides, vector dilations)
{
    int64 pad_before, pad_after;
    vector<int64> computed_padding

    for (int32 i = 0; i < 2; i++) {
        int64 ifm_dim = GetTensorSpatialDimIndex(4, data_format, i);
        int64 ofm_dim = GetTensorSpatialDimIndex(4, data_format, i);
        int64 filter_dim = first_filter_spatial_dim + 1

        int64 ifm_size = input_shape[ifm_dim]
        int64 ofm_size = output_dims[ofm_dim]
        int64 filter_size = filter.shape[filter_dim]
        int64 dim_dilation = dilations[i]
        int64 dim_stride = strides[i]
        int32 effective_filter_size = (filter_size - 1) * dim_dilation + 1
        int32 total_padding = ((ifm_size - 1) * dim_stride + effective_filter_size - ofm_size)
        total_padding = total_padding > 0 ? total_padding : 0

        pad_before = total_padding / 2
        pad_after = total_padding - pad_before

        computed_padding.push_back(pad_before)
    }

    return computed_padding
}
```

### lower_fused_activation()

```
Value lower_fused_activation(Value %input, string activation)
{
    bool is_quantized = isa<QuantizedType>(%input.dtype) ? true : false

    if (is_quantized) {
        if (activation == "NONE") {
            return %input
        }
        else if (activation == "RELU") {
            int32 quantized_0 = %input.zp
            int32 quantized_max = %input.storage_max
            return tosa.CLAMP(%input) {min_int=quantized_0, max_int=quantized_max}
        }
        else if (activation == "RELU6") {
            int32 quantized_0 = %input.zp
            int32 quantized_6 = %input.zp + (6.0 / %input.scale)
            return tosa.CLAMP(%input) {min_int=quantized_0, max_int=quantized_6}
        }
        else if (activation == "RELU_N1_TO_1") {
            int32 quantized_n1 = %input.zp + (-1.0 / %input.scale)
            int32 quantized_1 = %input.zp + (1.0 / %input.scale)
            return tosa.CLAMP(%input) {min_int=quantized_n1, max_int=quantized_1}
        }
    }
    else {
        if (activation == "NONE") {
            return %input
        }
        else if (activation == "RELU") {
            return tosa.RELUN(%input) {max_fp=numeric_limit<float32>::max()}
        }
        else if (activation == "RELU6") {
            return tosa.RELUN(%input) {max_fp=6.0}
        }
        else if (activation == "RELU_N1_TO_1") {
            return tosa.CLAMP(%input) {min_fp=-1.0, max_fp=1.0}
        }
        else if (activation == "TANH") {
            return tosa.TANH(%input)
        }
    }
}
```

### get_table_const_tensor()

```
Value get_table_const_tensor(function func)
{
    array<int16, 513> table_array
    for (int32 i = -256; i <= 256; i++) {
        table_array[i] = func(i)
    }

    return tosa.CONST() {value=table_array}
}
```

### lower_gather_op()

```
Value lower_gather_op(Value %params, Value %indices, int32 batch_dims, int32 axis)
{
    assert batch_dims <= %indices.rank
    assert axis >= batch_dims

    int32 N = W = K = C = 1

    for (int32 i = 0; i < batch_dims; i++) N *= %params.shape[i]
    for (int32 i = batch_dims; i < %indices.rank; i++) W *= %indices.shape[i]
    K = %params.shape[axis]
    for (int32 i = batch_dims; i < axis; i++) C *= %params.shape[i]
    for (int32 i = (axis + 1); i < %params.rank; i++) C *= %params.shape[i]

    vector<int32> params_idx_batch, params_idx_left, params_idx_indices, params_idx_right
    for (int32 i = 0; i < %params.rank; i++) {
        if (i < batch_dims && i < axis)
            params_idx_batch.push_back(i)
        else if (i < axis)
            params_idx_left.push_back(i)
        else if (i < (axis + 1))
            params_idx_indices.push_back(i)
        else
            params_idx_right.push_back(i)
    }

    vector<int32> params_perm = {params_idx_batch, params_idx_left, params_idx_indices, params_idx_right}
    vector<int32> result_perm
    for (int32 i = 0; i < batch_dims; i++)
        result_perm.push_back(i)
    for (int32 i = 0; i < params_idx_left.size(); i++)
        result_perm.push_back(params_idx_left[i])
    for (int32 i = batch_dims; i < %indices.rank; i++)
        result_perm.push_back(i)
    for (int32 i = 0; i < params_idx_right.size(); i++)
        result_perm.push_back(params_idx_right[i])

    %const_params_perm = tosa.CONST() {value=params_perm}
    %const_result_perm = tosa.CONST() {value=result_perm}

    %op1_transpose_params = tosa.TRANSPOSE(%params, %const_params_perm)
    %op2_reshape_op1 = tosa.RESHAPE(%op1_transpose_params) {shape={N,K,C}}
    %op3_reshape_indices = tosa.RESHAPE(%indices) {shape={N,W}}
    %op4_gather_op2_op3 = tosa.GATHER(%op2_reshape_op1, %op3_reshape_indices)
    %op5_reshape_op4 = tosa.RESHAPE(%op4_gather_op2_op3) {shape={N,W,C}}
    %op6_transpose_op5 = tosa.TRANSPOSE(%op5_reshape_op4, %const_result_perm)
}
```

### lower_gather_nd_op()

```
Value lower_gather_nd_op(Value %params, Value %indices)
{
    int32 N = W = K = C = ND = 1

    ND = %indices.shape[%indices.rank - 1]

    assert ND < %params.rank

    for (int32 i = 0; i < (%indices.rank - 1); i++) W *= %indices.shape[i]
    for (int32 i = 0; i < ND; i++) K = %params.shape[i]
    for (int32 i = ND; i < %params.rank; i++) C *= %params.shape[i]

    vector<int32> flatten_coeff_vec
    for (int32 i = 0; i < ND; i++) flatten_coeff_vec.push_back(i)
    flatten_coeff_vec.push_back(1)

    %const_flatten_coeff = tosa.CONST() {value=flatten_coeff_vec}
    %op1_reshape_params = tosa.RESHAPE(%params) {shape={N,K,C}}
    %op2_reshape_indices = tosa.RESHAPE(%indices) {shape={W,ND}}
    %op3_mul_op2_flatten_coeff = tosa.MUL(%op2_reshape_indices, %const_flatten_coeff)
    %op4_rsum_op3 = tosa.REDUCE_SUM(%op3_mul_op2_flatten_coeff) {axis=1}
    %op5_reshape_op4 = tosa.RESHAPE(%op4_rsum_op3) {shape={N,W}}
    %op6_gather_op1_op5 = tosa.GATHER(%op1_reshape_params, %op5_reshape_op4)
    %op7_reshape_op6 = tosa.RESHAPE(%op6_gather_op1_op5) {shape={N,W,C}}
}
```

### lower_one_hot_op()

```
Value lower_one_hot_op(Value %indices, Value %depth, Value %on_value, Value %off_value, int32 axis)
{
    int32 N = W = C = 1
    int32 K = %depth.as_constant()
    int32 left_dim = right_dim = 1
    for(int32 i : %indices.rank) {
        int32 dim = %indices.shape[i]
        N *= dim
        if (i >= axis)
            right_dim *= dim
        else
            left_dim *= dim
    }

    %perm_const = tosa.CONST() {value={0, 2, 1}}
    %op1_reshape_on_value = tosa.RESHAPE(%on_value) {shape={1, 1, 1}}
    %op2_tile_op1 = tosa.TILE(%op1_reshape_on_value) {multiples={N, W, C}}
    %op3_reshape_off_value = tosa.RESHAPE(%off_value) {shape={1, 1, 1}}
    %op4_tile_op1 = tosa.TILE(%op3_reshape_off_value) {multiples={N, K, C}}
    %op5_reshape_indices = tosa.RESHAPE(%indices) {shape={N, W}}
    %op6_scatter_op4_op5_op2 = tosa.SCATTER(%op4_tile_op1, %op5_reshape_indices, %op2_tile_op1)
    %op7_reshape_op6 = tosa.RESHAPE(%op6_scatter_op4_op5_op2) {shape={left_dim, right_dim, K}}
    %op8_transpose_op7 = tosa.TRANSPOSE(%op7_reshape_op6, %perm_const)
    %op9_reshape_op8 = tosa.RESHAPE(%op8_transpose_op7) {shape=%output.shape}
}


## MLIR Passes Management

Legalization is built on multiple MLIR passes.

| MLIR Pass Name            | Input Dialect | Output Dialect | Description     |
| ------------------------- | ------------- | -------------- | --------------- |
| legalize_tf               | TensorFlow    | TOSA           | Legalize        |
:                           :               :                : TensorFlow      :
:                           :               :                : dialect to TOSA :
:                           :               :                : dialect         :
| fuse_tf_bias              | TensorFlow    | TOSA           | Mapping         |
:                           :               :                : tf.BiasAdd +    :
:                           :               :                : tf.Conv2D to    :
:                           :               :                : tosa.CONV2D     :
| legalize_tfl              | TensorFlow    | TOSA           | Legalize        |
:                           : Lite          :                : TensorFlow Lite :
:                           :               :                : dialect to TOSA :
:                           :               :                : dialect         :
| convert_tfl_uint8         | TensorFlow    | TensorFlow     | Convert         |
:                           : Lite          : Lite           : quantized uint8 :
:                           :               :                : graph to int8   :
:                           :               :                : graph           :

TF to TOSA legalization could be summarized by following pseudocode:

```

void legalize_tf_to_tosa(mlir::Module module) { mlir::PassManager pm

```
// other MLIR passes to optimize TF

pm.addPass(fuse_tf_bias)
pm.addPass(legalize_tf)

// other MLIR passes to optimize TOSA
```

} ```

TFLite to TOSA legalization could be summarized by following pseudocode:

```
void legalize_tfl_to_tosa(mlir::Module module)
{
    mlir::PassManager pm

    // other MLIR passes to optimize TFLite

    pm.addPass(convert_tfl_uint8)
    pm.addPass(legalize_tfl)

    // other MLIR passes to optimize TOSA
}
```

Each of the passes is described in more detail in the subsequent chapters.

## TensorFlow MLIR Dialect Legalization (legalize_tf)

### tf.Abs

This operator is trivially lowered to tosa.ABS

### tf.AddN

**TensorFlow Dialect**

```
%output = tf.AddN(%inputs)
```

**TOSA Lowering**

```
%output = tosa.ADD(%inputs:0, %inputs:1)
for (int32 i = 2; i < %inputs.size; i++) {
    %output = tosa.ADD(%inputs:i, %output)
}
```

### tf.Add

Element-wise addition.

**TensorFlow Dialect**

```
%output = tf.Add(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.ADD.

### tf.Addv2

Element-wise addition.

**TensorFlow Dialect**

```
%output = tf.Addv2(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.ADD.

### tf.All

Computes the "logical and" of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.all(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_ALL>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.Any

Computes the "logical or" of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.any(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_ANY>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.ArgMax

Returns the index with the largest value across the given axis of the input
tensor.

**TensorFlow Dialect**

```
%output = tf.ArgMax(%input, %dimension)
```

**TOSA Lowering**

```
int64 axis = positive_axis(%dimension)
%output = tosa.ARGMAX(%input) {axis=axis}
```

### tf.ArgMin

Returns the index with the smallest value across the given axis of the input
tensor.

**TensorFlow Dialect**

```
%output = tf.ArgMin(%input, %dimension)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.Assert

Asserts that the given condition is true.

**TensorFlow Dialect**

```
%output = tf.Assert(%condition, %summarize)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.AssignAddVariableOp

Adds a value to the current value of a variable.

**TensorFlow Dialect**

```
%output = tf.AssignAddVariableOp(%resource, %value, %dtype)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.AssignSubVariableOp

Subtracts a value to the current value of a variable.

**TensorFlow Dialect**

```
%output = tf.AssignSubVariableOp(%resource, %value, %dtype)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.AssignVariableOp

Assigns a new value to a variable.

**TensorFlow Dialect**

```
%output = tf.AssignVariableOp(%resource, %value, %dtype)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.AvgPool

Performs average pooling on the input.

**TensorFlow Dialect**

```
%output = tf.AvgPool(%value) {ksize, strides, padding, data_format}
```

**TOSA Lowering**

```
assert(data_format == "NHWC")

tosa_padding =
     get_padding_values_from_pad_type(%input, ksize, padding, data_format,
                                      FORMAT_OHWI, strides, {1, 1, 1, 1})
%output = tosa.AVG_POOL2D(%value) {ksize=ksize, strides=strides, padding=tosa_padding}
```

### tf.BatchMatMul

Multiplies slices of two tensors in batches.

**TensorFlow Dialect**

```
%output = tf.BatchMatMul(%x, %y, %adj_x, %adj_y)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.BatchMatMulV2

Multiplies slices of two tensors in batches.

**TensorFlow Dialect**

```
%output = tf.BatchMatMulV2(%x, %y, %adj_x, %adj_y)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.BatchNormWithGlobalNormalization

✗ Deprecated operator.

### tf.BatchToSpaceND

BatchToSpaceND for N-D tensors of type T.

**TensorFlow Dialect**

```
%output = tf.BatchToSpaceND(%input, %block_shape, %crops)
```

**TOSA Lowering**

```
%output = lower_batch_to_space_nd_op(%input, %block_shape, %crops, output.shape)
```

### tf.BiasAddGrad

Training profile: TOSA lowering not yet defined.

### tf.BiasAdd

Add bias to value.

**TensorFlow Dialect**

```
%output = tf.BiasAdd(%bias, %value) {data_format}
```

**TOSA Lowering**

```
assert(data_format == 'NHWC')
%output = tosa.ADD(%value, %bias)
```

### tf.BitCast

Bitcasts a tensor from one type to another without copying data.

**TensorFlow Dialect**

```
%output = tf.BitCast(%input, %dtype)
```

**TOSA Lowering**

No TOSA lowering defined.

### tf.BitwiseAnd

This operator is trivially lowered to tosa.BITWISE_AND.

### tf.BitwiseOr

This operator is trivially lowered to tosa.BITWISE_OR.

### tf.BroadcastGradientArgs

Training profile: TOSA lowering not yet defined.

### tf.BroadcastTo

No TOSA lowering defined.

### tf.Cast

This operator is trivially lowered to tosa.CAST.

### tf.Ceil

This operator is trivially lowered to tosa.CEIL.

### tf.CheckNumerics

No TOSA lowering defined.

### tf.ComplexAbs

No TOSA lowering defined.

### tf.Complex

No TOSA lowering defined.

### tf.ConcatOffset

No TOSA lowering defined. Training profile: TOSA lowering not yet defined.

### tf.Concat

No TOSA lowering defined.

### tf.ConcatV2

Concatenates tensors along one dimension.

**TensorFlow Dialect**

```
%output = tf.ConcatV2(%values, %axis)
```

**TOSA Lowering**

```
%output = lower_concatv2_op(%values, %axis)
```

### tf.Conj

No TOSA lowering defined.

### tf.Const

This operator is trivially lowered to tosa.CONST.

### tf.Conv2DBackpropFilter

No TOSA lowering defined.

### tf.Conv2DBackpropInput

Computes the gradients of convolution with respect to the input.

**TensorFlow Dialect**

```
%output = tf.Conv2DBackpropInput(%input_sizes, %filter, %out_backprop) {strides, use_cudnn_on_gpu, padding, explicit_paddings, data_format, dilations}
```

**TOSA Lowering**

```
// Transpose filter from HWIO to OHWI
%tosa_filter = tosa.TRANSPOSE(%filter) {perms={2, 0, 1, 3}}

vector output_shape

for (int32 i = 0; i < input_sizes.size(); i++) {
   output_shape.push_back(input_size[i])
}

if (%padding == "EXPLICIT") {
   tosa_padding =
       get_padding_values_from_explicit_pad_attr(explict_padding, data_format)
} else {
    tosa_padding =
        get_transpose_conv2d_padding_values_from_pad_type(%input_sizes, %filter, output_shape, padding, data_format, FORMAT_HWIO, strides, dilations)
}

// Create a zero bias tensor
%zero_bias = tosa.CONST() {value={0}}
%output = tosa.TRANSPOSE_CONV2D(%out_backprop) {weight=%tosa_filter, bias=%zero_bias, outpad=tosa_pading, stride=strides, dilation==dilations, out_shape=out_shape}
```

### tf.Conv2D

Computes a 2-D convolution given 4-D input and filter tensors.

**TensorFlow Dialect**

```
%output = tf.Conv2D(%input, %filter) {strides, padding, explicit_paddings, data_format, dilations}
```

**TOSA Lowering**

```
assert(data_format == "NHWC")

// Transpose filter from HWIO to OHWI
%filter_tranpose = tosa.TRANSPOSE(%filter {perms={3, 0, 1, 2}}

if (padding == "EXPLICIT") {
   tosa_padding =
       get_padding_values_from_explicit_pad_attr(explict_padding, data_format)
} else {
    %tosa_padding =
        get_padding_values_from_pad_type(%input, %filter.shape, padding, data_format,
                                         FORMAT_HWIO, strides, dilations)
}

// Create a zero bias tensor
%zero_bias = tosa.CONST() {value={0}}

%output = tosa.CONV2D(%input, %filter_transpose, %zero_bias) {padding=tosa_padding, stride=strides, dilation=dilations}
```

### tf.Conv3D

TOSA lowering to tosa.CONV3D to be defined.

### tf.Cos

No TOSA lowering defined.

### tf.CrossReplicaSum

No TOSA lowering defined.

### tf.DepthToSpace

DepthToSpace for tensors of type T.

**TensorFlow Dialect**

```
%output = tf.DepthToSpace(%input) {block_size, data_format}
```

**TOSA Lowering**

```
%output = lower_depth_to_space_op(%input, block_size, data_format)
```

### tf.DepthwiseConv2dNative

Computes a 2-D depthwise convolution given 4-D input and filter tensors.

**TensorFlow Dialect**

```
%output = tf.DepthwiseConv2dNative(%input, %filter) {strides, padding, data_format, dilations}
```

**TOSA Lowering**

```
if (padding == "EXPLICIT") {
   tosa_padding =
       get_padding_values_from_explicit_pad_attr(explict_padding, data_format)
} else {
    tosa_padding =
        get_padding_values_from_pad_type(%input, %filter.shape, padding, data_format,
                                         FORMAT_HWIO, strides, dilations)
}

bias_dim = %filter.shape[2] * %filter.shape[3]

// Create a zero-bias tensor
%zero_bias = tosa.CONST() {value={0} * bias_dim}

%output = tosa.DEPTHWISE_CONV2D(%input, %filter, %zero_bias) {stride=strides, dilation=dilations, padding=padding}
```

### tf.DivNoNan

No TOSA lowering defined.

### tf.Div

No TOSA lowering defined.

### tf.DynamicStitch

No TOSA lowering defined.

### tf.Einsum

No TOSA lowering defined.

### tf.Elu

Computes exponential linear: exp(features) - 1 if &lt;0, features otherwise

**TensorFlow Dialect**

```
%output = tf.Elu(%features)
```

**TOSA Lowering**

```
%output = lower_elu_op(%features)
```

### tf.EmptyTensorList

No TOSA lowering defined.

### tf.Equal

Returns the truth value of (x == y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.Equal(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.EQUAL.

### tf.Exp

This operator is trivially lowered to tosa.EXP.

### tf.ExpandDims

Inserts a dimension of 1 into a tensor’s shape

**TensorFlow Dialect**

```
%output = tf.ExpandDims(%input, %axis)
```

**TOSA Lowering**

```
%output = lower_expand_dims(%input, %axis.to_constant())
```

### tf.FakeQuantWithMinMaxArgs

Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

**TensorFlow Dialect**

```
%output = tf.FakeQuantWithMinMaxArgs(%inputs) {min, max, num_bits, narrow_range}
```

**TOSA Lowering**

```
%output = lower_fake_quant_op(%inputs, %min, %max, %num_bits, %narrow_range)
```

### tf.FakeQuantWithMinMaxVars

Fake-quantize the 'inputs' tensor of type float via global flats scalars min.

**TensorFlow Dialect**

```
%output = tf.FakeQuantWithMinMaxVars(%inputs, %min, %max) {num_bits, narrow_range}
```

**TOSA Lowering**

```
%output = lower_fake_quant_op(%inputs, %output.type, %min.to_constant(), %max.to_constant(), num_bits, narrow_range)
```

### tf.FakeQuantWithMinMaxVarsPerChannel

Fake-quantize the 'inputs' tensor of type float and one of the shapes \[d\].

**TensorFlow Dialect**

```
%output = tf.FakeQuantWithMinMaxVarsPerChannel(%inputs, %min, %max) {num_bits, narrow_range}
```

No TOSA lowering defined.

### tf.Fill

Creates a tensor filled with a scalar value

**TensorFlow Dialect**

```
%output = tf.Fill(%dims, %value)
```

**TOSA Lowering**

```
int64 total_size = 1

for (int32 i = 0; i < %dims.shape[0]; i++) {
    total_size *= %dims[i]
}

vector<%value.dtype> fill_arr(total_size, %value)

%output = tosa.CONST() {value={fill_arr}}
```

### tf.FloorDiv

Returns x // y element-wise.

**TensorFlow Dialect**

```
%output = tf.FloorDiv(%x, %y)
```

**TOSA Lowering**

```
%output = lower_floor_div(%lhs, %rhs)
```

### tf.FloorMod

Returns element-wise remainder of division when x &lt; 0 xor x &lt; y is true.

**TensorFlow Dialect**

```
%output = tf.FloorMod(%x, %y)
```

**TOSA Lowering**

```
%output = lower_floor_mod(%lhs, %rhs)
```

### tf.Floor

This operator is trivially lowered to tosa.FLOOR.

### tf.FusedBatchNormGrad

Training profile: TOSA lowering not yet defined.

### tf.FusedBatchNormGradV2

Training profile: TOSA lowering not yet defined.

### tf.FusedBatchNormGradV3

Training profile: TOSA lowering not yet defined.

### tf.FusedBatchNorm

Batch normalization.

**TensorFlow Dialect**

```
%output = tf.FusedBatchNorm(%x, %scale, %offset, %mean, %variance) {epsilon, data_format, is_training}


assert(data_format == 'NHWC')
assert(is_training == false)

%epsilon_const = tosa.CONST() {value={epsilon}}

%op1 = tosa.SUB(%x, %bmean)
%op2 = tosa.ADD(%variance, %epsilon_const)
%op3 = tosa.RSQRT(%op2)
%op4 = tosa.MUL(%op1, %op3)
%op5 = tosa.MUL(%op4, %scale)
%output = tosa.ADD(%op5, %offset)
```

### tf.FusedBatchNormV3

Batch normalization.

**TensorFlow Dialect**

```
%output = tf.FusedBatchNormV3(%x, %scale, %offset, %mean, %variance) {epsilon, data_format, is_training}
```

**TOSA Lowering**

```
assert(data_format == 'NHWC')
assert(is_training == false)

%epsilon_const = tosa.CONST() {value={epsilon}}

%op1 = tosa.SUB(%x, %bmean)
%op2 = tosa.ADD(%variance, %epsilon_const)
%op3 = tosa.RSQRT(%op2)
%op4 = tosa.MUL(%mean, %op3)
%op5 = tosa.MUL(%op4, %scale)
%output = tosa.ADD(%op5, %offset)
```

### tf.GatherNd

Gather slices from params into a Tensor with shape specified by indices.

**TensorFlow Dialect**

```
%output = tf.GatherNd(%params, %indices)
```

**TOSA Lowering**

```
%output = lower_gather_nd_op(%params, %indices)
```

### tf.Gather

Gathers slices from params according to indices.

**TensorFlow Dialect**

```
%output = tf.Gather(%params, %indices)
```

**TOSA Lowering**

```
%output = lower_gather_op(%params, %indices, 0, 0)
```

### tf.GatherV2

Gathers slices from params axis according to indices.

**TensorFlow Dialect**

```
%output = tf.GatherV2(%params, %indices, %axis) {batch_dims}
```

**TOSA Lowering**

```
%output = lower_gather_op(%params, %indices, batch_dims, %axis.to_constant())
```

### tf.GreaterEqual

Returns the truth value of (x &gt;= y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.GreaterEqual(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.GREATER_EQUAL.

### tf.Greater

RetruReturns the truth value of (x &gt; y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.Greater(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.GREATER.

### tf.HashTableV2

No TOSA lowering defined.

### tf.IdentityN

Returns a list of tensors with the same shapes and contents as the input.

**TensorFlow Dialect**

```
%output = tf.IdentityN(%input)
```

**TOSA Lowering**

```
%output = tosa.IDENTITYN(%input)
```

### tf.Identity

Returns a tensor with the same shape and contents as the input.

**TensorFlow Dialect**

```
%output = tf.Identity(%input)
```

**TOSA Lowering**

```
%output = tosa.IDENTITY(%input)
```

### tf.If

No TOSA lowering defined.

### tf.Imag

No TOSA lowering defined.

### tf.InfeedDequeueTuple

No TOSA lowering defined.

### tf.Invert

This operator is trivially lowered to tosa.BITWISE_NOT.

### tf.InvertPermutation

No TOSA lowering defined.

### tf.IsFinite

No TOSA lowering defined.

### tf.IteratorGetNext

No TOSA lowering defined.

### tf.L2Loss

Training profile: TOSA lowering not yet defined.

### tf.LRN

No TOSA lowering defined.

### tf.LeakyRelu

Computes rectified linear: max(features, features \* alpha).

**TensorFlow Dialect**

```
%output = tf.LeakyRelu(%features) {alpha}
```

**TOSA Lowering**

```
%alpha_tensor = tosa.CONST() {value={alpha}}
%features_alpha = tosa.MUL(%features, %alpha_tensor)
%greater = tosa.GREATER(%features, %features_alpha)
%output = tosa.SELECT(%greater, %features, %features_alpha)
```

### tf.LeftShift

Computes the bitwise left-shift of x by y bits, element-wise.

**TensorFlow Dialect**

```
%output = tf.LeftShift(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.LOGICAL_LEFT_SHIFT.

### tf.LegacyCall

No TOSA lowering defined.

### tf.LessEqual

Returns the truth value of (x ⇐ y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.LessEqual(%x, %y)
```

**TOSA Lowering**

```
%output_greater = tosa.GREATER(%x, %y)
%output = tosa.LOGICAL_NOT(%output_greater)
```

### tf.Less

Returns the truth value of (x &lt; y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.LessEqual(%x, %y)
```

**TOSA Lowering**

```
%output_greater_equal = tosa.GREATER_EQUAL(%x, %y)
%output = tosa.LOGICAL_NOT(%output_greater_equal)
```

### tf.LiNSpace

No TOSA lowering defined.

### tf.Log1p

No TOSA lowering defined.

### tf.Log

This operator is trivially lowered to tosa.LOG.

### tf.LogSoftmax

Computes log softmax activations.

**TensorFlow Dialect**

```
%output = tf.LogSoftmax(%logits)
```

**TOSA Lowering**

```
%output = lower_log_softmax_op(%logits)
```

### tf.LogicalAnd

Returns the truth value of x AND y, element-wise.

**TensorFlow Dialect**

```
%output = tf.LogicalAnd(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.LOGICAL_AND.

### tf.LogicalNot

This operator is trivially lowered to tosa.LOGICAL_NOT.

### tf.LogicalOr

Returns the truth value of x OR y, element-wise.

**TensorFlow Dialect**

```
%output = tf.LogicalOr(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.LOGICAL_OR.

### tf.LookupTableFindV2

No TOSA lowering defined.

### tf.LookupTableInputV2

No TOSA lowering defined.

### tf.LookupTableSizeV2

No TOSA lowering defined.

### tf.MatMul

Multiply the matrix a by the matrix b

**TensorFlow Dialect**

```
%output = tf.MatMul(%a, %b)
```

**TOSA Lowering**

```
%output = tosa.MATMUL(%a, %b)
```

### tf.MatrixDiag

No TOSA lowering defined.

### tf.MatrixDiagV2

No TOSA lowering defined.

### tf.MatrixDiagV3

No TOSA lowering defined.

### tf.MatrixSetDiag

No TOSA lowering defined.

### tf.MatrixSetDiagV2

No TOSA lowering defined.

### tf.MatrixSetDiagV3

No TOSA lowering defined.

### tf.Max

Computes the maximum of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Max(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_MAX>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.MaxPoolGrad

Training profile: TOSA lowering not yet defined.

### tf.MaxPool

Performs max pooling on the input.

**TensorFlow Dialect**

```
%output = tf.MaxPool(%input) {ksize, strides, padding, data_format}
```

**TOSA Lowering**

```
assert(data_format == "NHWC")

tosa_padding =
     get_padding_values_from_pad_type(%input, ksize, padding, data_format,
                                      FORMAT_OHWI, strides, {1, 1, 1, 1})
%output = tosa.MAX_POOL2D(%value) {ksize=ksize, strides=strides, padding=tosa_padding}
```

### tf.Maximum

This operator is trivially lowered to tosa.MAXIMUM.

### tf.Mean

Computes the mean of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Mean(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
int32 num_elements_on_axis = 1
for (int32 axis : %reduction_indices) {
    num_elements_on_axis *= %input.shape[axis]
}
float32 div_scale = 1.0 / num_elements_on_axis

%cst_div_scale = tosa.CONST() {value={div_scale}}
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_SUM>(%input, %output.shape, %reduction_indices, keep_dims)
%op2_mul_op1 = tosa.MUL(%op1_rsum_in, %cst_div_scale)
```

### tf.Min

Computes the minimum of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Min(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_MIN>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.Minimum

This operator is trivially lowered to tosa.MAXIMUM.

### tf.MirrorPad

No TOSA lowering defined.

### tf.MlirPassthroughOp

No TOSA lowering defined.

### tf.MulNoNan

No TOSA lowering defined.

### tf.Mul

Returns the product of x and y, element-wise.

**TensorFlow Dialect**

```
%output = tf.Mul(%x, %y)
```

**TOSA Lowering** This operator is trivially lowered to tosa.MUL.

### tf.Neg

This operator is trivially lowered to tosa.NEGATE.

### tf.NoOp

No TOSA lowering defined.

### tf.NonMaxSuppressionV4

No TOSA lowering defined.

### tf.NonMaxSuppressionV5

No TOSA lowering defined.

### tf.NotEqual

Returns the truth value of (x != y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.NotEqual(%x, %y)
```

**TOSA Lowering**

```
%equal = tosa.EQUAL(%x, %y)
%output = tosa.NOT(%equal)
```

### tf.OneHot

OneHot operator.

**TensorFlow Lite Dialect**

```
%output = tf.OneHot(%indices, %depth, %on_value, %off_value) {axis}
```

**TOSA Lowering**

```
%output = lower_one_hot_op(%indices, %depth, %on_value, %off_value, axis)
```

### tf.OutputEnqueueTuple

No TOSA lowering defined.

### tf.Pack

Packs a list of N rank-R tensors into one rank-(R+1) tensor.

**TensorFlow Dialect**

```
%output = tf.Pack(%values) {axis}
```

**TOSA Lowering**

```
%output = lower_pack_op(%values, axis)
```

### tf.Pad

This operator is trivially lowered to tosa.PAD.

### tf.PadV2

No TOSA lowering defined.

### tf.ParseExampleV2

No TOSA lowering defined.

### tf.PartitionedCall

No TOSA lowering defined.

### tf.Placeholder

Not seen in practice. No lowering needed.

### tf.PlaceholderWithDefault

Not seen in practice. No lowering needed.

### tf.Pow

This operator is trivially lowered to tosa.POW.

### tf.PreventGradient

Training profile: TOSA lowering not yet defined.

### tf.Prod

Computes the product of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Prod(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_PRODUCT>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.QuantizeAndDequantize

No TOSA lowering defined.

### tf.QuantizeAndDequantizeV2

No TOSA lowering defined.

### tf.QuantizeAndDequantizeV3

No TOSA lowering defined.

### tf.RFFT

No TOSA lowering defined.

### tf.RandomShuffle

No TOSA lowering defined.

### tf.RandomStandardNormal

No TOSA lowering defined.

### tf.RandomUniform

No TOSA lowering defined.

### tf.Range

No TOSA lowering defined.

### tf.Rank

Returns the rank of the tensor.

**TensorFlow Dialect**

```
%output = tf.Rank(%input)
```

**TOSA Lowering**

```
%output = tosa.CONST() {value={%input.rank}}
```

### tf.ReadVariableOp

No TOSA lowering defined.

### tf.RealDiv

Returns x / y element-wise for real types.

**TensorFlow Dialect**

```
%output = tf.RealDiv(%x, %y)
```

**TOSA Lowering**

```
%recip = tosa.RECIPROCAL(%y)
%output = tosa.MUL(%x, %recip)
```

### tf.Real

No TOSA lowering defined.

### tf.Reciprocal

This operator is trivially lowered to tosa.RECIPROCAL.

### tf.Relu6

Computes rectified linear 6: min(max(features, 0), 6).

**TensorFlow Dialect**

```
%output = tf.Relu6(%features)
```

**TOSA Lowering**

```
%output = tosa.RELUN(%features) {max_val=6}
```

### tf.ReluGrad

Training profile: TOSA lowering not yet defined.

### tf.Relu

Computes rectified linear 6: max(features, 0)

**TensorFlow Dialect**

```
%output = tf.Relu(%features)
```

**TOSA Lowering**

```
%output = tosa.RELUN(%features) {max_val=0}
```

### tf.Reshape

Reshapes a tensor.

**TensorFlow Dialect**

```
%output = tf.Reshape(%tensor, %shape)
```

**TOSA Lowering**

```
%output = tosa.RESHAPE(%tensor) {new_shape=%shape.as_constant}
```

### tf.ResizeBilinear

Resizes images to size using bilinear interpolation.

**TensorFlow Dialect**

```
%output = tf.ResizeBilinear(%images, %size) {align_corners, half_pixel_centers}
```

inferred from output shape. **TOSA Lowering**

```
%output = lower_resize_op(%images, %size, float, "BILINEAR")
```

### tf.ResizeNearestNeighbor

Resizes images to size using nearest neighbor interpolation.

**TensorFlow Dialect**

```
%output = tf.ResizeNearestNeighbor(%images, %size) {align_corners, half_pixel_centers}
```

inferred from output shape. **TOSA Lowering**

```
%output = lower_resize_op(%images, %size, %output, float, "NEAREST_NEIGHBOR")
```

### tf.ResourceApplyAdam

Training profile: TOSA lowering not yet defined.

### tf.ResourceApplyGradientDescent

Training profile: TOSA lowering not yet defined.

### tf.ResourceApplyKerasMomentum

Training profile: TOSA lowering not yet defined.

### tf.ResourceGather

Training profile: TOSA lowering not yet defined.

### tf.ResourceScatterUpdate

Training profile: TOSA lowering not yet defined.

### tf.ReverseSequence

No TOSA lowering defined.

### tf.ReverseV2

Reverses specific dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.ReverseV2(%tensor, %axis)
```

**TOSA Lowering**

```
%output = lower_reversev2_op(%tensor, %axis)
```

### tf.RightShift

Computes the bitwise left-shift of x by y bits, element-wise.

**TensorFlow Dialect**

```
%output = tf.LeftShift(%x, %y)
```

**TOSA Lowering**

```
if (is_unsigned(%x.dtype)) {
  %output = tosa.LOGICAL_RIGHT_SHIFT(%x, %y)
} else {
  %output = tosa.ARITHMETIC_RIGHT_SHIFT(%x, %y)
}
```

### tf.Round

Rounds the values of a tensor to the nearest integer, element-wise.

**TensorFlow Dialect**

```
%output = tf.Round(%x)
```

**TOSA Lowering**

```
%output = lower_round_op(%x)
```

### tf.RsqrtGrad

Training profile: TOSA lowering not yet defined.

### tf.Rsqrt

This operator is trivially lowered to tosa.RSQRT.

### tf.SegmentMax

No TOSA lowering defined.

### tf.SegmentMean

No TOSA lowering defined.

### tf.SegmentMin

No TOSA lowering defined.

### tf.SegmentProd

No TOSA lowering defined.

### tf.SegmentSum

No TOSA lowering defined.

### tf.Select

No TOSA lowering defined.

### tf.SelectV2

Selects elements from t or e depending on condition.

**TensorFlow Dialect**

```
%output = tf.SelectV2(%condition, %t, %e)
```

**TOSA Lowering**

```
%output = lower_selectv2_op(%condition, %t, %e, %output.shape)
```

### tf.ShapeN

No TOSA lowering defined.

### tf.Shape

Returns the shape of a tensor.

**TensorFlow Dialect**

```
%output = tf.Shape(%input)
```

**TOSA Lowering**

```
%output = lower_shape_op(%input)
```

### tf.Sigmoid

This operator is trivially lowered to tosa.SIGMOID.

### tf.Sign

No TOSA lowering defined.

### tf.Sin

No TOSA lowering defined.

### tf.Size

No TOSA lowering defined.

### tf.Slice

Returns a slice from input.

**TensorFlow Dialect**

```
%output = tf.Slice(%input, %begin, %size)
```

**TOSA Lowering**

```
vector <size_t> output_size
try {
  output_size = %size.as_constant()
} except(ConversionFailed) {
  output_size = %output.shape
}

%output = tosa.SLICE(%input) {start=begin, size=output_size}
```

### tf.Snapshot

No TOSA lowering defined.

### tf.SoftmaxCrossEntropyWithLogits

Training profile: TOSA lowering not yet defined.

### tf.Softmax

Computes softmax activations

**TensorFlow Dialect**

```
%output = tf.Softmax(%logits)
```

**TOSA Lowering**

```
%op1 = tosa.EXP(%logits)
%op2 = tosa.REDUCE_SUM(op1) {reduce_axis=(%logits.rank - 1)}
%op3 = tosa.RECIPROCAL(%op2)
%output = tosa.MUL(%op1, %op3)
```

### tf.Softplus

No TOSA lowering defined.

### tf.SpaceToBatchND

SpaceToBatch for N-D tensors of type T.

**TensorFlow Dialect**

```
%output = tf.SpaceToBatchND(%input, %block_shape, %paddings)
```

**TOSA Lowering**

```
%output = lower_space_to_batch_nd_op(%input, %block_shape, %paddings)
```

### tf.SpaceToDepth

SpaceToDepth for tensors of type T.

**TensorFlow Dialect**

```
%output = tf.SpaceToDepth(%input) {block_size, data_format}
```

**TOSA Lowering**

```
%output = lower_space_to_depth_op(%input, block_size, data_format)
```

### tf.SparseMatMul

No TOSA lowering defined.

### tf.SparseSoftmaxCrossEntropyWithLogits

No TOSA lowering defined.

### tf.SparseToDense

No TOSA lowering defined.

### tf.Split

Splits a tensor into num_split tensors along one dimension

**TensorFlow Dialect**

```
%output = tf.Split(%split_dim, %value) {num_split}
```

**TOSA Lowering**

```
%output = lower_split_op(%value, %split_dim.as_constant(), num_split)
```

### tf.SplitV

Splits a tensor into num_split tensors along one dimension

**TensorFlow Dialect**

```
%output = tf.SplitV(%value, %size_splits, %split_dim) {num_split}
```

**TOSA Lowering**

```
%output = lower_splitv_op(%value, %size_splits.as_constant(), %split_dim.as_constant())
```

### tf.Sqrt

No TOSA lowering defined.

### tf.Square

Computes the square of x, element-wise.

**TensorFlow Dialect**

```
%output = tf.Square(%x)
```

**TOSA Lowering**

```
%output = tosa.MUL(%x, %x)
```

### tf.SquareDifference

Computes (x-y)\*(x-y) element-wise

**TensorFlow Dialect**

```
%output = tf.SquareDifference(%x, %y)
```

**TOSA Lowering**

```
%diff = tosa.SUB(%x, %y)
%output = tosa.MUL(%diff, %diff)
```

### tf.Squeeze

Removes dimensions of size 1 from the shape of a tensor.

**TensorFlow Dialect**

```
%output = tf.Squeeze(%input) {squeeze_dims}
```

**TOSA Lowering**

```
%output = lower_squeeze_op(%input, squeeze_dims)
```

### tf.StatefulPartitionedCall

No TOSA lowering defined.

### tf.StopGradient

Training profile: TOSA lowering not yet defined.

### tf.StridedSliceGrad

Training profile: TOSA lowering not yet defined.

### tf.StridedSlice

Return a strided slice from input.

**TensorFlow Dialect**

```
%output = tf.StridedSlice(%input, %begin, %end, %strides) {begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask}
```

**TOSA Lowering**

```
%output = lower_strided_slice_op(%input, %begin, %end, %strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
```

### tf.Sub

This operator is trivially lowered to tosa.SUB.

### tf.Sum

Computes the sum of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Sum(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_SUM>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tf.TPUCompilationResult

No TOSA lowering defined.

### tf.TPUCopyWithLayout

No TOSA lowering defined.

### tf.TPUExecuteAndUpdateVariables

No TOSA lowering defined.

### tf.TPUExecute

No TOSA lowering defined.

### tf.TPUGetLayout

No TOSA lowering defined.

### tf.TPUReplicateMetadata

No TOSA lowering defined.

### tf.TPUReplicatedInput

No TOSA lowering defined.

### tf.TPUReplicatedOutput

No TOSA lowering defined.

### tf.TPUReshardVariables

No TOSA lowering defined.

### tf.TanhGrad

Training profile: TOSA lowering not yet defined.

### tf.Tanh

This operator is trivially lowered to tosa.TANH.

### tf.TensorListFromTensor

No TOSA lowering defined.

### tf.TensorListGetItem

No TOSA lowering defined.

### tf.TensorListLength

No TOSA lowering defined.

### tf.TensorListPushBack

No TOSA lowering defined.

### tf.TensorListReserve

No TOSA lowering defined.

### tf.TensorListResize

No TOSA lowering defined.

### tf.TensorListSetItem

No TOSA lowering defined.

### tf.TensorListStack

No TOSA lowering defined.

### tf.TensorScatterUpdate

No TOSA lowering defined.

### tf.Tile

Constructs a tensor by tiling a given tensor.

**TensorFlow Dialect**

```
%output = tf.Tile(%input, %multiples)
```

**TOSA Lowering**

```
%output = tosa.TILE(%input) {multiples=%multiples.as_constant()}
```

### tf.ToBool

No TOSA lowering defined.

### tf.TopKV2

No TOSA lowering defined.

### tf.Transpose

Shuffle dimensions of x according to a permutation.

**TensorFlow Dialect**

```
%output = tf.Transpose(%x, %perm)
```

**TOSA Lowering**

```
%output = tosa.TRANSPOSE(%x) {perm=%perm.as_constant()}
```

### tf.TruncateDiv

No TOSA lowering defined.

### tf.Unique

No TOSA lowering defined.

### tf.Unpack

Unpacks a given dimension of a rank-R tensor into num rank-(R-1) tensors.

**TensorFlow Dialect**

```
%output = tf.Unpack(%value) {axis, num}
```

**TOSA Lowering**

```
%output = lower_unpack_op(%value, axis, num)
```

### tf.UnsortedSegmentMax

No TOSA lowering defined.

### tf.UnsortedSegmentMin

No TOSA lowering defined. === tf.UnsortedSegmentProd

No TOSA lowering defined. === tf.UnsortedSegmentSum

No TOSA lowering defined.

### tf.VarHandle

No TOSA lowering defined.

### tf.VariableShape

No TOSA lowering defined.

### tf.Where

No TOSA lowering defined.

### tf.While

No TOSA lowering defined.

### tf.Xdivy

No TOSA lowering defined.

### tf.XlaDynamicUpdateSlice

No TOSA lowering defined.

### tf.XlaSharding

No TOSA lowering defined.

### tf.ZerosLike

Returns a tensor of zeros with the same shape and type as x.

**TensorFlow Dialect**

```
%output = tf.ZerosLike(%x)
```

**TOSA Lowering**

```
%output = tosa.CONST() {value={0} * %x.num_elements}
```

## TensorFlow Lite MLIR Dialect Legalization (legalize_tfl)

### tfl.abs

This operator is trivially lowered to tosa.ABS

### tfl.add_n

add_n operator.

**TensorFlow Lite Dialect**

```
%sum = tfl.add_n(%inputs)
```

**TOSA Lowering**

```
%output = tosa.ADD(%inputs:0, %inputs:1)
for (int32 i = 2 i < %inputs.size i++) {
    %output = tosa.ADD(%inputs:i, %output)
}
```

### tfl.add

Element-wise addition operation.

**TensorFlow Lite Dialect**

```
%output = tfl.add(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%result = tosa.ADD(%lhs, %rhs)
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 max_scale_2x = 2.0 * max(%lhs.scale, %rhs.scale)
float64 lhs_scale = float64(1 << input_shift) * %lhs.scale / max_scale_2x
float64 rhs_scale = float64(1 << input_shift) * %rhs.scale / max_scale_2x
float64 output_scale = max_scale_2x / (%output.scale * float64(1 << input_shift))

```

Legalization:

```
%op1_rescale_lhs = tosa.RESCALE(%lhs) {scale=lhs_scale, input_zp=%lhs.zp, output_zp=0} // %lhs.dtype->i32
%op2_rescale_rhs = tosa.RESCALE(%rhs) {scale=rhs_scale, input_zp=%rhs.zp, output_zp=0} // %rhs.dtype->i32
%op3_add_op1_op2 = tosa.ADD(%op1_rescale_lhs, %op2_rescale_rhs)
%op4_rescale_op3 = tosa.RESCALE(%op3_add_op1_op2) {scale=output_scale} // i32->%output.dtype
```

### tfl.arg_max

ArgMax operator.

**TensorFlow Lite Dialect**

```
%output = tfl.arg_max(%input, %dim)
```

**TOSA Lowering**

```
%result = tosa.ARGMAX(%input) {axis=positive_axis(%dim_const.as_constant(), %input.rank)}
```

### tfl.arg_min

No TOSA lowering defined.

### tfl.average_pool_2d

Average_pool_2d operator.

**TensorFlow Lite Dialect**

```
%output = tfl.average_pool_2d(%input) {filter_height, filter_width, padding, stride_h, stride_w, fused_activation_function}
```

**TOSA Lowering**

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, tensor<{filter_height, filter_width}, tosa.int32>,
                                      {1, stride_h, stride_w, 1}, {1, 1, 1, 1})
```

If input/output tensors are all non-quantized typed,

Legalization:

```
%avgpool2d = tosa.AVG_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%avgpool2d, fused_activation)
}
else {
    %result = %avgpool2d
}
```

If input/output tensors are all quantized typed,

Legalization:

```
%avgpool2d = tosa.AVG_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding, quantization_info={input_zp=%input.zp, output_zp=%output.zp}}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%avgpool2d, fused_activation)
}
else {
    %result = %avgpool2d
}
```

### tfl.basic_lstm

No TOSA lowering defined.

### tfl.batch_to_space_nd

BatchToSpaceNd operator.

**TensorFlow Lite Dialect**

```
%output = tfl.batch_to_space_nd(%input, %block_shape, %indices)
```

**TOSA Lowering**

```
%result = convert_batch_to_space_nd_op(%input, %block_shape, %indices)
```

### tfl.cast

This operator is trivially lowered to tosa.CAST

### tfl.ceil

Ceil operator.

**TensorFlow Lite Dialect**

```
%y = tfl.ceil(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%result = tosa.CEIL(%x)
```

### tfl.concatenation

Concatenation operator.

**TensorFlow Lite Dialect**

```
%output = tfl.concatenation(%values) {axis}
```

**TOSA Lowering**

```
%result = lower_concatv2_op(%values, axis)
```

### tfl.pseudo_const

This operator is trivially lowered to tosa.CONST

### tfl.conv_2d

Convolution operator.

**TensorFlow Lite Dialect**

```
%output = tfl.conv_2d(%input, %filter, %bias) {dilation_h_factor, dilation_w_factor, fused_activation_function, padding, stride_h, stride_w}
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%conv2d = tosa.CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%conv2d, fused_activation_function)
}
else {
    %result = %conv2d
}
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 output_rescale_scale = (%input.scale * %filter.scale) / %output.scale

tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%conv2d = tosa.CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}, quantization_info={input_zp=%input.zp, weight_zp=%filter.zp}}
%rescale = tosa.RESCALE(%conv2d) {scale=output_rescale_scale, input_zp=0, output_zp=%output.zp} // %conv2d.dtype->%output.dtype
if(fused_activation != NONE) {
    %result = convert_fused_activation(%rescale, fused_activation_function)
}
else {
    %result = %rescale
}
```

### tfl.convolution_2d_transpose_bias

No TOSA lowering defined.

### tfl.cos

No TOSA lowering defined.

### tfl.densify

No TOSA lowering defined.

### tfl.depth_to_space

DepthToSpace operator.

**TensorFlow Dialect**

```
%output = tfl.depth_to_space(%input) {block_size}
```

**TOSA Lowering**

```
%output = lower_depth_to_space_op(%input, block_size, "NHWC")
```

### tfl.depthwise_conv_2d

Depthwise-separable convolution operator.

**TensorFlow Lite Dialect**

```
%output = tfl.depthwise_conv_2d(%input, %filter, %bias) {dilation_h_factor, dilation_w_factor, fused_activation_function, padding, stride_h, stride_w, depth_multiplier}
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%depthwise_conv2d = tosa.DEPTHWISE_CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%depthwise_conv2d, fused_activation_function)
}
else {
    %result = %depthwise_conv2d
}
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 output_rescale_scale = (%input.scale * %filter.scale) / %output.scale

tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%depthwise_conv2d = tosa.DEPTHWISE_CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}, quantization_info={input_zp=%input.zp, weight_zp=%filter.zp}}
%rescale = tosa.RESCALE(%conv2d) {scale=output_rescale_scale, input_zp=0, output_zp=%output.zp} // %depthwise_conv2d.dtype->%output.dtype
if(fused_activation != NONE) {
    %result = convert_fused_activation(%rescale, fused_activation_function)
}
else {
    %result = %rescale
}
```

### tfl.dequantize

Dequantize operator.

**TensorFlow Lite Dialect**

```
%output = tfl.dequantize(%input)
```

**TOSA Lowering**

```
%result = lower_dequantize_op(%input, %input.scale, %input.zp)
```

### tfl.div

Division operator.

**TensorFlow Lite Dialect**

```
%output = tfl.div(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%rcp = tosa.RECIPROCAL(%rhs)
%mul = tosa.MUL(%lhs, %rcp)
```

### tfl.elu

Exponential Linear Unit operator.

**TensorFlow Lite Dialect**

```
%y = tfl.elu(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%rcp = lower_elu_op(%x)
```

### tfl.embedding_lookup

Embedding lookup operator.

**TensorFlow Lite Dialect**

```
%output = tfl.embedding_lookup(%lookup, %value)
```

### tfl.equal

This operator is trivially lowered to tosa.EQUAL

### tfl.exp

Natural exponentiation operator.

**TensorFlow Lite Dialect**

```
%y = tfl.exp(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%result = tosa.EXP(%x)
```

### tfl.expand_dims

Inserts a dimension of 1 into a tensor’s shape.

**TensorFlow Lite Dialect**

```
%output = tfl.expand_dims(%input, %dim)
```

**TOSA Lowering**

```
%result = lower_expand_dims(%input, %dim.as_constant())
```

### tfl.external_const

No TOSA lowering defined.

### tfl.fake_quant

FakeQuant operator

**TensorFlow Lite Dialect**

```
%output = tfl.fake_quant(%input) {min, max, num_bits, narrow_range}
```

**TOSA Lowering**

```
%result = convert_fake_quant_op(%input, min, max, num_bits, narrow_range)
```

### tfl.fill

Fill the tensor with given value.

**TensorFlow Lite Dialect**

```
%res = tfl.fill(%dims, %value)
```

**TOSA Lowering**

Prepare:

```
total_size = 1
dim_vec = %dim.as_constant()
for(int32 i = 0 i < dim_vec.size() i++) {
    total_size *= dim_vec[i]
}
filled_val = %value.as_constant()[0]
output_type = tensor<dim_vec, filled_val.dtype>
```

Legalization:

```
%result = tosa.CONST() {value={filled_val} * total_size}
```

### tfl.floor_div

Floor div operator.

**TensorFlow Lite Dialect**

```
%output = tfl.floor_div(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%recip = tosa.RECIPROCAL(%rhs)
%mul = tosa.MUL(%lhs, %recip)
%result = tosa.FLOOR(%mul)
```

### tfl.floor_mod

Division remainder.

**TensorFlow Lite Dialect**

```
%output = tfl.floor_mod(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%recip = tosa.RECIPROCAL(%rhs)
%mul = tosa.MUL(%lhs, %recip)
%floor = tosa.FLOOR(%mul)
%result = tosa.SUB(%mul, %floor)
```

### tfl.floor

This operator is trivially lowered to tosa.FLOOR

### tfl.fully_connected

Fully connected op.

**TensorFlow Lite Dialect**

```
%output = tfl.fully_connected(%input, %filter, %bias) {fused_activation_function}
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Prepare:

```
// input[N, IC] x filter[OC, IC] + bias[OC] -> output[N, OC]
auto input_reshape_shape = {%input.num_elements / %filter.shape[1], %filter.shape[1]}
```

Legalization:

```
if(!(%bias)) {
    %bias_val = tosa.CONST() {value={0} * %filter.shape[3]}
}
else {
    %bias_val = %bias
}
if(%input.rank != 2) {
    %input_val = tosa.RESHAPE(%input) {shape=input_reshape_shape}
}
else {
    %input_val = %input
}
%fc = tosa.FULLY_CONNECTED(%input_val, %filter, %bias_val)
if(fused_activation != NONE) {
    %result = convert_fused_activation(%fc, fused_activation_function)
}
else {
    %result = %fc
}
```

If input/output tensors are all quantized typed,

Prepare:

```
auto input_reshape_shape = {%input.num_elements / %filter.shape[1], %filter.shape[1]}
float64 output_rescale_scale = (%input.scale * %filter.scale) / %output.scale
```

Legalization:

```
if(!(%bias)) {
    %bias_val = tosa.CONST() {value={0} * %filter.shape[3]}
}
else {
    %bias_val = %bias
}
if(%input.rank != 2) {
    %input_val = tosa.RESHAPE(%input) {shape=input_reshape_shape}
}
else {
    %input_val = %input
}
%fc = tosa.FULLY_CONNECTED(%input_val, %filter, %bias_val)
%rescale = tosa.RESCALE(%fc) {scale=output_rescale_scale, input_zp=0, output_zp=%output.zp} // %fc.dtype->%output.dtype
if(fused_activation != NONE) {
    %result = convert_fused_activation(%rescale, fused_activation_function)
}
else {
    %result = %rescale
}
```

### tfl.gather_nd

Gather_nd operator.

**TensorFlow Dialect**

```
%output = tfl.gather_nd(%params, %indices)
```

**TOSA Lowering**

```
%output = lower_gather_nd_op(%params, %indices)
```

### tfl.gather

Gather operator.

**TensorFlow Dialect**

```
%output = tfl.gather(%params, %indices) {axis}
```

**TOSA Lowering**

```
%output = lower_gather_op(%params, %indices, 0, axis)
```

### tfl.greater_equal

This operator is trivially lowered to tosa.GREATER_EQUAL

### tfl.greater

This operator is trivially lowered to tosa.GREATER

### tfl.hard_swish

Hardswish activation function.

**TensorFlow Lite Dialect**

```
%output = tfl.hard_swish(%input)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

```
%const_3 = tosa.CONST() {value={3.0}}
%const_rcp6 = tosa.CONST() {value={1.0 / 6.0}}
%op1_add_in_3 = tosa.ADD(%input, %const_3)
%op2_relun_op1 = tosa.RELUN(%op1_add_in_3) {max=6.0}
%op3_mul_in_op2 = tosa.MUL(%input, %op2_relun_op1)
%op4_mul_op3_rcp6 = tosa.MUL(%op3, %const_rcp6)
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 input_sample_grain = 1.0 / 64.0
auto hardswish_func = [input_sample_grain](int32 x) -> int32 {
    float64 v = (float64)x * input_sample_grain
    float64 w = v + 3.0
    w = (w < 0.0) ? 0.0 : ((w > 6.0) ? 6.0 : w)
    v = (v * w) / 6.0
    return std::lround(32768.0 * v)
}
float64 input_rescale_scale = (%input.scale * 128.0) / input_sample_grain
float64 output_rescale_scale = 1.0 / (128.0 * 32768.0 * %output.scale)
int32 quantized_3 = (int32)(std::ceil(3.0 / %input.scale)) + %input.zp
```

Legalization:

```
%table_const = get_table_const_tensor(hardswish_func)
%const_3 = tosa.CONST() {value={quantized_3}}
%op1_rescale_in = tosa.RESCALE(%input) {scale=input_rescale_scale, input_zp=%input.zp, output_zp=0} // %input.dtype->i16
%op2_table_op1 = tosa.TABLE(%op1_rescale_in, %table_const)
%op3_rescale_op2 = tosa.RESCALE(%op2_table_op1) {scale=output_rescale_scale, input_zp=0, output_zp=%output.zp} // i32->%output.dtype
%op4_rescale_in = tosa.RESCALE(%input {scale=1.0, input_zp=0, output_zp=0} // %input.dtype->i32
%op5_ge_op4 = tosa.GREATER_EQUAL(%op4_rescale_in, %const_3)
%op6_select_op5_in_op3 = tosa.SELECT(%op5_ge_op4, %input, %op3_rescale_op2)
```

### tfl.l2_normalization

No TOSA lowering defined.

### tfl.lstm

No TOSA lowering defined.

### tfl.leaky_relu

Leaky Relu Operator.

**TensorFlow Lite Dialect**

```
%output = tfl.leaky_relu(%input) {alpha}
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%const_0 = tosa.CONST() {value={0.0}}
%const_alpha = tosa.CONST() {value={alpha}}
%op1_mul_in_alpha = tosa.MUL(%input, %const_alpha)
%op2_ge_in_0 = tosa.GREATER_EQUAL(%input, %const_0)
%op3_select_op2_in_op1 = tosa.SELECT(%op2_ge_in_0, %input, $op1_mul_in_alpha)
```

If input/output tensors are all quantized typed,

Prepare:

```
float32 scaled_alpha = (%input.scale * alpha) / %output.scale
float32 scaled_identity = %input.scale / %output.scale
```

Legalization:

```
%const_0 = tosa.CONST() {value={0}}
%op1_rescale_in = tosa.RESCALE(%input) {scale=1.0, input_zp=%input.zp} // %input.dtype->i32
%op2_ge_in_0 = tosa.GREATER_EQUAL(%input, %const_0)
%op3_rescale_in_alpha = tosa.RESCALE(%input) {scale=scaled_alpha, input_zp=%input.zp, output_zp=%output_zp} // %input.dtype->%output.dtype
%op4_rescale_in_identity = tosa.RESCALE(%input) {scale=scaled_identity, input_zp=%input.zp, output_zp=%output_zp} // %input.dtype->%output.dtype
%op5_select_op2_op3_op4 = tosa.SELECT(%op2_ge_in_0, %op4_rescale_in_identity, %op3_rescale_in_alpha)
```

### tfl.less_equal

Less_equal operator.

**TensorFlow Lite Dialect**

```
%output = tfl.less_equal(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_greater_lhs_rhs = tosa.GREATER(%lhs, %rhs)
%op2_not_op1 = tosa.LOGICAL_NOT(%op1_greater_lhs_rhs)
```

If input/output tensors are all quantized typed,

Legalization:

```
assert (%lhs.scale == %rhs.scale) && (%lhs.zp == %rhs.zp)

%op1_rescale_lhs = tosa.RESCALE(%lhs) {scale=1.0, input_zp=%lhs.zp, output_zp=0} // %lhs.dtype->i32
%op2_rescale_rhs = tosa.RESCALE(%rhs) {scale=1.0, input_zp=%rhs.zp, output_zp=0} // %rhs.dtype->i32
%op3_greater_op1_op2 = tosa.GREATER(%op1_rescale_lhs, %op2_rescale_rhs)
%op4_not_op3 = tosa.LOGICAL_NOT(%op3_greater_op1_op2)
```

### tfl.less

Less operator.

**TensorFlow Lite Dialect**

```
%output = tfl.less(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_ge_lhs_rhs = tosa.GREATER_EQUAL(%lhs, %rhs)
%op2_not_op1 = tosa.LOGICAL_NOT(%op1_ge_lhs_rhs)
```

If input/output tensors are all quantized typed,

Legalization:

```
assert (%lhs.scale == %rhs.scale) && (%lhs.zp == %rhs.zp)

%op1_rescale_lhs = tosa.RESCALE(%lhs) {scale=1.0, input_zp=%lhs.zp, output_zp=0} // %lhs.dtype->i32
%op2_rescale_rhs = tosa.RESCALE(%rhs) {scale=1.0, input_zp=%rhs.zp, output_zp=0} // %rhs.dtype->i32
%op3_ge_op1_op2 = tosa.GREATER_EQUAL(%op1_rescale_lhs, %op2_rescale_rhs)
%op4_not_op3 = tosa.LOGICAL_NOT(%op3_ge_op1_op2)
```

### tfl.local_response_normalization

No TOSA lowering defined.

### tfl.log

No TOSA lowering defined.

### tfl.log_softmax

Log softmax operator.

**TensorFlow Lite Dialect**

```
%output = tfl.log_softmax(%input)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%output = lower_log_softmax_op(%logits)
```

No TOSA lowering defined if input/output tensors are all quantized typed.

### tfl.logical_and

This operator is trivially lowered to tosa.LOGICAL_AND

### tfl.logical_not

This operator is trivially lowered to tosa.LOGICAL_NOT

### tfl.logical_or

This operator is trivially lowered to tosa.LOGICAL_OR

### tfl.logistic

Logistic operator.

**TensorFlow Lite Dialect**

```
%y = tfl.logistic(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_sigmoid_in = tosa.SIGMOID(%x)
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 input_sample_grain = 1.0 / 16.0
auto sigmoid_func = [input_sample_grain](int32 x) -> int32 {
  float64 v = static_cast<float64>(x) * input_sample_grain
  v = 1.0 / (1.0 + std::exp(-v))
  return std::lround(32768.0 * v)
}

float32 input_rescale_scale = (%x.scale * 128.0) / input_sample_grain
float32 output_rescale_scale = 1.0 / (%y.scale * 32768.0 * 128.0);
```

Legalization:

```
%table_const = get_table_const_tensor(sigmoid_func)
%op1_rescale_in = tosa.RESCALE(%x) {scale=input_rescale_scale, input_zp=%x.zp, output_zp=0} // %x.dtype->i16
%op2_table_op1 = tosa.TABLE(%op1_rescale_in, %table_const)
%op3_rescale_op2 = tosa.RESCALE(%op2_table_op1) {scale=output_rescale_scale, input_zp=0, output_zp=%y.zp} // %int32->%y.dtype
```

### tfl.matrix_diag

No TOSA lowering defined.

### tfl.matrix_set_diag

No TOSA lowering defined.

### tfl.max_pool_2d

Max Pool 2d op.

**TensorFlow Lite Dialect**

```
%output = tfl.max_pool_2d(%input) {filter_height, filter_width, padding, stride_h, stride_w, fused_activation_function}
```

**TOSA Lowering**

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, tensor<{filter_height, filter_width}, tosa.int32>,
                                      {1, stride_h, stride_w, 1}, {1, 1, 1, 1})
```

If input/output tensors are all non-quantized typed,

Legalization:

```
%maxpool2d = tosa.MAX_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%maxpool2d, fused_activation)
}
else {
    %result = %maxpool2d
}
```

If input/output tensors are all quantized typed,

Legalization:

```
%maxpool2d = tosa.MAX_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding, quantization_info={input_zp=%input.zp, output_zp=%output.zp}}
if(fused_activation != NONE) {
    %result = convert_fused_activation(%maxpool2d, fused_activation)
}
else {
    %result = %maxpool2d
}
```

### tfl.max_pooling_with_argmax_2d

No TOSA lowering defined.

### tfl.max_unpooling_2d

No TOSA lowering defined.

### tfl.maximum

This operator is trivially lowered to tosa.MAXIMUM

### tfl.mean

Mean operator.

**TensorFlow Lite Dialect**

```
%output = tfl.mean(%input, %axis) {keep_dims}
```

**TOSA Lowering**

Prepare:

```
int32 num_elements_on_axis = 1
for (int32 axis : %reduction_indices) {
    num_elements_on_axis *= %input.shape[axis]
}
float32 div_scale = 1.0 / num_elements_on_axis
```

If input/output tensors are all non-quantized typed,

Legalization:

```
%cst_div_scale = tosa.CONST() {value={div_scale}}
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_SUM>(%input, %output.shape, %axis, keep_dims)
%op2_mul_op1 = tosa.MUL(%op1_rsum_in, %cst_div_scale)
```

If input/output tensors are all quantized typed,

Legalization:

```
%rsum = lower_reduce_op<tosa.REDUCE_SUM>(%op1_rescale_in, %output.shape, %reduction_indices, keep_dims, 1.0f, %input_zp, div_scale * %input.scale / %output.scale, %output.zp)
```

### tfl.minimum

This operator is trivially lowered to tosa.MINIMUM

### tfl.mirror_pad

No TOSA lowering defined.

### tfl.mul

Mul operator.

**TensorFlow Lite Dialect**

```
%output = tfl.mul(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_mul_in = tosa.MUL(%lhs, %rhs)
```

If input/output tensors are all quantized typed,

Legalization:

```
%op1_rescale_lhs = tosa.RESCALE(%lhs) {scale=1.0f, input_zp=%lhs.zp, output_zp=0} // %lhs.dtype->i32
%op2_rescale_rhs = tosa.RESCALE(%rhs) {scale=1.0f, input_zp=%rhs.zp, output_zp=0} // %rhs.dtype->i32
%op3_mul_op1_op2 = tosa.MUL(%op1_rescale_lhs, %op2_rescale_rhs)
%op4_rescale_op3 = tosa.RESCALE(%op3_mul_op1_op2) {scale=%lhs.scale * %rhs.scale / %output.scale, input_zp=0, output_zp=%output.zp} // i32->%output.dtype
```

### tfl.neg

This operator is trivially lowered to tosa.NEGATE

### tfl.non_max_suppression_v4

No TOSA lowering defined.

### tfl.non_max_suppression_v5

No TOSA lowering defined.

### tfl.not_equal

Not_equal operator.

**TensorFlow Lite Dialect**

```
%output = tfl.not_equal(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_equal_lhs_rhs = tosa.EQUAL(%lhs, %rhs)
%op2_not_op1 = tosa.LOGICAL_NOT(%op1_equal_lhs_rhs)
```

If input/output tensors are all quantized typed,

Legalization:

```
assert (%lhs.scale == %rhs.scale) && (%lhs.zp == %rhs.zp)

%op1_rescale_lhs = tosa.RESCALE(%lhs) {scale=1.0f, input_zp=%lhs.zp, output_zp=0} // %lhs.dtype->i32
%op2_rescale_rhs = tosa.RESCALE(%rhs) {scale=1.0f, input_zp=%rhs.zp, output_zp=0} // %rhs.dtype->i32
%op3_equal_op1_op2 = tosa.EQUAL(%op1_rescale_lhs, %op2_rescale_rhs)
%op4_not_op3 = tosa.LOGICAL_NOT(%op3_equal_op1_op2) // i32->%output.dtype
```

### tfl.NumericVerify

No TOSA lowering defined.

### tfl.one_hot

OneHot operator.

**TensorFlow Lite Dialect**

```
%output = tfl.one_hot(%indices, %depth, %on_value, %off_value) {axis}
```

**TOSA Lowering**

```
%output = lower_one_hot_op(%indices, %depth, %on_value, %off_value, axis)
```

### tfl.prelu

No TOSA lowering defined.

### tfl.pack

Packs a list of tensors along a dimension into one tensor.

**TensorFlow Dialect**

```
%output = tf.pack(%values) {axis}
```

**TOSA Lowering**

```
%output = lower_pack_op(%values, axis)
```

### tfl.pad

This operator is trivially lowered to tosa.PAD

### tfl.padv2

No TOSA lowering defined.

### tfl.pow

No TOSA lowering defined.

### tfl.pseudo_qconst

This operator is trivially lowered to tosa.CONST

### tfl.quantize

Quantize operator

**TensorFlow Lite Dialect**

```
%output = tfl.quantize(%input)
```

**TOSA Lowering**

Legalization:

```
if (isa<QuantizedType>(%input.dtype)) {
    %op1_rescale_in = tosa.RESCALE(%input) {scale=%input.scale / %output.scale, input_zp=%input.zp, output_zp=%output.zp}
}
else {
    %output = lower_quantize_op(%output.dtype, %input, %output.zp, %output.scale)
}
```

### tfl.range

No TOSA lowering defined.

### tfl.rank

Rank operator

**TensorFlow Lite Dialect**

```
%output = tfl.rank(%input)
```

**TOSA Lowering**

Legalization:

```
%const = tosa.CONST() {value={%input.rank}}
```

### tfl.reduce_any

Computes the "logical or" of elements across dimensions of a tensor.

**TensorFlow Lite Dialect**

```
%output = tfl.reduce_any(%input, %reduction_indices) {keep_dims}
```

**TOSA Lowering**

Legalization:

```
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_ANY>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tfl.reduce_max

Max-reduction operator.

**TensorFlow Lite Dialect**

```
%output = tfl.reduce_max(%input, %axes) {keep_dims}
```

**TOSA Lowering**

Legalization:

```
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_MAX>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tfl.reduce_min

Computes the min reduction along the specified axes.

**TensorFlow Lite Dialect**

```
%output = tfl.reduce_min(%input, %axes) {keep_dims}
```

**TOSA Lowering**

Legalization:

```
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_MIN>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tfl.reduce_prod

Prod-reduction operator.

**TensorFlow Lite Dialect**

```
%output = tfl.reduce_prod(%input, %axes) {keep_dims}
```

**TOSA Lowering**

If input/output tensors are all float typed,

Legalization:

```
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_PROD>(%input, %output.shape, %reduction_indices, keep_dims)
```

### tfl.relu_n1_to_1

No TOSA lowering defined.

### tfl.relu6

Relu6 operator.

**TensorFlow Lite Dialect**

```
%y = tfl.relu6(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_relun_in = tosa.RELUN(%input) {max_int=0, max_fp=6.0}
```

If input/output tensors are all quantized typed,

Legalization:

```
%op1_rescale_in = tosa.RESCALE(%lhs) {scale=%x.scale / %y.scale, input_zp=%x.zp, output_zp=0} // %x.dtype->i32
%op2_relun_op1 = tosa.RELUN(%op1_rescale_in) {max_int=(6.0 / %y.scale), max_fp=0.0}
%op3_rescale_op2 = tosa.RESCALE(%op2_relun_op1) {scale=1.0, input_zp=0, output_zp=%y.zp // i32->%y.dtype
```

### tfl.relu

Relu operator.

**TensorFlow Lite Dialect**

```
%y = tfl.relu(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_relun_in = tosa.RELUN(%input) {max_int=0, max_fp=std::numeric_limits<float>::max()}
```

If input/output tensors are all quantized typed,

Legalization:

```
%op1_rescale_in = tosa.RESCALE(%lhs) {scale=%x.scale / %y.scale, input_zp=%x.zp, output_zp=0} // %x.dtype->i32
%op2_relun_op1 = tosa.RELUN(%op1_rescale_in) {max_int=std::numeric_limits<int32>::max(), max_fp=0.0}
%op3_rescale_op2 = tosa.RESCALE(%op2_relun_op1) {scale=1.0, input_zp=0, output_zp=%y.zp // i32->%y.dtype
```

### tfl.reshape

This operator is trivially lowered to tosa.RESHAPE

### tfl.resize_bilinear

ResizeBilinear Op.

**TensorFlow Lite Dialect**

```
%output = tfl.resize_bilinear(%input, %size) {aligned_corners, half_pixel_centers}
```

**TOSA Lowering**

```
%output = lower_resize_op(%input, %size, %input.dtype, "BILINEAR")
```

### tfl.resize_nearest_neighbor

ResizeBilinear Op.

**TensorFlow Lite Dialect**

```
%output = tfl.resize_bilinear(%input, %size) {aligned_corners, half_pixel_centers}
```

**TOSA Lowering**

```
%output = lower_resize_op(%input, %size, %input.dtype, "NEAREST_NEIGHBOR")
```

### tfl.reverse_sequence

No TOSA lowering defined.

### tfl.reverse_v2

ReverseV2 Operator.

**TensorFlow Lite Dialect**

```
%output = tfl.reverse_v2(%input, %axis)
```

**TOSA Lowering**

```
%output = lower_reversev2_op(%tensor, %axis)
```

### tfl.round

Round operator.

**TensorFlow Lite Dialect**

```
%output = tfl.round(%input)
```

**TOSA Lowering**

```
%const_half = tosa.CONST() {value={0.5}}
%op1_add_in_half = tosa.ADD(%input, %const_half)
%op2_floor_op1 = tosa.FLOOR(%op1_add_in_half)
```

### tfl.rsqrt

No TOSA lowering defined.

### tfl.svdf

No TOSA lowering defined.

### tfl.segment_sum

No TOSA lowering defined.

### tfl.select

This operator is trivially lowered to tosa.SELECT

### tfl.select_v2

This operator is trivially lowered to tosa.SELECT

### tfl.shape

Shape operator

**TensorFlow Lite Dialect**

```
%output = tfl.shape(%input)
```

**TOSA Lowering**

Legalization:

```
%const = tosa.CONST() {value=%input.shape}
```

### tfl.sin

No TOSA lowering defined.

### tfl.slice

This operator is trivially lowered to tosa.SLICE

### tfl.softmax

Softmax operator.

**TensorFlow Lite Dialect**

```
%output = tfl.softmax(%input)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_exp_in = tosa.EXP(%input)
%op2_rsum_op1 = tosa.REDUCE_SUM(%op1_exp_in) {axis=(%input.rank-1)}
%op3_rcp_op2 = tosa.RECIPROCAL(%op2)
%op4_mul_op1_op3 = tosa.MUL(%op1, %op3)
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 exp_sample_grain = 1.0 / 16.0
auto exp_func = [exp_sample_grain](int32 x) -> int32 {
  double v = static_cast<float64>(x) * exp_sample_grain
  v = v < 0.0 ? std::exp(v) : 1.0
  return std::lround(32768.0 * v)
}

float64 one_over_one_plus_x_sample_grain = 1.0 / 256.0
auto one_over_one_plus_x_func = [one_over_one_plus_x_sample_grain](int32 x) -> int32 {
  double v = static_cast<float64>(x) * one_over_one_plus_x_sample_grain
  v = v < 0.0 ? 1.0 : 1.0 / (1.0 + v)
  return std::lround(32768.0 * v)
}

float64 op4_rescale_scale = (%input.scale * 128.0) / exp_sample_grain
float64 op19_rescale_scale = 1.0 / (%output.scale * 256.0)
```

Legalization:

```
%const_exp_table = get_table_const_tensor(exp_func)
%const_one_over_one_plus_x_table = get_table_const_tensor(one_over_one_plus_x_func)
%const_3 = tosa.CONST() {value={3}}
%const_34 = tosa.CONST() {value={12+20-8}}
%const_2_to_31 = tosa.CONST() {value={1<<31}}
%const_16 = tosa.CONST() {value={16}}

%op1_rescale_in = tosa.RESCALE(%lhs) {scale=1.0f, input_zp=%x.zp, output_zp=0} // %x.dtype->i32
%op2_rmax_op1 = tosa.REDUCE_MAX(%op1_rescale_in) {axis=(%input.rank-1)}
%op3_sub_op1_op2 = tosa.SUB(%op1_rescale_in, %op2_relun_op1)
%op4_rescale_op3 = tosa.RESCALE(%op3_sub_op1_op2) {scale=op4_rescale_scale, input_zp=0, output_zp=0} // i32->i16
%op5_table_op4 = tosa.TABLE(%op4_rescale_op3, %const_exp_table)
%op6_rshift_op5_3 = tosa.ARITHMETIC_RIGHT_SHIFT(%op5_table_op4, %const_3)
%op7_rsum_op6 = tosa.REDUCE_SUM(%op6_rshift_op5_3) {axis=(%input.rank-1)}
%op8_clz_op7 = tosa.CLZ(%op7_rsum_op6)
%op9_sub_34_op8 = tosa.SUB(%const_34, %op8_clz_op7)
%op10_lshift_op7_op8 = tosa.LOGICAL_LEFT_SHIFT(%op7_rsum_op6, %op8_clz_op7)
%op11_sub_op10 = tosa.SUB(%op10_lshift_op7_op8, %const_2_to_31)
%op12_rshift_op11_16 = tosa.ARITHMETIC_RIGHT_SHIFT(%op11_sub_op10, %const_16)
%op13_cast_op12 = tosa.CAST(%op12_rshift_op11_16) // i32->i16
%op14_table_op13 = tosa.TABLE(%op13_cast_op12, %const_one_over_one_plus_x_table)
%op15_rescale_op14 = tosa.RESCALE(%op14_table_op13) {scale=1.0/128.0, input_zp=0, output_zp=0} // i32->i16
%op16_rescale_op5 = tosa.RESCALE(%op5_table_op4) {scale=1.0/128.0, input_zp=0, output_zp=0} // i32->i16
%op17_mul_op16_op15 = tosa.MUL(%op15_rescale_op14, %op16_rescale_op5)
%op18_rshift_op17_op9 = tosa.ARITHMETIC_RIGHT_SHIFT(%op17_mul_op16_op15, %op9_sub_34_op8)
%op19_rescale_op18 = tosa.RESCALE(%op18_rshift_op17_op9) {scale=op19_rescale_scale, input_zp=0, output_zp=%output.zp}
```

### tfl.space_to_batch_nd

SpaceToBatchNd operator.

**TensorFlow Dialect**

```
%output = tfl.space_to_batch_nd(%input, %block_shape, %paddings)
```

**TOSA Lowering**

```
%output = lower_space_to_batch_nd_op(%input, %block_shape, %paddings)
```

### tfl.space_to_depth

SpaceToDepth operator.

**TensorFlow Dialect**

```
%output = tfl.space_to_depth(%input) {block_size}
```

**TOSA Lowering**

```
%output = lower_space_to_depth_op(%input, block_size, "NHWC")
```

### tfl.pseudo_sparse_const

No TOSA lowering defined.

### tfl.pseudo_sparse_qconst

No TOSA lowering defined.

### tfl.sparse_to_dense

No TOSA lowering defined.

### tfl.split

Splits a tensor into num_split tensors along one dimension.

**TensorFlow Dialect**

```
%output = tfl.split(%split_dim, %value) {num_split}
```

**TOSA Lowering**

```
%output = lower_split_op(%value, %split_dim.as_constant(), num_split)
```

### tfl.split_v

Splits a tensor into num_split tensors along one dimension.

**TensorFlow Dialect**

```
%output = tfl.split_v(%value, %size_splits, %split_dim) {num_splits}
```

**TOSA Lowering**

```
%output = lower_splitv_op(%value, %size_splits.as_constant(), %split_dim.as_constant())
```

### tfl.sqrt

No TOSA lowering defined.

### tfl.square

Square operator.

**TensorFlow Lite Dialect**

```
%y = tfl.square(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_mul_in = tosa.MUL(%x, %x)
```

If input/output tensors are all quantized typed,

Legalization:

```
%op1_rescale_x = tosa.RESCALE(%x) {scale=1.0f, input_zp=%x.zp, output_zp=0} // %x.dtype->i32
%op2_mul_op1_op1 = tosa.MUL(%op1_rescale_x, %op1_rescale_x)
%op3_rescale_op2 = tosa.RESCALE(%op2_mul_op1_op1) {scale=%(x.scale * %x.scale) / %output.scale, input_zp=0, output_zp=%y.zp} // i32->%y.dtype
```

### tfl.squared_difference

Squared difference operator.

**TensorFlow Lite Dialect**

```
%output = tfl.squared_difference(%lhs, %rhs)
```

**TOSA Lowering**

Legalization:

```
%op1_sub_in = tosa.SUB(%lhs, %rhs)
%op2_mul_op1 = tosa.MUL(%op1_sub_in, %op1_sub_in)
```

### tfl.squeeze

Removes dimensions of size 1 from the shape of a tensor.

**TensorFlow Dialect**

```
%output = tfl.squeeze(%input) {squeeze_dims}
```

**TOSA Lowering**

```
%output = lower_squeeze_op(%input, squeeze_dims)
```

### tfl.strided_slice

StridedSlice Op.

**TensorFlow Dialect**

```
%output = tfl.strided_slice(%input, %begin, %end, %strides) {begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask}
```

**TOSA Lowering**

```
%output = lower_strided_slice_op(%input, %begin, %end, %strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
```

### tfl.sub

This operator is trivially lowered to tosa.SUB

### tfl.sum

Sum operator.

**TensorFlow Lite Dialect**

```
%output = tfl.sum(%input, %axis) {keep_dims}
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_rsum_in = lower_reduce_op<tosa.REDUCE_SUM>(%input, %output.shape, %axis, keep_dims)
```

If input/output tensors are all quantized typed,

Legalization:

```
%rsum = lower_reduce_op<tosa.REDUCE_SUM>(%op1_rescale_in, %output.shape, %reduction_indices, keep_dims, 1.0f, %input_zp, (%input.scale / %output.scale), %output.zp)
```

### tfl.tanh

Hyperbolic tangent operator.

**TensorFlow Lite Dialect**

```
%y = tfl.tanh(%x)
```

**TOSA Lowering**

If input/output tensors are all non-quantized typed,

Legalization:

```
%op1_tanh_in = tosa.TANH(%x)
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 input_sample_grain = 1.0 / 32.0
auto tanh_func = [input_sample_grain](int32 x) -> int32 {
  float64 v = static_cast<float64>(x) * input_sample_grain
  v = std::exp(-2.0 * v)
  v = (1.0 - v) / (1.0 + v)
  return std::lround(32768.0 * v)
}

float32 input_rescale_scale = (%x.scale * 128.0) / input_sample_grain
float32 output_rescale_scale = 1.0 / (%y.scale * 32768.0 * 128.0);
```

Legalization:

```
%table_const = get_table_const_tensor(tanh_func)
%op1_rescale_in = tosa.RESCALE(%x) {scale=input_rescale_scale, input_zp=%x.zp, output_zp=0} // %x.dtype->i16
%op2_table_op1 = tosa.TABLE(%op1_rescale_in, %table_const)
%op3_rescale_op2 = tosa.RESCALE(%op2_table_op1) {scale=output_rescale_scale, input_zp=0, output_zp=%y.zp} // %int32->%y.dtype
```

### tfl.tile

This operator is trivially lowered to tosa.TILE

### tfl.topk_v2

No TOSA lowering defined.

### tfl.transpose_conv

Transpose convolution operator.

**TensorFlow Lite Dialect**

```
%output = tfl.transpose_conv(%output_shape, %weights, %input) {padding, stride_h, stride_w}
```

**TOSA Lowering**

Prepare:

```
tosa_padding =
    get_transpose_conv2d_padding_values_from_pad_type(%input.type, %weights.type, %output_shape, padding, "NHWC", FORMAT_HWIO, {stride_h, stride_w}, {1, 1})
```

If input/output tensors are all non-quantized typed,

Legalization:

```
%bias = tosa.CONST() {value={0.0} * %output.shape[3]}
%conv2d = tosa.TRANSPOSE_CONV2D(%input, %weight, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={1, 1}}
```

If input/output tensors are all quantized typed,

Prepare:

```
float64 output_rescale_scale = (%input.scale * %weights.scale) / %output.scale
```

Legalization:

```
%bias = tosa.CONST() {value={0} * %output.shape[3]}
%conv2d = tosa.TRANSPOSE_CONV2D(%input, %weight, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={1, 1}}
%rescale = tosa.RESCALE(%conv2d) {scale=output_rescale_scale, input_zp=0, output_zp=%output.zp} // %conv2d.dtype->%output.dtype
```

### tfl.transpose

This operator is trivially lowered to tosa.TRANSPOSE

### tfl.unidirectional_sequence_lstm

No TOSA lowering defined.

### tfl.unidirectional_sequence_rnn

No TOSA lowering defined.

### tfl.unique

No TOSA lowering defined.

### tfl.unpack

Unpacks a tensor along a dimension into multiple tensors.

**TensorFlow Dialect**

```
%output = tfl.unpack(%input) {num, axis}
```

**TOSA Lowering**

```
%output = lower_unpack_op(%input, axis, num)
```

### tfl.where

No TOSA lowering defined.

### tfl.while

No TOSA lowering defined.

### tfl.yield

This operator is trivially lowered to tosa.YIELD

### tfl.zeros_like

ZerosLike operator.

**TensorFlow Dialect**

```
%output = tfl.zeros_like(%input)
```

**TOSA Lowering**

```
%output = tosa.CONST() {value={0} * %input.num_elements}
```

## fuse_tf_bias

Legalize (tf.Conv2D + tf.BiasAdd) to tosa.CONV2D. This is currently the only N:1
mapping in TOSA legalization.

From:

```
%conv2d = tf.Conv2D(%input, %filter) {...}
%bias_add = tf.BiasAdd(%conv2d, %bias)
```

To:

```
%conv2d = tosa.CONV2D(%input, %filter, %bias)
```

## convert_tfl_uint8

This pass does three things:

1.  Convert const from quantized uint8 to quantized int8, with value within
    remapped as well.
2.  If input placeholders is quantized uint8 typed, insert "tosa.RESCALE()
    {scale=1.0, input_zp=input_zp, output_zp=input_zp-128} // qu8->qi8" in
    between
3.  If output tensor is quantized uint8 typed, insert "tosa.RESCALE()
    {scale=1.0, input_zp=output_zp+128, output_zp=output_zp} // qi8->qu8" in
    between
