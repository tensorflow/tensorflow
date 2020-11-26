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
{&lt;attribute = …​} : (tensor&lt;input\_shape, input\_type&gt;, …​) →
tensor&lt;output\_shape, output\_type&gt;

### Value Attributes

For the purposes of brevity and clarity in this document, the pseudocode allows
the following notation on value attribute.

Shorthand           | Description
------------------- | ---------------------------------------------------
`%t.shape`          | Shape vector for the tensor
`%t.shape[i]`       | Size of dimension i for the tensor
`%t.rank`           | Rank of the tensor
`%t.dtype`          | Datatype of the tensor
`%t.dtype.scale`    | Quantized scaling parameter (double)
`%t.dtype.zp`       | Quantized zero-point (int64)
`%t.dtype.signed`   | Boolean indicating the type is signed
`%t.dtype.num_bits` | Number of bits in the datatype
`%t.num_elements`   | Number of elements in the tensor
`%t.type`           | Tuple of `tensor<%t.shape, %t.dtype>`
`%t.size`           | For tensor lists: the number of tensors in the list

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

### apply_rank_broadcast()

```
// Applies a TOSA-lowering broadcast to tensor 'a' with respect
// to sibling tensor 'b'.
//
// The resulting tensors will have matching ranks.  TOSA broadcast
// operators accept degenerate broadcasting (1 vs non-1)
Value apply_rank_broadcast(Value %a, Value %b) {

    if (%a.rank < %b.rank) {

       auto new_a_shape = [1] * %b.rank

       if (%a.rank <= 1) {
          new_a_shape[%b.rank - 1] = a.shape[0]
          %out = tosa.RESHAPE(%a, new_a_shape)
          return %out
       }

       // Working from the right on both tensors, try to match all of a's
       // dimensions to b
       int adim = %a.rank - 1
       for (int bdim = b.rank() - 1; bdim >= 0 && adim >= 0; bdim--) {
           if (%a.shape[adim] == %b.shape[bdim] ||
               %a.shape[adim] == 1 ||
               %b.shape[bdim] == 1) {
              new_a_shape[bdim] = a.shape[adim]
              adim--
        }

        assert(adim == -1)
        assert(product(a.shape) == product(new_a_shape))

        %out = tosa.RESHAPE(%a) {new_shape=new_a_shape} (tensor<%a.type>) -> tensor<new_a_shape, %a.dtype>
    } else {
        %out = %a
    }

    return %out;
}
```

### get_padding_values_from_explicit_pad_attr()

```
vector<int64_t> get_padding_values_from_explict_pad_attr(vector<int64_t> explicit_pad,
                                                         tensorflow::TensorFormat data_format_tf)
{
    int64_t pad_before, pad_after
    vector<int64_t> computed_paddings

    for (int i = 0; i < 2; i++) {
        int64_t dim = GetTensorSpatialDimIndex(4, data_format_tf, i)
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
vector<int64_t> get_padding_values_from_pad_type(tensorflow::Padding padding, tensorflow::TensorFormat data_format,
                                        uint32_t first_filter_spatial_dim, type input_type, type filter_type
                                        vector strides, vector dilations)
{
    assert(padding != tensorflow::Padding::EXPLICIT);

    vector<int64_t> computed_padding;

    // Padding over H and W dimensions
    for (int i = 0; i < 2; i++) {
        int ifm_dim = get_tensor_spatial_dim_index(4, data_format, i);

        int filter_dim = first_filter_spatial_dim + i;

        int dim_dilation = dilations[ifm_dim];
        int dim_stride   = strides[ifm_dim];

        int64_t op_size, pad_before_tf, pad_after_tf;

        tensorflow::GetWindowedOutputSizeVerboseV2(input_type.shape[ifm_dim], filter_type.shape[filter_dim],
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
int32_t positive_axis(int32_t axis, int32_t rank)
{
   if (axis < 0)
       axis += rank;

   return axis;
}
```

### compute_scale_32()

```
void compute_scale_32(double scale, int32_t& multiplier, int32_t& shift)
{
    /* Generates mantissa and shift values where mantissa is in [-1.0,-0.5] or
    [0.5, 1.0] such that
    multiplier = mantissa*2^shift */

    const double mantissa = std::frexp(scale, &shift);
    auto shifted_m = std::round(mantissa * (int64_t(1) << 31));

    assert(shifted_m <= (int64_t(1) << 31)); // can't be greater that 1.0
    if (shifted_m == (int64_t(1) << 31)) {
        shifted_m /= 2;
        shift++;
    }
    // TOSA expect right shift to be positive, and embed (1 << 31) into right
    // shift bits
    shift = (-shift) + 31;

    assert(shifted_m <= std::numeric_limits<int32_t>::max());

    multiplier = static_cast<int32_t>(shifted_m);

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

    for (int i = 0; i < crops_dim; i++) {
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

    for (int i = 0; i < %block.rank; i++) {
        a1_shape[i] = %block.shape[i]
    }

    a1_shape[%block.rank] = %input.shape.[0] / %block.num_elements

    for (int i = 1; i < %input.rank; i++) {
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
    for (int i = 0; i < %block.rank; i++) {
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
    for (int i = 0; i < %block.rank; i++) {
        a3_shape[i + 1] = %input.shape[i + 1] * %block.shape[i]
    }

    for (int i = 0; remaining_block_shape; i++) {
        a3_shape[1 + %block.rank + 1] = %input.shape[%block.rank + 1 + i]
    }

    // Step 4 Crop the start/end dimensions using slice
    vector <size_t> a4_begin(%input.rank), a4_size(%input.rank)

    for (int i = 0; i < %input.rank; i++) {
        if (i == 0 || i > crop_dims) {
           a4_begin[i] = 0
           a4_size[i] = output_shape[i]
        } else {
          a4_begin[i] = %crops[i-1].first
          a4_size[i] = crops[i - 1].first - crops[i - 1].second
        }
    }

    %a1_reshape = tosa.RESHAPE(%input) {new_shape=a1_shape} : (tensor<%input.type>) -> tensor<a1_shape, %input.dtype>
    %a2_transpose = tosa.TRANSPOSE(%a1_reshape) {perms=a2_perm} : (tensor<%a1_reshape.type>) -> tensor<%a2_transpose.type>
    %a3_reshape = tosa.RESHAPE(%a2_transpose) {new_shape=a3_shape} : (tensor<%a2_transpose.type>) -> tensor<a3_shape, %input.dtype>
    %output = tosa.SLICE(%a3_reshape) {begin=a4_begin, size=a4_size} : (tensor<%a3_reshape.type>) -> tensor<a4_size, %input.dtype>

    return %output
}
```

### lower_concatv2_op()

```
Value lower_concatv2_op(Value %values, int32_t axis)
{
    int32_t tosa_axis = positive_axis(axis)

    assert(%values.size >= 2)

    // Convert scalar inputs to a tensor
    if (%values:0.size == 0) {
       for (int i = 0; i < %values.size; i++) {
          %values:i = tosa.RESHAPE(%values:i) {new_shape=1} : (tensor<{}, %values:i.dtype>) -> tensor<{1}, %values:i>
       }
    }

    // TODO: rescale

    %concat_op = tosa.CONCAT(%values:0, %values:1) {axis=tosa_axis} : (tensor<%values:0.type>, tensor<%values:1.type>) -> tensor<%concat_op.type>

    for (int i = 2; i < %values.size; i++) {
        // TODO: rescale
        %concat_op = tosa.CONCAT(%concat_op, %values:i) {axis=tosa_axis} : (tensor<%concat_op.type>, tensor<%values:i.type>) -> tensor<%concat_op.type>
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

    %a2_reshape = tosa.RESHAPE(%input) {new_shape=a2_shape} : (tensor<%input.type>) -> tensor<a2_shape, %input.dtype)
    %a3_transpose = tosa.TRANSPOSE(%a2_reshape) {perms={0, 1, 3, 2, 4, 5}} : (tensor<%a2_reshape.type>) -> tensor<%a3_transpose.type>
    %output = tosa.RESHAPE(%a3_transpose) {new_shape=a4_shape} : (tensor<%a3_transpose.type>) -> tensor<a4_shape, %input.dtype>

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
    %one_const = tosa.CONST({1}) : () -> tensor<{1}, float>
    %zero_const = tosa.CONST({0}) : () -> tensor<{1}, float>

    vector bcast_shape
    for (int i = 0; i < %value.rank; i++) {
        bcast_shape.push_back(1)
    }

    %one_reshape = tosa.RESHAPE(%one_const) {new_shape=bcast_shape} : (tensor<%one_const.type>) -> tensor<%one_reshape.type>
    %zero_reshape = tosa.RESHAPE(%zero_const) {new_shape=bcast_shape} : (tensor<%zero_const.type>) -> tensor<%zero_reshape.type>

    %exp_in = tosa.EXP(%value) : (tensor<%value.type>) -> tensor<%exp_in.type>
    %sub = tosa.SUB(%exp_in, %one_reshape) : (tensor<%exp_in.type>, tensor<%one_reshape.type>) -> tensor<%sub.type>
    %ge  = tosa.GREATER_EQUAL(%value, %zero_reshape) : (tensor<%value.type>, tensor<%zero_reshape.type>) -> tensor<%value.shape, bool>
    %output = tosa.SELECT(%ge, %value, %sub) : (tensor<%ge.type>, tensor<%value.type>, %tensor<%sub.type>) -> tensor<%output.type>
    return %output
}
```

### lower_expand_dims()

```
Value lower_expand_dims(Value %input, int32_t axis)
{
    vector<size_t> reshape_dims

    if (axis < 0 || axis >= %input.rank) {
        // Insert at the end of the tensor
        axis += %input.rank
        for (int i = 0; i < input.rank; i++) {
           reshape_dims.push_back(%input.shape[i])
        }
    } else {
        for (int i= 0 ; i < %input.rank; i++) {
            if (i == axis) {
                reshape_dims.push_back(1)
            }
            reshape_dims.push_back(%input.shape[i])
        }
    }

    %output = tosa.RESHAPE(%input) {new_shape=reshape_dims} (tensor<%input.type>) -> tensor<%output.type>
    return %output
}
```

### lower_fake_quant_op()

```
Value lower_fake_quant_op(Value %inputs, type output_type, double min, double max,
                            int64_t num_bits, bool narrow_range)
{
    assert(num_bits == 8 || num_bits == 16)

    int64_t qmax = (1L << (num_bits - 1)) - 1;
    int64_t qmin = -(1L << (num_bits - 1))

    if (narrow_range) {
       qmin = qmin + 1
    }

    double scale = (max - min) / double(qmax - qmin)

    int64_t zeropoint = (int64_t)std::round((-min) / scale + double(qmin))

    %quantized = lower_quantized_op(%inputs.type, %inputs, 1.0 / scale, zeropoint)

    %dequantized = lower_dequantized_op(output_type, %quantized_op, scale, zeropoint)

    return %dequantized
}
```

### lower_floor_div()

```
Value lower_floor_div(Value %lhs, Value %rhs)
{
    %recip = tosa.RECIPROCAL(%rhs) : (tensor<%rhs.type>) -> tensor<%recip.type>
    %mul = tosa.MUL(%lhs, %recip) : (tensor<%lhs.type>, tensor<%recip.type>) -> tensor<%mul.type>
    %output = tosa.FLOOR(%mul) : (tensor<%mul.type>) -> tensor<%output.type>

    return %output
}
```

### lower_floor_mod()

```
Value lower_floor_mod(Value %lhs, Value %rhs)
{
    %recip = tosa.RECIPROCAL(%rhs) : (tensor<%rhs.type>) -> tensor<%recip.type>
    %mul = tosa.MUL(%lhs, %recip) : (tensor<%lhs.type>, tensor<%recip.type>) -> tensor<%mul.type>
    %floor = tosa.FLOOR(%mul) : (tensor<%mul.type>) -> tensor<%floor.type>
    %output = tosa.SUB(%mul, %floor) : (tensor<%mul.type>, tensor<%floor.type>) -> tensor<%output.type>
    return %output
}
```

### lower_quantized_op()

```
Value lower_quantized_op(type output_type, Value %inputs, double scale, int64_t zeropoint)
{
    // TODO: fill in this function
}
```

### lower_dequantized_op()

```
Value lower_dequantized_op(type output_type, Value %inputs, double scale, int64_t zeropoint)
{
    // TODO: fill in this function
}
```

### lower_log_softmax_op()

```
Value lower_log_softmax_op(Value %logits)
{
    %op1 = tosa.EXP(%logits) : (tensor<%logits.type>) -> tensor<%op1.type>
    %op2 = tosa.REDUCE_SUM(%logits) {axis=(%logits.rank-1)} : (tensor<%logits.type>) -> tensor<%op2.type>
    %op3 = tosa.RECIPROCAL(%op2) : (tensor<%op2.type>) -> tensor<%op3.type>
    %op4 = tosa.MUL(%op1, %op3) : (tensor<%op1.type>, tensor<%op3.type>) -> tensor<%op4.type>
    %op5 = tosa.LOG(%op4) : (tensor<%op4.type>) -> tensor<%op5.type>

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
       for (int i = 0; i < %input.size; i++) {
           %input[i] = tosa.RESHAPE(%input[i], {1})
       }
   }

   vector<size_t> output_shape
   for (int i = 0; i < input_tensor_rank; i++) {
       output_shape.push_back(%input[0].shape[i]
   }

   output_shape[concat_axis] = output_shape[concat_axis] * %input.size

   // First pair of tensors
   %concat = tosa.CONCAT(%input[0], %input[1]) {axis=concat_axis} : (tensor<%input[0].type>, tensor<%input[1].type>) -> tensor<%concat.type>

   // Remaining tensors
   for (int i = 2; i < %input.size; i++) {
      %concat = tosa.CONCAT(%concat, %input[i]) {axis=concat_axis} : (tensor<%concat.type>, tensor<%input[i].type>) -> tensor<%concat.type>
   }

   if (input_tensor_rank == 0) {
      // No reshape needed for rank 0, already done
      %output = %concat
   } else

      %reshape = tosa.RESHAPE(%concat) {new_shape=output_shape} : (tensor<%concat.type>) -> tensor<%reshape.type>

      if (concat_axis == input_tensor_rank) {
         // Output shape is [A, B, C, .. n] in this case,
         // need to reshape to [N, A, B, C, ..] with perm [1, 2, 3, .. 0]
         concat_axis = 0

         vector <size_t> perms
         for (int i = 0; i < %input[0].rank; i++)
            perms.push_back(i + 1)
         perms.push_back(0)

         %output = tosa.TRANSPOSE(%reshape) {perms=perms} : (tensor<%reshape.type>) -> tensor<%output.type>
     } else {
         %output = %reshape
     }

     return %output
}
```

### lower_reduce_op()

```
Value lower_reduce_op<tosa_op_t OP>(Value %input, shape_t output_shape, Value %axes, bool keep_dims)
{

    vector axes_vec = %axes.as_constant();

    // Special case of no axes means no transformation
    if (axes_vec.size() == 0) {
       return tosa.IDENTITY(%input) : (%input.type) -> %output.type
    }

    shape_t shape = %input.shape;
    %output = %input;

    // TODO: rescaling on quantized types
    for (int i = 0; i < axes_vec.size(); i++) {
        int32_t axis = positive_axis(axes_vec[i], %input.rank);

        shape[axis] = 1;
        %output = tosa.OP(%output) {axis=axis} : (tensor<%output.type>) -> tensor<shape, %output.dtype>
    }

    // TODO: Rescale
    if (!keep_dims) {
       %output = tosa.RESHAPE(%output) {new_shape=output_shape} : (tensor<%output.type>) -> tensor<output_shape, %output.dtype>
    }

    return %output;
}
```

### lower_resize_op()

```
Value lower_resize_op(Value %images, Value %size, shape output_shape, dtype output_dtype, mode_t mode)
{
    int64_t input_height = %images.shape[1]
    int64_t input_width = %images.shape[2]
    int64_t output_height = output_shape[1]
    int64_t output_width = output_shape[2]

    int32_t shift = 11

    double frac_y = (double)output_height / (double)input_height
    double frac_x = (double)output_width  / (double)input_width
    int32_t stride_y = (int32_t)std::round(frac_y * double(1 << shift))
    int32_t stride_x = (int32_t)std::round(frac_x * double(1 << shift))

    // Stride is int16
    while (stride_y >= 32768 || stride_x >= 32768) {
        shift--
        stride_y = (int32_t)std::round(frac_y * double(1 << shift))
        stride_x = (int32_t)std::round(frac_x * double(1 << shift))
    }

    %output = tosa.RESIZE(%images) {output_size={output_height, output_width},
                          offset={0, 0}, shift=shift, mode=mode} : (tensor<%images.type) -> tensor<output_shape, output_dtype>

}
```

### lower_reversev2_op()

```
Value lower_reverse_v2_op(Value %tensor, Value %axis)
{
    Value %output = %tensor

    if (%axis.num_elements == 0) {
       %output = tosa.IDENTITY(%tensor) : (tensor<%tensor.type>) -> tensor<%tensor.type>
    } else {
        for (int i = 0; i < %axis.shape[0]; i++) {
            size_t axis_val = positive_axis(%axis.as_constant()[i])
            %output = tosa.REVERSE(%output) {axis=%axis_val} : (tensor<%tensor.type>) -> tensor<%tensor.type>
        }
    }

    return %output
}
```

### lower_round_op()

```
Value lower_round_op(Value %x)
{
    %half = tosa.CONST() {value=0.5} : () -> tensor<{1}, float>
    %add = tosa.ADD(%x, %half) : (tensor<%x.type>, tensor<%half.type>) -> tensor<%add.type>
    %output = tosa.FLOOR(%add) : (tensor<%add.type>) -> tensor<%output.type>

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
       for (int i = 0; i < (output_shape.size - %condition.rank); i++) {
           cond_shape.push_front(1)
       }

       %condition = tosa.RESHAPE(%condition) {new_shape=cond_shape} : (tensor<%condition.type>) -> tensor<cond_shape, %condition.dtype>
    }

    %output = tosa.SELECT(%condition, %t, %e) : (tensor<%condition.type>, tensor<%t.type>, tensor<%t.type>) -> tensor<output_shape, %t.type>

    return %output
}
```

### lower_shape_op()

```
Value lower_shape_op(Value %input)
{
    vector <size_t> input_shape = %input.shape

    %shape = tosa.CONST() {value=input_shape} () -> tensor<{%input.rank}, int32_t>
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

    for (int i = 0; i < %padding.shape[0]; i++) {
        a1_padding[i + 2] = %padding.as_constant()[i]
    }

    %a1_pad = tosa.PAD(%input) {padding=a1_padding} : (tensor<%input.type>) -> tensor<%a1_pad.type>

    // Step 2. Reshape to
    // [batch + padded_shape[1] / block_shape[0], block_shape[0], ...
    //    padded_shape[M] / block_shape[M-1], block_shape[M-1]] +
    //    remaining_shape

    vector <size_t> a2_shape(1 + block_rank * 2 + remaining_shape_rank)
    a2_shape[0] = %input.shape[0]
    for (int i = 0; i < block_rank; i++) {
        a2_shape[1 + i * 2 + 0] = %a1_pad.shape[1 + i] / block_shape.as_constant()[i]
        a2_shape[1 + i * 2 + 1] = block_shape.as_constant()[i]
    }

    for (int i = 0; i < remaining_shape_rank; i++) {
        a2_shape[1 + block_rank * 2 + i] = %input.shape[1 + block_rank + i]
    }

    %a2_reshape = tosa.RESHAPE(%a1_pad) {new_shape=a2_shape} : (tensor<%a1_pad.type>) -> tensor<%a2_reshape.type>

    // Step 3 transpose to
    //  block-shape +
    //  [batch] +
    //  [padded_shape[1] / block_shape[0],
    // ...
    //  [padded_shape[M] / block_shape[M-1]] +
    //  remaining_shape
    vector <size_t> a3_perm(%a2_reshape.rank)
    size_t block_num_elems = 1

    for (int i = 0; i < block_rank; i++) {
        a3_perm[i] = 1 + 2 * i + 1
        a3_perm[block_rank + 1 + i] = 2 * i + 1
        block_num_elems *= %block.as_constant()[i]
    }

    a3_perm[block_rank] = 0
    for (int i = (1 + block_rank * 2); i < %a2_reshape.rank; i++) {
        a3_perm[i] = i
    }

    %a3_reshape = tosa.RESHAPE(%a2_reshape) {perm=a3_perm} : (tensor<%a2_reshape.type>) -> tensor<%a3_reshape.type>

    // Step 4. Reshape transposed tensor to
    // [ batch * prod(block_shape)] +
    // [ padded_shape[1] / block_shape[0],
    //   ...,
    // padded_shape[M] / block_shape[M-1]] +
    // remaining_shape

    vector <size_t> a4_shape(%input.rank)
    a4_shape[0] = batch_size * block_num_elements

    for (int i = 0; i < block_rank; i++) {
        a4_shape[i + 1] = %a1_pad.shape[i + 1] / %block.as_constant()[i]
    }

    for (int i = 0; i < remaining_block_shape; i++) {
        a4_shape[1 + block_rank + i] = %input.shape[1 + block_rank + i]
    }

    %output = tosa.RESHAPE(%a3_reshape) {new_shape=a4_shape} : (tensor<%a3_reshape.type>) -> tensor<%output.type>

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
    %a2_reshape = tosa.RESHAPE(%input) {new_shape=a2_shape} : (tensor<%input.type>) -> tensor<a2_shape, %input.dtype>
    %a3_transpose = tosa.TRANSPOSE(%a2_reshape) {perm={0, 1, 3, 2, 4, 5}} : (tensor<%a2_reshape.type>) -> tensor<%a3_transpose.type>

    vector <size_t> a4_shape = {%input.shape[0],
                                %input_shape[1] / block_size[0],
                                %input_shape[2] / block_size[1],
                                %input_shape[3] * block_size[0] * block_size[1]}
    %output = tosa.RESHAPE(%a3_transpose) {new_shape=%a4_shape} : (tensor<%a3_transpose.type>) -> tensor<a4_shape, %input.dtype>
    return %output
}
```

### lower_split_op()

```
Value lower_split_op(Value %value, size_t axis, size_t num_split)
{
    Value %output[]

    size_t slice_size = %value.shape[axis] / num_split

    for (int i = 0; i < num_split; i++) {
        vector <size_t> begin_vals, size_vals

        for (int j = 0; j < %value.rank; j++) {
            if (j == axis) {
               begin_vals.push_back(slice_size * i)
               size_vals.push_back(slice_size)
            } else {
               begin_vals.push_back(0)
               size_vals.push_bac(%value.shape[j])
            }

            %output[i] = tosa.SLICE(%value) {start=begin_vals, size=size_vals} (tensor<%value.type>) -> tensor<size_vals, %value.dtype>
        }

    }

    %output_list = tosa.IDENTITYN(%output) (tensor<%output:*.type>) -> tensor<%output_list:*.type>
    return %output_list
}
```

### lower_splitv_op()

```
Value lower_splitv_op(Value %value, vector <size_t> size_split, size_t axis)
{
   Value %output[]

   size_t curr_split_start = 0

   for (int i = 0; i < size_split.size(); i++) {
       vector <size_t> begin_vals, size_vals

       for (int j = 0; j < %value.rank; j++) {
           if (j == axis) {
              begin_vals.push_back(curr_split_start)
              size_vals.push_back(size_split[i])
           } else {
              begin_vals.push_back(0)
              size_vals.push_back(input.shape[j])
           }
       }

       %output[i] = tosa.SLICE(%value) {start=begin_vals, size=size_vals} (tensor<%value.type>) -> tensor<size_vals, %value.dtype>

       curr_split_start += size_split[i]
   }

    %output_list = tosa.IDENTITYN(%output) (tensor<%output:*.type>) -> tensor<%output_list:*.type>
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
       for (int i = 0; i < %input.rank; i++) {
           if (%input.shape[i] != 1) {
              reshape_dims.push_back(%input_shape[i])
           }
       }
    } else {
      // Remove the specified dimensions
      for (int i = 0; i < %input.rank; i++) {
          if (!squeeze_dims.find(i) || %input.shape[i] != -1) {
              reshape_dims.push_back(%input_shape[i])
          }
      }
    }

    %output = tosa.RESHAPE(%input) {new_shape=reshape_dims} (tensor<%input.type>) -> tensor<reshape_dims, %input.dtype>

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

    for (int i = 0; i < %input.rank; i++) {
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
    %a1_slice = tosa.SLICE(%input) {start=a1_start, size=a1_size} : (tensor<%input.type>) -> tensor<a1_size, %input.type>

    // Step 2: Reshape the sliced array: 2x as many dimensions as %input
    %a2_reshape = tosa.RESHAPE(%a1_slice) {new_shape=a2_shape} : (tensor<%a1_slice.type>) -> tensor<a2_shape, %input.type>

    // Step 3: Take a slice of the [0] index along each of the strided dimensions (even dimensions)
    %a3_slice = tosa.SLICE(%a2_reshape) {start=a3_start, size=a3_size} : (tensor<%a2_reshape.type>) -> tensor<a3_size, %input.type>

    // Step 4: Reshape the now-strided tensor back down to the desired number of dimensions
    %output = tosa.RESHAPE(%a3_slice) {new_shape=a4_shape} : (tensor<%a3_slice.type>) -> tensor<a4_shape, %input.type>

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
       for (int i = 0; i < %input.rank; i++) {
           if (i != axis)
              perms.push_back(i)
       }

       %transposed_value = tosa.TRANSPOSE(%value) {perms=perms} : (tensor<%value.type>) -> tensor<%transposed_value.shape, %value.dtype>

   } else {
      %transposed_value = %value
   }

   // Step 2: Slice [N, A, B, C] into [N] [A, B, C]
   for (int i = 0; i < %transposed_value.rank; i++) {
       vector <size_t> begin_vals, size_vals, shape_vals

       begin_vals.push_back(i)
       size_vals.push_back(1)

       for (int j = 1; j < %transposed_value.rank; j++) {
           begin_vals.push_back(0)
           size_vals.push_back(transposed_value.shape[j])
           shape_vals.push_back(transposed_value.shape[j])
       }

       %slice = %tosa.SLICE(%transposed_value) {begin=begin_vals, size=size_vals} (tensor<%tranposed_value.type>) -> tensor<size_vals, %value.dtype>
       %output_arr[i] = %tosa.RESHAPE(%slice) {new_shape=shape_vals} {begin=begin_vals, size=size_vals} (tensor<%slice.type>) -> tensor<shape_vals, %value.dtype>
   }

   // Combine array of sliced tensors into a list of tensors
   %output = tosa.IDENTITYN(%output_arr) (tensor<%output_arr:*.type>) -> tensor<%output_arr:*.type>
   return %output
}
```

### get_transpose_conv2d_padding_values_from_pad_type()

```
vector<int64_t> get_transpose_conv2d_padding_values_from_pad_type(tensorflow::Padding padding, tensorflow::TensorFormat data_format,
                                                         uint32_t first_filter_spatial_dim, type input_type, type filter_type
                                                         vector strides, vector dilations)
{
    int64_t pad_before, pad_after;
    vector<int64_t> computed_padding

    for (int i = 0; i < 2; i++) {
        int64_t ifm_dim = GetTensorSpatialDimIndex(4, data_format, i);
        int64_t ofm_dim = GetTensorSpatialDimIndex(4, data_format, i);
        int64_t filter_dim = first_filter_spatial_dim + 1

        int64_t ifm_size = input_shape[ifm_dim]
        int64_t ofm_size = output_dims[ofm_dim]
        int64_t filter_size = filter.shape[filter_dim]
        int64_t dim_dilation = dilations[i]
        int64_t dim_stride = strides[i]
        int effective_filter_size = (filter_size - 1) * dim_dilation + 1
        int total_padding = ((ifm_size - 1) * dim_stride + effective_filter_size - ofm_size)
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
    // TODO: fill in this function
}
```

### get_table_const_tensor()

```
Value get_table_const_tensor(function func)
{
    // TODO: fill in this function
}
```

## MLIR Passes Management

Legalization is built on multiple MLIR passes.

| MLIR Pass Name            | Input Dialect | Output Dialect | Description     |
| ------------------------- | ------------- | -------------- | --------------- |
| legalize_tf               | TensorFlow    | TOSA           | Legalize        |
:                           :               :                : TensorFlow      :
:                           :               :                : dialect to TOSA :
:                           :               :                : dialect         :
| legalize_tflite           | TensorFlow    | TOSA           | Legalize        |
:                           : Lite          :                : TensorFlow Lite :
:                           :               :                : dialect to TOSA :
:                           :               :                : dialect         :
| convert_tflite_qu8_to_qi8 | TensorFlow    | TensorFlow     | Convert         |
:                           : Lite          : Lite           : quantized uint8 :
:                           :               :                : graph to int8   :
:                           :               :                : graph           :
| constant_folding          | TOSA          | TOSA           | Constant        |
:                           :               :                : folding with    :
:                           :               :                : memory ops into :
:                           :               :                : single constant :
| make_broadcastable        | TOSA          | TOSA           | Reshape binary  |
:                           :               :                : op inputs to    :
:                           :               :                : have same rank  :
:                           :               :                : to run          :
:                           :               :                : broadcast       :

The pass list can be summarize as following pseudocode:

```
void generate_tosa(mlir::Module module, dialect_t input_dialect)
{
    mlir::PassManager pm

    switch(input_dialect)
    case TF:
        pm.addPass(legalize_tf)
        break
    case TFLite:
        pm.addPass(convert_tflite_qu8_to_qi8)
        pm.addPass(legalize_tflite)
        break
    default:
        break

    pm.addPass(constant_folding)
    pm.addPass(make_broadcastable)

    pm.run(module)
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
%output = tosa.ADD(%inputs:0, %inputs:1) : (tensor<%inputs:0.type>, tensor<%inputs:1.type>) -> tensor<%output.type>
for (int i = 2; i < %inputs.size; i++) {
    %output = tosa.ADD(%inputs:i, %output) : (tensor<%inputs:i.type>, tensor<%output.type>) -> tensor<%output.type>
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
%output = tf.all(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_ALL>(%input, %output.shape, %reduction_indicies, keep_dims)
```

### tf.Any

Computes the "logical or" of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.any(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_ANY>(%input, %output.shape, %reduction_indicies, keep_dims)
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
int64_t axis = positive_axis(%dimension)
%output = tosa.ARGMAX(%input) {axis=axis} : (tensor<%input.type>) -> tensor<%output.type>
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
%output = tosa.AVG_POOL2D(%value) {ksize=ksize, strides=strides, padding=tosa_padding} : (tensor<%value.type>) -> tensor<%output.type>
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
%bcast_value = apply_rank_broadcast(%value, %bias)
%bcast_bias  = apply_rank_broadcast(%bias, %value)
%output = tosa.ADD(%bcast_value, %bcast_bias) : (tensor<%bcast_value.type>, tensor<%bcast_bias.type>) -> tensor<%output.type>
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
%tosa_filter = tosa.TRANSPOSE(%filter) {perms={2, 0, 1, 3}} : (tensor<%filter.type>) -> tensor<%tosa_filter.type>

vector output_shape

for (int i = 0; i < input_sizes.size(); i++) {
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
%zero_bias = tosa.CONST() {value=0} () -> tensor<{1}, %input.dtype>
%output = tosa.TRANSPOSE_CONV2D(%out_backprop) {weight=%tosa_filter, bias=%zero_bias, outpad=tosa_pading, stride=strides, dilation==dilations, out_shape=out_shape} (tensor<%out_backprop.type>) -> tensor<%output.type>
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
%filter_tranpose = tosa.TRANSPOSE(%filter {perms={3, 0, 1, 2}} (tensor<%filter.type> -> tensor<%filter_transpose.type>

if (padding == "EXPLICIT") {
   tosa_padding =
       get_padding_values_from_explicit_pad_attr(explict_padding, data_format)
} else {
    %tosa_padding =
        get_padding_values_from_pad_type(%input, %filter.shape, padding, data_format,
                                         FORMAT_HWIO, strides, dilations)
}

// Create a zero bias tensor
%zero_bias = tosa.CONST() {value=0} () -> tensor<{1}, %input.dtype>

%output = tosa.CONV2D(%input, %filter_transpose, %zero_bias) {padding=tosa_padding, stride=strides, dilation=dilations} : (tensor<%input.type>, tensor<%filter_transpose.type>, tensor<%zero_bias.type>) -> tensor<%output.type>
```

### tf.Conv3D

TOSA lowering to tosa.CONV3D to be defined.

### tf.Cos

No TOSA lowering defined.

### tf.CrossReplicaSum

No TOSA lowering defined.

### tf.DepthToSpace

DepthToSpaceND for tensors of type T.

**TensorFlow Dialect**

```
%output = tf.DepthToSpace(%input) {block_size, data_format}
```

**TOSA Lowering**

```
%output = lower_depth_to_space_op(%input, block_size, data_format)
```

### tf.DepthwiseConv2dNative

Computes a 2-D depthwise convlution given 4-D input and filter tensors.

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
%zero_bias = tosa.CONST() {value={0} * bias_dim} () -> tensor<{bias_dim}, %input.dtype>

%output = tosa.DEPTHWISE_CONV2D(%input, %filter, %zero_bias) {stride=strides, dilation=dilations, padding=padding} : (tensor<%input.type>, tensor<%filter.type>, tensor<%zero_bias.type>) -> tensor<%output.type>
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

Fake-quantize the 'inputs' tensor of type float via global flats sclars min.

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
int64_t total_size = 1

for (int i = 0; i < %dims.shape[0]; i++) {
    total_size *= %dims[i]
}

vector<%value.dtype> fill_arr(total_size, %value)

%output = tosa.CONST() {value=fill_arr} () -> tensor<%output.type>
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

%epsilon_const = tosa.CONST() {value={epsilon}} () -> tensor<{1}, int64_t>

%op1 = tosa.SUB(%x, %bmean) : (tensor<%x.type>, tensor<%bmean.type>) -> tensor<%op1.type>
%op2 = tosa.ADD(%variance, %epsilon_const) : (tensor<%variance.type>, tensor<%epsilon_const.type>) -> tensor<%op2.type>
%op3 = tosa.RSQRT(%op2) : (tensor<%op2.type>) -> tensor<%op3.type>
%op4 = tosa.MUL(%op1, %op3) : (tensor<%op1.type>, tensor<%op3.type>) -> tensor<%op4.type>
%op5 = tosa.MUL(%op4, %scale) : (tensor<%op4.type>, tensor<%scale.type>) -> tensor<%op5.type>
%output = tosa.ADD(%op5, %offset) : (tensor<%.type>, tensor<%.type>) -> tensor<%output.type>
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

%epsilon_const = tosa.CONST() {value={epsilon}} () -> tensor<{1}, int64_t>

%op1 = tosa.SUB(%x, %bmean) : (tensor<%x.type>, tensor<%mean.type>) -> tensor<%op1.type>
%op2 = tosa.ADD(%variance, %epsilon_const) : (tensor<%variance.type>, tensor<%epsilon_const.type>) -> tensor<%op2.type>
%op3 = tosa.RSQRT(%op2) : (tensor<%op2.type>) -> tensor<%op3.type>
%op4 = tosa.MUL(%mean, %op3) : (tensor<%mean.type>, tensor<%op3.type>) -> tensor<%op4.type>
%op5 = tosa.MUL(%op4, %scale) : (tensor<%op4.type>, tensor<%scale.type>) -> tensor<%op5.type>
%output = tosa.ADD(%op5, %offset) : (tensor<%.type>, tensor<%.type>) -> tensor<%output.type>
```

### tf.GatherNd

No TOSA lowering defined.

### tf.Gather

Gathers slices from params according to indicies.

**TensorFlow Dialect**

```
%output = tf.Gather(%params, %indices) {validate_indicies}
```

**TOSA Lowering**

```
%output = tosa.GATHER(%params, %indicies) {axis=0} (tensor<%params.type>, tensor<%indicies.type>) -> tensor<%output.type>
```

### tf.GatherV2

Gathers slices from params axis according to indicies.

**TensorFlow Dialect**

```
%output = tf.GatherV2(%params, %indices, %axis) {batch_dims}
```

**TOSA Lowering**

```
%output = tosa.GATHER(%params, %indicies) {axis=%axis.to_constant()} (tensor<%params.type>, tensor<%indicies.type>) -> tensor<%output.type>
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
%output = tosa.IDENTITYN(%input) : (tensor<%input:*.type>) -> tensor<%output:*.type>
```

### tf.Identity

Returns a tensor with the same shape and contents as the input.

**TensorFlow Dialect**

```
%output = tf.Identity(%input)
```

**TOSA Lowering**

```
%output = tosa.IDENTITY(%input) : (tensor<%input.type>) -> tensor<%output.type>
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
%alpha_tensor = tosa.CONST() {value=alpha} : () -> tensor<{1}, alpha.type>
%features_alpha = tosa.MUL(%features, %alpha_tensor) : (tensor<%features.type>, tensor<%alpha_tensor.type>) -> tensor<%features_alpha.type>
%greater = tosa.GREATER(%features, %features_alpha) : (tensor<%features.type>, tensor<%features_alpha.type>) -> tensor<%greater.type>
%output = tosa.SELECT(%greater, %features, %features_alpha)  : (tensor<%greater.type>, tensor<%features.type>, tensor<%features_alpha.type>) -> tensor<%output.type>
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
%bcast_x = apply_rank_broadcast(%x, %y)
%bcast_y = apply_rank_broadcast(%y, %x)

%output_greater = tosa.GREATER(%bcast_x, %bcast_y) : (tensor<%bcast_x.type>, tensor<%bcast_y.type>) -> tensor<%output_greater.type>
%output = tosa.LOGICAL_NOT(%output_greater) : (tensor<%output_greater.type>) -> tensor<%output_greater.type>
```

### tf.Less

Returns the truth value of (x &lt; y) element-wise with broadcasting.

**TensorFlow Dialect**

```
%output = tf.LessEqual(%x, %y)
```

**TOSA Lowering**

```
%bcast_x = apply_rank_broadcast(%x, %y)
%bcast_y = apply_rank_broadcast(%y, %x)

%output_greater_equal = tosa.GREATER_EQUAL(%bcast_x, %bcast_y) : (tensor<%bcast_x.type>, tensor<%bcast_y.type>) -> tensor<%output_greater.type>
%output = tosa.LOGICAL_NOT(%output_greater_equal) : (tensor<%output_greater_equal.type>) -> tensor<%output_greater.type>
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
%output = tosa.MATMUL(%a, %b) : (tensor<%a.type>, tensor<%b.type>) -> tensor<%output.type>
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
%output = tf.Max(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_MAX>(%input, %output.shape, %reduction_indicies, keep_dims)
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
%output = tosa.MAX_POOL2D(%value) {ksize=ksize, strides=strides, padding=tosa_padding} : (tensor<%value.type>) -> tensor<%output.type>
```

### tf.Maximum

This operator is trivially lowered to tosa.MAXIMUM.

### tf.Mean

Computes the mean of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Mean(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_MEAN>(%input, %output.shape, %reduction_indicies, keep_dims)
```

### tf.Min

Computes the minimum of elements across dimensions of a tensor.

**TensorFlow Dialect**

```
%output = tf.Min(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_MIN>(%input, %output.shape, %reduction_indicies, keep_dims)
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
%bcast_x = apply_rank_broadcast(%x, %y)
%bcast_y = apply_rank_broadcast(%y, %x)

%equal = tosa.EQUAL(%bcast_x, %bcast_y) : (tensor<%bcast_x.type>, tensor<%bcast_y.type>) -> tensor<%equal.type>
%output = tosa.NOT(%equal) : (tensor<%equal.type>) -> tensor<%output.type>
```

### tf.OneHot

No TOSA lowering defined.

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
%output = tf.Prod(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_PRODUCT>(%input, %output.shape, %reduction_indicies, keep_dims)
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
%output = tosa.CONST() {value=%input.rank} : () -> tensor<{1}, int64_t>
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
%recip = tosa.RECIPROCAL(%y) : (tensor<%y.type>) -> tensor<%recip.type>
%output = tosa.MUL(%x, %recip) : (tensor<%x.type>, tensor<%recip.type>) -> tensor<%output.type>
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
%output = tosa.RELUN(%features) {max_val=6} : (tensor<%features.type>) -> tensor<%output.type>
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
%output = tosa.RELUN(%features) {max_val=0} : (tensor<%features.type>) -> tensor<%output.type>
```

### tf.Reshape

Reshapes a tensor.

**TensorFlow Dialect**

```
%output = tf.Reshape(%tensor, %shape)
```

**TOSA Lowering**

```
%output = tosa.RESHAPE(%tensor) {new_shape=%shape.as_constant} (tensor<%tensor.type>) -> tensor<%output.type>
```

### tf.ResizeBilinear

Resizes images to size using bilinear interpolation.

**TensorFlow Dialect**

```
%output = tf.ResizeBilinear(%images, %size) {align_corners, half_pixel_centers}
```

inferred from output shape. **TOSA Lowering**

```
%output = lower_resize_op(%images, %size, float, BILINEAR)
```

### tf.ResizeNearestNeighbor

Resizes images to size using nearest neighbor interpolation.

**TensorFlow Dialect**

```
%output = tf.ResizeNearestNeighbor(%images, %size) {align_corners, half_pixel_centers}
```

inferred from output shape. **TOSA Lowering**

```
%output = lower_resize_op(%images, %size, %output, float, NEAREST)
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
%bcast_x = apply_rank_broadcast(%x, %y)
%bcast_y = apply_rank_broadcast(%y, %x)
if (is_unsigned(%x.dtype)) {
  %output = tosa.LOGICAL_RIGHT_SHIFT(%bcast_x, %bcast_y) : (tensor<%bcast_x.type>, tensor<%bcast_y.type>) -> tensor<%output.type>
} else {
  %output = tosa.ARITHMETIC_RIGHT_SHIFT(%bcast_x, %bcast_y) : (tensor<%bcast_x.type>, tensor<%bcast_y.type>) -> tensor<%output.type>
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

%output = tosa.SLICE(%input) {start=begin, size=output_size} : (tensor<%input.type>) -> tensor<output_size, %input.dtype>
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
%op1 = tosa.EXP(%logits) : (tensor<%logits.type>) -> tensor<%op1.type>
%op2 = tosa.REDUCE_SUM(op1) {reduce_axis=(%logits.rank - 1)} : (tensor<%op1.type>) -> tensor<%op2.type>
%op3 = tosa.RECIPROCAL(%op2) : (tensor<%op2.type>) -> tensor<%op3.type>
%output = tosa.MUL(%op1, %op3) : (tensor<%op1.type>, tensor<%op3.type>) -> tensor<%output.type>
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
%output = tosa.MUL(%x, %x) (tensor<%x.type>, tensor<%x.type>) -> tensor<%output.type>
```

### tf.SquareDifference

Computes (x-y)\*(x-y) element-wise

**TensorFlow Dialect**

```
%output = tf.SquareDifference(%x, %y)
```

**TOSA Lowering**

```
%diff = tosa.SUB(%x, %y) (tensor<%x.type>, tensor<%y.type>) -> tensor<%diff.type>
%output = tosa.MUL(%diff, %diff) (tensor<%diff.type>, tensor<%diff.type>) -> tensor<%output.type>
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
%output = tf.Sum(%input, %reduction_indicies) {keep_dims}
```

**TOSA Lowering**

```
%output = lower_reduce_op<tosa.REDUCE_SUM>(%input, %output.shape, %reduction_indicies, keep_dims)
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
%output = tosa.TILE(%input) {multiples=%multiples.as_constant()} (tensor<%input.type>) -> tensor<%output.shape, %input.type>
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
%output = tosa.TRANSPOSE(%x) {perm=%perm.as_constant()} (%tensor<%x.type>) -> tensor<%output.type>
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
%output = tosa.CONST() {value={0} * %x.num_elements} : () -> tensor<%x.type>
```

## TensorFlow Lite MLIR Dialect Legalization (legalize_tflite)

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
%output = tosa.ADD(%inputs:0, %inputs:1) : (tensor<%inputs:0.type>, tensor<%inputs:1.type>) -> tensor<%output.type>
for (int i = 2 i < %inputs.size i++) {
    %output = tosa.ADD(%inputs:i, %output) : (tensor<%inputs:i.type>, tensor<%output.type>) -> tensor<%output.type>
}
```

### tfl.add

Element-wise addition operation.

**TensorFlow Lite Dialect**

```
%output = tfl.add(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all native typed,

Legalization:

```
%bcast_lhs = apply_rank_broadcast(%lhs, %rhs)
%bcast_rhs = apply_rank_broadcast(%rhs, %lhs)
%result = tosa.ADD(%bcast_lhs, %bcast_rhs) : (tensor<%bcast_lhs.type>, tensor<%bcast_rhs.type>) -> tensor<%output.type>
```

If input/output tensors are all quantized typed,

Prepare:

```
int32_t lhs_multiplier, rhs_multiplier, output_multiplier
int32_t lhs_shift, rhs_shift, output_shift
int32_t input_shift = 20
double max_scale_2x = 2.0 * max(%lhs.scale, %rhs.scale)
double lhs_scale = double(1 << input_shift) * %lhs.scale / max_scale_2x
double rhs_scale = double(1 << input_shift) * %rhs.scale / max_scale_2x
double output_scale = max_scale_2x / (%output.scale * double(1 << input_shift))

compute_scale_32(lhs_scale, lhs_multiplier, lhs_shift)
compute_scale_32(rhs_scale, rhs_multiplier, rhs_shift)
compute_scale_32(output_scale, output_multiplier, output_shift)

auto lhs_int32_type = tensor<%lhs.shape, tosa.int32>
auto rhs_int32_type = tensor<%rhs.shape, tosa.int32>
auto output_int32_type = tensor<%output.shape, tosa.int32>
```

Legalization:

```
%rescaled_lhs = tosa.RESCALE(%bcast_lhs) {multiplier=lhs_multiplier, shift=lhs_shift} : (tensor<%lhs.type>) -> lhs_int32_type
%rescaled_rhs = tosa.RESCALE(%bcast_rhs) {multiplier=rhs_multiplier, shift=rhs_shift} : (tensor<%rhs.type>) -> rhs_int32_type
%bcast_lhs = apply_rank_broadcast(%rescaled_lhs, %rescaled_rhs)
%bcast_rhs = apply_rank_broadcast(%rescaled_rhs, %rescaled_lhs)
%add = tosa.ADD(%bcast_lhs, %bcast_rhs) : (tensor<%bcast_lhs.type>, tensor<%bcast_rhs.type>) -> output_int32_type
%result = tosa.RESCALE(%add) {multiplier=output_multiplier, shift=output_shift} : (output_int32_type) -> tensor<%output.type>
```

### tfl.arg_max

ArgMax operator.

**TensorFlow Lite Dialect**

```
%output = tfl.arg_max(%input, %dim)
```

**TOSA Lowering**

```
%result = tosa.ARGMAX(%input) : {axis=positive_axis(%dim_const.as_constant(), %input.rank)} (tensor<%input.type>) -> tensor<%output.type>
```

### tfl.arg_min

No TOSA lowering defined.

### tfl.average_pool_2d

Average_pool_2d operator.

**TensorFlow Lite Dialect**

```
%output = tfl.average_pool_2d(%input) {filter_height, filter_width, padding, stride_h, stride_w, fused_activation_function} : (tensor<%input.type>) -> tensor<%output.type>
```

**TOSA Lowering**

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, tensor<{filter_height, filter_width}, tosa.int32>,
                                      {1, stride_h, stride_w, 1}, {1, 1, 1, 1})
```

If input/output tensors are all native typed,

Legalization:

```
%avgpool2d = tosa.AVG_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding} : (tensor<%input.type>) -> tensor<%output.type>
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
%avgpool2d = tosa.AVG_POOL2D(%input) {kernel={filter_height, filter_width}, stride={stride_h, stride_w}, padding=tosa_padding, quantization_info={input_zp=%input.zp, output_zp=%output.zp}} : (tensor<%input.type>) -> tensor<%output.type>
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

If input/output tensors are all native typed,

```
%result = tosa.CEIL(%x) : (tensor<%x.type>) -> tensor<%y.type>
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

If input/output tensors are all native typed,

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%conv2d = tosa.CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}} : (tensor<%input.type>, tensor<%filter.type>, tensor<%bias.type>) -> tensor<%output.type>
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
int32_t output_rescale_multiplier
int32_t output_rescale_shift
double output_rescale_scale = %input.scale * %filter.scale / %output.scale

compute_scale_32(output_rescale_scale, output_rescale_multiplier, output_rescale_shift)

auto acc_type = tensor<%output.shape, tosa.int32> // TODO: support 16x8->48

tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%conv2d = tosa.CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}, quantization_info={input_zp=%input.zp, weight_zp=%filter.zp}} : (tensor<%input.type>, tensor<%filter.type>, tensor<%bias.type>) -> acc_type
%rescale = tosa.RESCALE(%conv2d) {multiplier=output_multiplier, shift=output_shift} : (acc_type) -> tensor<%output.type>
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

### tfl.depthwise_conv_2d

Depthwise-separable convolution operator.

**TensorFlow Lite Dialect**

```
%output = tfl.depthwise_conv_2d(%input, %filter, %bias) {dilation_h_factor, dilation_w_factor, fused_activation_function, padding, stride_h, stride_w, depth_multiplier}
```

**TOSA Lowering**

If input/output tensors are all native typed,

Prepare:

```
tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%depthwise_conv2d = tosa.DEPTHWISE_CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}} : (tensor<%input.type>, tensor<%filter.type>, tensor<%bias.type>) -> tensor<%output.type>
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
int32_t output_rescale_multiplier, output_rescale_shift
double output_rescale_scale = %input.scale * %filter.scale / %output.scale

compute_scale_32(output_rescale_scale, output_rescale_multiplier, output_rescale_shift)

auto acc_type = tensor<%output.shape, tosa.int32> // TODO: support 16x8->48

tosa_padding =
     get_padding_values_from_pad_type(padding, NHWC, 1,
                                      %input.type, %filter.type,
                                      {1, stride_h, stride_w, 1}, {1, dilation_h_factor, dilation_w_factor, 1})
```

Legalization:

```
%depthwise_conv2d = tosa.DEPTHWISE_CONV2D(%input, %filter, %bias) {padding=tosa_padding, stride={stride_h, stride_w}, dilation={dilation_h_factor, dilation_w_factor}, quantization_info={input_zp=%input.zp, weight_zp=%filter.zp}} : (tensor<%input.type>, tensor<%filter.type>, tensor<%bias.type>) -> tensor<%output.type>
%rescale = tosa.RESCALE(%depthwise_conv2d) {multiplier=output_multiplier, shift=output_shift} : (acc_type) -> tensor<%output.type>
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
%result = convert_dequantized_op(%output.type, %input, %input.dtype.scale, %input.dtype.zp)
```

### tfl.div

Division operator.

**TensorFlow Lite Dialect**

```
%output = tfl.div(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all native typed,

```
%rcp = tosa.RECIPROCAL(%rhs) : (tensor<%rhs.type>) -> tensor<%rhs.type>
%mul = tosa.MUL(%lhs, %rcp) : (tensor<%lhs.type>, tensor<%rcp.type>) -> tensor<%output.type>
```

### tfl.elu

Exponential Linear Unit operator.

**TensorFlow Lite Dialect**

```
%y = tfl.elu(%x)
```

**TOSA Lowering**

If input/output tensors are all native typed,

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

If input/output tensors are all native typed,

```
%result = tosa.EXP(%x) : (tensor<%x.type>) -> tensor<%y.type>
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
for(int i = 0 i < dim_vec.size() i++) {
    total_size *= dim_vec[i]
}
filled_val = %value.as_constant()[0]
output_type = tensor<dim_vec, filled_val.dtype>
```

Legalization:

```
%result = tosa.CONST() {value=[filled_val] * total_size} : () -> output_type
```

### tfl.floor_div

Floor div operator.

**TensorFlow Lite Dialect**

```
%output = tfl.floor_div(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all native typed,

```
%recip = tosa.RECIPROCAL(%rhs) : (tensor<%rhs.shape, tosa.float>) -> tensor<%rhs.shape, tosa.float>
%mul = tosa.MUL(%lhs, %recip) : (tensor<%lhs.shape, tosa.float>, tensor<%rhs.shape, tosa.float>) -> tensor<%output.shape, tosa.float>
%result = tosa.FLOOR(%mul) : (tensor<%output.type>) -> tensor<%output.type>
```

### tfl.floor_mod

Division remainder.

**TensorFlow Lite Dialect**

```
%output = tfl.floor_mod(%lhs, %rhs)
```

**TOSA Lowering**

If input/output tensors are all native typed,

```
%recip = tosa.RECIPROCAL(%rhs) : (tensor<%rhs.shape, tosa.float>) -> tensor<%rhs.shape, tosa.float>
%mul = tosa.MUL(%lhs, %recip) : (tensor<%lhs.shape, tosa.float>, tensor<%rhs.shape, tosa.float>) -> tensor<%output.shape, tosa.float>
%floor = tosa.FLOOR(%mul) : (tensor<%output.type>) -> tensor<%output.type>
%result = tosa.SUB(%mul, %floor) : (tensor<%output.type>) -> tensor<%output.type>
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

If input/output tensors are all native typed,

Prepare:

```
// input[N, IC] x filter[OC, IC] + bias[OC] -> output[N, OC]
auto bias_shape = {%filter.shape[0]}
auto bias_type = tensor<bias_shape, tosa.float>
auto input_reshape_shape = {%input.num_elements / %filter.shape[1], %filter.shape[1]}
auto input_type = tensor<input_reshape_shape, %input.dtype>
```

Legalization:

```
if(!(%bias)) {
    %bias_val = tosa.CONST() {value=[0] * %filter.shape[3]} : () -> bias_type
}
else {
    %bias_val = %bias
}
if(%input.rank != 2) {
    %input_val = tosa.RESHAPE(%input) {shape=input_reshape_shape} : (tensor<%input.type>) -> input_type
}
else {
    %input_val = %input
}
%fc = tosa.FULLY_CONNECTED(%input_val, %filter, %bias_val) : (tensor<%input_val.type>, tensor<%filter_val.type>, tensor<%bias_val.type>) -> tensor<%output.type>
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
auto acc_dtype = tosa.int32 // TODO: support 16x8->48
auto bias_shape = {%filter.shape[3]}
auto bias_type = tensor<bias_shape, acc_dtype>
auto input_reshape_shape = {%input.num_elements / %filter.shape[1], %filter.shape[1]}
auto input_type = tensor<input_reshape_shape, %input.dtype>
auto acc_type = tensor<%output.shape, acc_dtype>
int32_t output_rescale_multiplier, output_rescale_shift
double output_rescale_scale = %input.scale * %filter.scale / %output.scale

compute_scale_32(output_rescale_scale, output_rescale_multiplier, output_rescale_shift)
```

Legalization:

```
if(!(%bias)) {
    %bias_val = tosa.CONST() {value=[0] * %filter.shape[3]} : () -> bias_type
}
else {
    %bias_val = %bias
}
if(%input.rank != 2) {
    %input_val = tosa.RESHAPE(%input) {shape=input_reshape_shape} : (tensor<%input.type>) -> input_type
}
else {
    %input_val = %input
}
%fc = tosa.FULLY_CONNECTED(%input_val, %filter, %bias_val) : (input_type, tensor<%filter_val.type>, bias_type) -> acc_type
%rescale = tosa.RESCALE(%fc) {multiplier=output_rescale_multiplier, shift=output_rescale_shift} : (acc_type) -> tensor<%output.type>
if(fused_activation != NONE) {
    %result = convert_fused_activation(%rescale, fused_activation_function)
}
else {
    %result = %rescale
}
```

### tfl.gather_nd

No TOSA lowering defined.

### tfl.gather

TODO: TOSA lowering

### tfl.greater_equal

This operator is trivially lowered to tosa.GREATER_EQUAL

### tfl.greater

This operator is trivially lowered to tosa.GREATER

### tfl.hard_swish

Hardswish activation function.

**TensorFlow Lite Dialect**

```
%out = tfl.hard_swish(%input)
```

**TOSA Lowering**

If input/output tensors are all native typed,

```
%const_3 = tosa.CONST() {value={3.0}} : () -> tensor<{1}, float>
%const_rcp6 = tosa.CONST() {value={1.0 / 6.0}} : () -> tensor<{1}, float>
%op1_add_in_3 = tosa.ADD(%input, %const_3) : (tensor<%input.type>, tensor<{1}, float>) -> tensor<%out.type>
%op2_relun_op1 = tosa.RELUN(%op1_add_in_3) {max=6.0} : (tensor<%out.type>) -> tensor<%out.type>
%op3_mul_in_op2 = tosa.MUL(%input, %op2_relun_op1) : (tensor<%input.type>, tensor<%out.type>) -> tensor<%out.type>
%op4_mul_op3_rcp6 = tosa.MUL(%op3, %const_rcp6) : (tensor<%out.type>, tensor<{1}, float>) -> tensor<%out.type>
```

If input/output tensors are all quantized typed,

Prepare:

```
const double input_sample_grain = 1.0 / 64.0;
auto hardswish_func = [input_sample_grain](int32_t x) -> int32_t {
    double v = (double)x * input_sample_grain
    double w = v + 3.0
    w = w < 0.0 ? 0.0 : w > 6.0 ? 6.0 : w
    v = v * w / 6.0
    return (int32_t)(std::round(32768.0 * v))
}
```

Legalization:

```
%table_const = get_table_const_tensor(hardswish_func)
%op1_rescale_in = tosa.RESCALE(%input) {multiplier=, shift=} (tensor<%input.type>) -> tensor<%input.shape, tosa.int16>
```

### tfl.l2_normalization

No TOSA lowering defined.

### tfl.lstm

No TOSA lowering defined.

### tfl.leaky_relu

TODO: TOSA lowering

### tfl.less_equal

TODO: TOSA lowering

### tfl.less

TODO: TOSA lowering

### tfl.local_response_normalization

No TOSA lowering defined.

### tfl.log

TODO: TOSA lowering

### tfl.log_softmax

TODO: TOSA lowering

### tfl.logical_and

This operator is trivially lowered to tosa.LOGICAL_AND

### tfl.logical_not

This operator is trivially lowered to tosa.LOGICAL_NOT

### tfl.logical_or

This operator is trivially lowered to tosa.LOGICAL_OR

### tfl.logistic

TODO: TOSA lowering

### tfl.matrix_diag

No TOSA lowering defined.

### tfl.matrix_set_diag

No TOSA lowering defined.

### tfl.max_pool_2d

TODO: TOSA lowering

### tfl.max_pooling_with_argmax_2d

No TOSA lowering defined.

### tfl.max_unpooling_2d

No TOSA lowering defined.

### tfl.maximum

This operator is trivially lowered to tosa.MAXIMUM

### tfl.mean

TODO: TOSA lowering

### tfl.minimum

This operator is trivially lowered to tosa.MINIMUM

### tfl.mirror_pad

No TOSA lowering defined.

### tfl.mul

TODO: TOSA lowering

### tfl.neg

This operator is trivially lowered to tosa.NEGATE

### tfl.non_max_suppression_v4

No TOSA lowering defined.

### tfl.non_max_suppression_v5

No TOSA lowering defined.

### tfl.not_equal

TODO: TOSA lowering

### tfl.NumericVerify

No TOSA lowering defined.

### tfl.one_hot

No TOSA lowering defined.

### tfl.prelu

TODO: TOSA lowering

### tfl.pack

TODO: TOSA lowering

### tfl.pad

{rivial_lowering} tosa.PAD

### tfl.padv2

No TOSA lowering defined.

### tfl.pow

TODO: TOSA lowering

### tfl.pseudo_qconst

This operator is trivially lowered to tosa.CONST

### tfl.quantize

TODO: TOSA lowering

### tfl.range

No TOSA lowering defined.

### tfl.rank

TODO: TOSA lowering

### tfl.reduce_any

TODO: TOSA lowering

### tfl.reduce_max

TODO: TOSA lowering

### tfl.reduce_min

TODO: TOSA lowering

### tfl.reduce_prod

TODO: TOSA lowering

### tfl.relu_n1_to_1

No TOSA lowering defined.

### tfl.relu6

TODO: TOSA lowering

### tfl.relu

TODO: TOSA lowering

### tfl.reshape

This operator is trivially lowered to tosa.RESHAPE

### tfl.resize_bilinear

TODO: TOSA lowering

### tfl.resize_nearest_neighbor

TODO: TOSA lowering

### tfl.reverse_sequence

No TOSA lowering defined.

### tfl.reverse_v2

TODO: TOSA lowering

### tfl.round

TODO: TOSA lowering

### tfl.rsqrt

TODO: TOSA lowering

### tfl.svdf

No TOSA lowering defined.

### tfl.segment_sum

No TOSA lowering defined.

### tfl.select

TODO: TOSA lowering

### tfl.select_v2

TODO: TOSA lowering

### tfl.shape

TODO: TOSA lowering

### tfl.sin

No TOSA lowering defined.

### tfl.slice

TODO: TOSA lowering

### tfl.softmax

TODO: TOSA lowering

### tfl.space_to_batch_nd

TODO: TOSA lowering

### tfl.space_to_depth

TODO: TOSA lowering

### tfl.pseudo_sparse_const

No TOSA lowering defined.

### tfl.pseudo_sparse_qconst

No TOSA lowering defined.

### tfl.sparse_to_dense

No TOSA lowering defined.

### tfl.split

TODO: TOSA lowering

### tfl.split_v

TODO: TOSA lowering

### tfl.sqrt

TODO: TOSA lowering

### tfl.square

TODO: TOSA lowering

### tfl.squared_difference

TODO: TOSA lowering

### tfl.squeeze

TODO: TOSA lowering

### tfl.strided_slice

### tfl.sub

This operator is trivially lowered to tosa.SUB

### tfl.sum

TODO: TOSA lowering

### tfl.tanh

TODO: TOSA lowering

### tfl.tile

TODO: TOSA lowering

### tfl.topk_v2

No TOSA lowering defined.

### tfl.transpose_conv

TODO: TOSA lowering

### tfl.transpose

This operator is trivially lowered to tosa.TRANSPOSE

### tfl.unidirectional_sequence_lstm

No TOSA lowering defined.

### tfl.unidirectional_sequence_rnn

No TOSA lowering defined.

### tfl.unique

No TOSA lowering defined.

### tfl.unpack

TODO: TOSA lowering

### tfl.where

No TOSA lowering defined.

### tfl.while

TODO: TOSA lowering

### tfl.yield

This operator is trivially lowered to tosa.YIELD

### tfl.zeros_like

TODO: TOSA lowering

## Common Passes

### make_broadcastable

### Applied to OP

For each of the following of OPs:

```
tosa.ADD, tosa.SUB, tosa.MUL, tosa.EQUAL, tosa.GREATER, tosa.GREATER_EQUAL
```

From:

```
%output = tosa.OP(%input1, %input2) : (tensor<%input1.type>, tensor<%input2.type>) -> tensor<%output.type>
```

To:

```
%bcast_input1 = apply_rank_broadcast(%input1, %input2)
%bcast_input2 = apply_rank_broadcast(%input2, %input1)
%result = tosa.OP(%bcast_input1, %bcast_input2) : (tensor<%bcast_input1.type>, tensor<%bcast_input2.type>) -> tensor<%output.type>
```

### constant_folding

#### tosa.CONST + tosa.RESHAPE

From:

```
%cst = tosa.CONST()
%transpose = tosa.RESHAPE(%cst)
```

To:

```
%result = tosa.CONST()
```

#### tosa.CONST + tosa.TRANSPOSE

From:

```
%cst = tosa.CONST()
%transpose = tosa.TRANSPOSE(%cst)
```

To:

```
%result = tosa.CONST()
```

### convert_tflite_qu8_to_qi8

From:

```
%cst = tosa.CONST() () -> tensor<%cst.shape, quant<u8>, ...>
```

From:

```
%result = tosa.CONST() () -> tensor<%cst.shape, quant<i8>, ...>
```
