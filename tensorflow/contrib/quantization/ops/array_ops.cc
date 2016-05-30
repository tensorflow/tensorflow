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

namespace tensorflow {

// --------------------------------------------------------------------------

REGISTER_OP("QuantizeV2")
    .Input("input: float")
    .Input("min_range: float")
    .Input("max_range: float")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype")
    .Attr("mode: {'MIN_COMBINED', 'MIN_FIRST'} = 'MIN_COMBINED'")
    .Doc(R"doc(
Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8, out[i] -= (range(T) + 1) / 2.0
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
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

One thing to watch out for is that the operator may choose to adjust the
requested minimum and maximum values slightly during the quantization process,
so you should always use the output ports as the range for further calculations.
For example, if the requested minimum and maximum values are close to equal,
they will be separated by a small epsilon value to prevent ill-formed quantized
buffers from being created. Otherwise, you can end up with buffers where all the
quantized values map to the same float value, which causes problems for
operations that have to perform further calculations on them.

min_range: The minimum scalar value possibly produced for the input.
max_range: The maximum scalar value possibly produced for the input.
output: The quantized data produced from the float input.
output_min: The actual minimum scalar value used for the output.
output_max: The actual maximum scalar value used for the output.

)doc");

REGISTER_OP("Dequantize")
    .Input("input: T")
    .Input("min_range: float")
    .Input("max_range: float")
    .Output("output: float")
    .Attr("T: quantizedtype")
    .Attr("mode: {'MIN_COMBINED', 'MIN_FIRST'} = 'MIN_COMBINED'")
    .Doc(R"doc(
Dequantize the 'input' tensor into a float Tensor.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
if T == qint8, in[i] += (range(T) + 1)/ 2.0
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

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
```

min_range: The minimum scalar value possibly produced for the input.
max_range: The maximum scalar value possibly produced for the input.

)doc");

REGISTER_OP("QuantizedConcat")
    .Input("concat_dim: int32")
    .Input("values: N * T")
    .Input("input_mins: N * float32")
    .Input("input_maxes: N * float32")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Doc(R"doc(
Concatenates quantized tensors along one dimension.

concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
  range [0, rank(values)).
values: The `N` Tensors to concatenate. Their ranks and types must match,
  and their sizes must match in all dimensions except `concat_dim`.
input_mins: The minimum scalar values for each of the input tensors.
input_maxes: The maximum scalar values for each of the input tensors.
output_min: The float value that the minimum quantized output value represents.
output_max: The float value that the maximum quantized output value represents.
output: A `Tensor` with the concatenation of values stacked along the
  `concat_dim` dimension.  This tensor's shape matches that of `values` except
  in `concat_dim` where it has the sum of the sizes.
)doc");

REGISTER_OP("QuantizeAndDequantize")
    .Input("input: T")
    .Attr("signed_input: bool = true")
    .Attr("num_bits: int32 = 8")
    .Attr("range_given: bool = false")
    .Attr("input_min: float = 0")
    .Attr("input_max: float = 0")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Quantizes then dequantizes a tensor.

This op simulates the precision loss from the quantized forward pass by:
1. Quantizing the tensor to fixed point numbers, which should match the target
   quantization method when it is used in inference.
2. Dequantizing it back to floating point numbers for the following ops, most
   likely matmul.

There are different ways to quantize. This version does not use the full range
of the output type, choosing to elide the lowest possible value for symmetry
(e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
quantization), so that 0.0 maps to 0.

To perform this op, we first find the range of values in our tensor. The range
we use is always centered on 0, so we find m such that

1. m = max(abs(input_min), abs(input_max)) if range_given is true,
2. m = max(max(abs(min_elem(input)), abs(max_elem(input))) otherwise.

Our input tensor range is then [-m, m].

Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
If signed_input is true, this is

  [min_fixed, max_fixed ] =
      [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].

Otherwise, if signed_input is false, the fixed-point range is

  [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].

From this we compute our scaling factor, s:

  s = (max_fixed - min_fixed) / (2 * m).

Now we can quantize and dequantize the elements of our tensor.  An element e
is transformed into e':

  e' = (e * s).round_to_nearest() / s.

Note that we have a different number of buckets in the signed vs. unsigned
cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
vs. 255 in the unsigned case.

For example, suppose num_bits = 8 and m = 1.  Then

  [min_fixed, max_fixed] = [-127, 127], and
  s = (127 + 127) / 2 = 127.

Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
{-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}.

input: Tensor to quantize and then dequantize.
signed_input: If the quantization is signed or unsigned.
num_bits: The bitwidth of the quantization.
range_given: If the range is given or should be computed from the tensor.
input_min: If range is given, this is the min of the range.
input_max: If range is given, this is the max of the range.
)doc");
}  // namespace tensorflow
