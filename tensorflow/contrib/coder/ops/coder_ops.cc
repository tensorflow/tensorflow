/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// clang-format off
REGISTER_OP("RangeEncode")
    .Input("data: int16")
    .Input("cdf: int32")
    .Output("encoded: string")
    .Attr("precision: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Using the provided cumulative distribution functions (CDF) inside `cdf`, returns
a range-code of `data`.

The shape of `cdf` should have one more axis than the shape of `data`, and the
prefix `cdf.shape[:-1]` should be broadcastable to `data.shape`. That is, for
every `i = 0,...,rank(data) - 1`, the op requires that either
`cdf.shape[i] == 1` or `cdf.shape[i] == data.shape[i]`. Note that this
broadcasting is limited in the sense that the number of axes must match, and
broadcasts only `cdf` but not `data`.

`data` should have an upper bound `m > 0` such that each element is an integer
in range `[0, m)`. Then the last dimension size of `cdf` must be `m + 1`. For
each element of `data`, the innermost strip of `cdf` is a vector representing a
CDF. For each k = 0,...,m, `cdf[..., k] / 2^precision` is the probability that
an outcome is less than `k` (not less than or equal to).

```
   cdf[..., 0] / 2^precision = Pr(data[...] < 0)
   cdf[..., 1] / 2^precision = Pr(data[...] < 1) = Pr(data[...] <= 0)
   cdf[..., 2] / 2^precision = Pr(data[...] < 2) = Pr(data[...] <= 1)
   ...
   cdf[..., m] / 2^precision = Pr(data[...] < m) = 1
```

Therefore each element of `cdf` must be in `[0, 2^precision]`.

Ideally `cdf[..., m]` should equal to `2^precision` but this is not a hard
requirement as long as `cdf[..., m] <= 2^precision`.

The encoded string neither contains the shape information of the encoded data
nor a termination symbol. Therefore the shape of the encoded data must be
explicitly provided to the decoder.

Implementation notes:

- Because of potential performance issues, the op does not check whether
elements of `data` is in the correct range `[0, m)`, or if `cdf` satisfies
monotonic increase property.

- For the range coder to decode the encoded string correctly, the decoder should
be able to reproduce the internal states of the encoder precisely. Otherwise,
the decoding would fail and once an error occur, all subsequent decoded values
are incorrect. For this reason, the range coder uses integer arithmetics and
avoids using any floating point operations internally, and `cdf` should contain
integers representing quantized probability mass rather than floating points. 

data: An int16 tensor.
cdf: An int32 tensor representing the CDF's of `data`. Each integer is divided
  by `2^precision` to represent a fraction.
encoded: A range-coded scalar string.
precision: The number of bits for probability quantization. Must be <= 16.
)doc");


REGISTER_OP("RangeDecode")
    .Input("encoded: string")
    .Input("shape: int32")
    .Input("cdf: int32")
    .Output("decoded: int16")
    .Attr("precision: int >= 1")
    .SetShapeFn([] (InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Decodes a range-coded `code` into an int32 tensor of shape `shape`.

This is the reverse op of RangeEncode. The shape of the tensor that was encoded
should be known by the caller.

Implementation notes:

- If wrong input was given (e.g., corrupt `encoded` string, or `cdf` or
`precision` do not match encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return error status.

encoded: A scalar string tensor from RangeEncode.
shape: An int32 1-D tensor representing the shape of the data encoded by
  RangeEncode.
decoded: An int16 tensor with shape equal to `shape`.
precision: The number of bits for probability quantization. Must be <= 16, and
  must match the precision used by RangeEncode that produced `encoded`.
)doc");

REGISTER_OP("PmfToQuantizedCdf")
    .Input("pmf: float")
    .Output("cdf: int32")
    .Attr("precision: int >= 1")
    .SetShapeFn([] (InferenceContext* c) {
      ShapeHandle in;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &in));
      DimensionHandle last;
      TF_RETURN_IF_ERROR(c->Add(c->Dim(in, -1), 1, &last));
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(in, -1, last, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Converts PMF to quantized CDF. This op uses floating-point operations
internally. Therefore the quantized output may not be consistent across multiple
platforms. For entropy encoders and decoders to have the same quantized CDF on
different platforms, the quantized CDF should be produced once and saved, then
the saved quantized CDF should be used everywhere.

After quantization, if PMF does not sum to 2^precision, then some values of PMF
are increased or decreased to adjust the sum to equal to 2^precision.

Note that the input PMF is pre-quantization. The input PMF is not normalized
by this op prior to quantization. Therefore the user is responsible for
normalizing PMF if necessary.
)doc");
// clang-format on
}  // namespace tensorflow
