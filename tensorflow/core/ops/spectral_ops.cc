/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
Fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform over the inner-most
dimension of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fft
@end_compatibility
)doc");

REGISTER_OP("IFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
Inverse fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its inverse 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifft
@end_compatibility
)doc");

REGISTER_OP("FFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    })
    .Doc(R"doc(
2D fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform over the inner-most
2 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fft2
@end_compatibility
)doc");

REGISTER_OP("IFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    })
    .Doc(R"doc(
Inverse 2D fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform over the
inner-most 2 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their inverse 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifft2
@end_compatibility
)doc");

REGISTER_OP("FFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
3D fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform over the inner-most 3
dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their 3D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.fftn with 3 dimensions.
@end_compatibility
)doc");

REGISTER_OP("IFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
Inverse 3D fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform over the
inner-most 3 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their inverse 3D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.ifftn with 3 dimensions.
@end_compatibility
)doc");

Status RFFTShape(InferenceContext* c, const bool forward, const int rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));

  // Check that fft_length has shape [rank].
  ShapeHandle unused_shape;
  DimensionHandle unused_dim;
  ShapeHandle fft_length_input = c->input(1);
  TF_RETURN_IF_ERROR(c->WithRank(fft_length_input, 1, &unused_shape));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(fft_length_input, 0), rank, &unused_dim));
  const Tensor* fft_length_tensor = c->input_tensor(1);

  // If fft_length is unknown at graph creation time, we can't predict the
  // output size.
  if (fft_length_tensor == nullptr) {
    // We can't know the dimension of any of the rank inner dimensions of the
    // output without knowing fft_length.
    for (int i = 0; i < rank; ++i) {
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->UnknownDim(), &out));
    }
  } else {
    auto fft_length_as_vec = fft_length_tensor->vec<int32>();
    for (int i = 0; i < rank; ++i) {
      // For RFFT, replace the last dimension with fft_length/2 + 1.
      auto dim = forward && i == rank - 1 && fft_length_as_vec(i) != 0
                     ? fft_length_as_vec(i) / 2 + 1
                     : fft_length_as_vec(i);
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->MakeDim(dim), &out));
    }
  }

  c->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("RFFT")
    .Input("input: float")
    .Input("fft_length: int32")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 1); })
    .Doc(R"doc(
Real-valued fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
`fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
followed by the `fft_length / 2` positive-frequency terms.

input: A float32 tensor.
fft_length: An int32 tensor of shape [1]. The FFT length.
output: A complex64 tensor of the same rank as `input`. The inner-most
  dimension of `input` is replaced with the `fft_length / 2 + 1` unique
  frequency components of its 1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.rfft
@end_compatibility
)doc");

REGISTER_OP("IRFFT")
    .Input("input: complex64")
    .Input("fft_length: int32")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 1); })
    .Doc(R"doc(
Inverse real-valued fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
signal over the inner-most dimension of `input`.

The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
`fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
`fft_length` is not provided, it is computed from the size of the inner-most
dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
compute `input` is odd, it should be provided since it cannot be inferred
properly.

input: A complex64 tensor.
fft_length: An int32 tensor of shape [1]. The FFT length.
output: A float32 tensor of the same rank as `input`. The inner-most
  dimension of `input` is replaced with the `fft_length` samples of its inverse
  1D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.irfft
@end_compatibility
)doc");

REGISTER_OP("RFFT2D")
    .Input("input: float")
    .Input("fft_length: int32")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 2); })
    .Doc(R"doc(
2D real-valued fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

input: A float32 tensor.
fft_length: An int32 tensor of shape [2]. The FFT length for each dimension.
output: A complex64 tensor of the same rank as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier transform. The
  inner-most dimension contains `fft_length / 2 + 1` unique frequency
  components.

@compatibility(numpy)
Equivalent to np.fft.rfft2
@end_compatibility
)doc");

REGISTER_OP("IRFFT2D")
    .Input("input: complex64")
    .Input("fft_length: int32")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 2); })
    .Doc(R"doc(
Inverse 2D real-valued fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 2 dimensions of `input`.

The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 2 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

input: A complex64 tensor.
fft_length: An int32 tensor of shape [2]. The FFT length for each dimension.
output: A float32 tensor of the same rank as `input`. The inner-most 2
  dimensions of `input` are replaced with the `fft_length` samples of their
  inverse 2D Fourier transform.

@compatibility(numpy)
Equivalent to np.fft.irfft2
@end_compatibility
)doc");

REGISTER_OP("RFFT3D")
    .Input("input: float")
    .Input("fft_length: int32")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 3); })
    .Doc(R"doc(
3D real-valued fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 3 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

input: A float32 tensor.
fft_length: An int32 tensor of shape [3]. The FFT length for each dimension.
output: A complex64 tensor of the same rank as `input`. The inner-most 3
  dimensions of `input` are replaced with the their 3D Fourier transform. The
  inner-most dimension contains `fft_length / 2 + 1` unique frequency
  components.

@compatibility(numpy)
Equivalent to np.fft.rfftn with 3 dimensions.
@end_compatibility
)doc");

REGISTER_OP("IRFFT3D")
    .Input("input: complex64")
    .Input("fft_length: int32")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 3); })
    .Doc(R"doc(
Inverse 3D real-valued fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 3 dimensions of `input`.

The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 3 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

input: A complex64 tensor.
fft_length: An int32 tensor of shape [3]. The FFT length for each dimension.
output: A float32 tensor of the same rank as `input`. The inner-most 3
  dimensions of `input` are replaced with the `fft_length` samples of their
  inverse 3D real Fourier transform.

@compatibility(numpy)
Equivalent to np.irfftn with 3 dimensions.
@end_compatibility
)doc");

// Deprecated ops:
REGISTER_OP("BatchFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use FFT");
REGISTER_OP("BatchIFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use IFFT");
REGISTER_OP("BatchFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use FFT2D");
REGISTER_OP("BatchIFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use IFFT2D");
REGISTER_OP("BatchFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use FFT3D");
REGISTER_OP("BatchIFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .Deprecated(15, "Use IFFT3D");

}  // namespace tensorflow
