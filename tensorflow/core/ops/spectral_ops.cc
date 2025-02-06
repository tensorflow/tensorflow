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
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("IFFT")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("FFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("IFFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("FFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

REGISTER_OP("IFFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

REGISTER_OP("FFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("IFFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

absl::Status RFFTShape(InferenceContext* c, const bool forward,
                       const int rank) {
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
  return absl::OkStatus();
}

REGISTER_OP("RFFT")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 1); });

REGISTER_OP("IRFFT")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 1); });

REGISTER_OP("RFFT2D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 2); });

REGISTER_OP("IRFFT2D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 2); });

REGISTER_OP("RFFT3D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 3); });

REGISTER_OP("IRFFT3D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 3); });

REGISTER_OP("RFFTND")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("IRFFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

// Deprecated ops:
REGISTER_OP("BatchFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT");
REGISTER_OP("BatchIFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT");
REGISTER_OP("BatchFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT2D");
REGISTER_OP("BatchIFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT2D");
REGISTER_OP("BatchFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT3D");
REGISTER_OP("BatchIFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT3D");

}  // namespace tensorflow
