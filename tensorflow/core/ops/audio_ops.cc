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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

namespace {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status DecodeWavShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

  DimensionHandle channels_dim;
  int32_t desired_channels;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_channels", &desired_channels));
  if (desired_channels == -1) {
    channels_dim = c->UnknownDim();
  } else {
    if (desired_channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     desired_channels);
    }
    channels_dim = c->MakeDim(desired_channels);
  }
  DimensionHandle samples_dim;
  int32_t desired_samples;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_samples", &desired_samples));
  if (desired_samples == -1) {
    samples_dim = c->UnknownDim();
  } else {
    if (desired_samples < 0) {
      return errors::InvalidArgument("samples must be non-negative, got ",
                                     desired_samples);
    }
    samples_dim = c->MakeDim(desired_samples);
  }
  c->set_output(0, c->MakeShape({samples_dim, channels_dim}));
  c->set_output(1, c->Scalar());
  return absl::OkStatus();
}

Status EncodeWavShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  c->set_output(0, c->Scalar());
  return absl::OkStatus();
}

Status SpectrogramShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  int32_t window_size;
  TF_RETURN_IF_ERROR(c->GetAttr("window_size", &window_size));
  if (window_size <= 1) {
    return errors::InvalidArgument("window size must be > 1, got ",
                                   window_size);
  }

  int32_t stride;
  TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));
  if (stride <= 0) {
    return errors::InvalidArgument("stride must be strictly positive, got ",
                                   stride);
  }

  DimensionHandle input_length = c->Dim(input, 0);
  DimensionHandle input_channels = c->Dim(input, 1);

  DimensionHandle output_length;
  if (!c->ValueKnown(input_length)) {
    output_length = c->UnknownDim();
  } else {
    const int64_t input_length_value = c->Value(input_length);
    const int64_t length_minus_window = (input_length_value - window_size);
    int64_t output_length_value;
    if (length_minus_window < 0) {
      output_length_value = 0;
    } else {
      output_length_value = 1 + (length_minus_window / stride);
    }
    output_length = c->MakeDim(output_length_value);
  }

  DimensionHandle output_channels =
      c->MakeDim(1 + NextPowerOfTwo(window_size) / 2);
  c->set_output(0,
                c->MakeShape({input_channels, output_length, output_channels}));
  return absl::OkStatus();
}

Status MfccShapeFn(InferenceContext* c) {
  ShapeHandle spectrogram;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &spectrogram));
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

  int32_t dct_coefficient_count;
  TF_RETURN_IF_ERROR(
      c->GetAttr("dct_coefficient_count", &dct_coefficient_count));

  DimensionHandle spectrogram_channels = c->Dim(spectrogram, 0);
  DimensionHandle spectrogram_length = c->Dim(spectrogram, 1);

  DimensionHandle output_channels = c->MakeDim(dct_coefficient_count);

  c->set_output(0, c->MakeShape({spectrogram_channels, spectrogram_length,
                                 output_channels}));
  return absl::OkStatus();
}

}  // namespace

REGISTER_OP("DecodeWav")
    .Input("contents: string")
    .Attr("desired_channels: int = -1")
    .Attr("desired_samples: int = -1")
    .Output("audio: float")
    .Output("sample_rate: int32")
    .SetShapeFn(DecodeWavShapeFn);

REGISTER_OP("EncodeWav")
    .Input("audio: float")
    .Input("sample_rate: int32")
    .Output("contents: string")
    .SetShapeFn(EncodeWavShapeFn);

REGISTER_OP("AudioSpectrogram")
    .Input("input: float")
    .Attr("window_size: int")
    .Attr("stride: int")
    .Attr("magnitude_squared: bool = false")
    .Output("spectrogram: float")
    .SetShapeFn(SpectrogramShapeFn);

REGISTER_OP("Mfcc")
    .Input("spectrogram: float")
    .Input("sample_rate: int32")
    .Attr("upper_frequency_limit: float = 4000")
    .Attr("lower_frequency_limit: float = 20")
    .Attr("filterbank_channel_count: int = 40")
    .Attr("dct_coefficient_count: int = 13")
    .Output("output: float")
    .SetShapeFn(MfccShapeFn);

}  // namespace tensorflow
