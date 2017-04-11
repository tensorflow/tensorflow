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

namespace tensorflow {

namespace {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status DecodeWavShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

  DimensionHandle channels_dim;
  int32 desired_channels;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_channels", &desired_channels));
  if (desired_channels == 0) {
    channels_dim = c->UnknownDim();
  } else {
    if (desired_channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     desired_channels);
    }
    channels_dim = c->MakeDim(desired_channels);
  }
  DimensionHandle samples_dim;
  int32 desired_samples;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_samples", &desired_samples));
  if (desired_samples == 0) {
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
  return Status::OK();
}

Status EncodeWavShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
  c->set_output(0, c->Scalar());
  return Status::OK();
}

}  // namespace

REGISTER_OP("DecodeWav")
    .Input("contents: string")
    .Attr("desired_channels: int = -1")
    .Attr("desired_samples: int = -1")
    .Output("audio: float")
    .Output("sample_rate: int32")
    .SetShapeFn(DecodeWavShapeFn)
    .Doc(R"doc(
Decode a 16-bit PCM WAV file to a float tensor.

The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

When desired_channels is set, if the input contains fewer channels than this
then the last channel will be duplicated to give the requested number, else if
the input has more channels than requested then the additional channels will be
ignored.

If desired_samples is set, then the audio will be cropped or padded with zeroes
to the requested length.

The first output contains a Tensor with the content of the audio samples. The
lowest dimension will be the number of channels, and the second will be the
number of samples. For example, a ten-sample-long stereo WAV file should give an
output shape of [10, 2].

contents: The WAV-encoded audio, usually from a file.
desired_channels: Number of sample channels wanted.
desired_samples: Length of audio requested.
audio: 2-D with shape `[length, channels]`.
sample_rate: Scalar holding the sample rate found in the WAV header.
)doc");

REGISTER_OP("EncodeWav")
    .Input("audio: float")
    .Input("sample_rate: int32")
    .Output("contents: string")
    .SetShapeFn(EncodeWavShapeFn)
    .Doc(R"doc(
Encode audio data using the WAV file format.

This operation will generate a string suitable to be saved out to create a .wav
audio file. It will be encoded in the 16-bit PCM format. It takes in float
values in the range -1.0f to 1.0f, and any outside that value will be clamped to
that range.

`audio` is a 2-D float Tensor of shape `[length, channels]`.
`sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

audio: 2-D with shape `[length, channels]`.
sample_rate: Scalar containing the sample frequency.
contents: 0-D. WAV-encoded file contents.
)doc");

}  // namespace tensorflow
