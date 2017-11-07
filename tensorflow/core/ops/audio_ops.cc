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
  int32 desired_samples;
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
  return Status::OK();
}

Status EncodeWavShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  c->set_output(0, c->Scalar());
  return Status::OK();
}

Status SpectrogramShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  int32 window_size;
  TF_RETURN_IF_ERROR(c->GetAttr("window_size", &window_size));
  int32 stride;
  TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));

  DimensionHandle input_length = c->Dim(input, 0);
  DimensionHandle input_channels = c->Dim(input, 1);

  DimensionHandle output_length;
  if (!c->ValueKnown(input_length)) {
    output_length = c->UnknownDim();
  } else {
    const int64 input_length_value = c->Value(input_length);
    const int64 length_minus_window = (input_length_value - window_size);
    int64 output_length_value;
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
  return Status::OK();
}

Status MfccShapeFn(InferenceContext* c) {
  ShapeHandle spectrogram;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &spectrogram));
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

  int32 dct_coefficient_count;
  TF_RETURN_IF_ERROR(
      c->GetAttr("dct_coefficient_count", &dct_coefficient_count));

  DimensionHandle spectrogram_channels = c->Dim(spectrogram, 0);
  DimensionHandle spectrogram_length = c->Dim(spectrogram, 1);

  DimensionHandle output_channels = c->MakeDim(dct_coefficient_count);

  c->set_output(0, c->MakeShape({spectrogram_channels, spectrogram_length,
                                 output_channels}));
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

REGISTER_OP("AudioSpectrogram")
    .Input("input: float")
    .Attr("window_size: int")
    .Attr("stride: int")
    .Attr("magnitude_squared: bool = false")
    .Output("spectrogram: float")
    .SetShapeFn(SpectrogramShapeFn)
    .Doc(R"doc(
Produces a visualization of audio data over time.

Spectrograms are a standard way of representing audio information as a series of
slices of frequency information, one slice for each window of time. By joining
these together into a sequence, they form a distinctive fingerprint of the sound
over time.

This op expects to receive audio data as an input, stored as floats in the range
-1 to 1, together with a window width in samples, and a stride specifying how
far to move the window between slices. From this it generates a three
dimensional output. The lowest dimension has an amplitude value for each
frequency during that time slice. The next dimension is time, with successive
frequency slices. The final dimension is for the channels in the input, so a
stereo audio input would have two here for example.

This means the layout when converted and saved as an image is rotated 90 degrees
clockwise from a typical spectrogram. Time is descending down the Y axis, and
the frequency decreases from left to right.

Each value in the result represents the square root of the sum of the real and
imaginary parts of an FFT on the current window of samples. In this way, the
lowest dimension represents the power of each frequency in the current window,
and adjacent windows are concatenated in the next dimension.

To get a more intuitive and visual look at what this operation does, you can run
tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
resulting spectrogram as a PNG image.

input: Float representation of audio data.
window_size: How wide the input window is in samples. For the highest efficiency
  this should be a power of two, but other values are accepted.
stride: How widely apart the center of adjacent sample windows should be.
magnitude_squared: Whether to return the squared magnitude or just the
  magnitude. Using squared magnitude can avoid extra calculations.
spectrogram: 3D representation of the audio frequencies as an image.
)doc");

REGISTER_OP("Mfcc")
    .Input("spectrogram: float")
    .Input("sample_rate: int32")
    .Attr("upper_frequency_limit: float = 4000")
    .Attr("lower_frequency_limit: float = 20")
    .Attr("filterbank_channel_count: int = 40")
    .Attr("dct_coefficient_count: int = 13")
    .Output("output: float")
    .SetShapeFn(MfccShapeFn)
    .Doc(R"doc(
Transforms a spectrogram into a form that's useful for speech recognition.

Mel Frequency Cepstral Coefficients are a way of representing audio data that's
been effective as an input feature for machine learning. They are created by
taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
higher frequencies that are less significant to the human ear. They have a long
history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
is a good resource to learn more.

spectrogram: Typically produced by the Spectrogram op, with magnitude_squared
  set to true.
sample_rate: How many samples per second the source audio used.
upper_frequency_limit: The highest frequency to use when calculating the
  ceptstrum.
lower_frequency_limit: The lowest frequency to use when calculating the
  ceptstrum.
filterbank_channel_count: Resolution of the Mel bank used internally.
dct_coefficient_count: How many output channels to produce per time slice.
)doc");

}  // namespace tensorflow
