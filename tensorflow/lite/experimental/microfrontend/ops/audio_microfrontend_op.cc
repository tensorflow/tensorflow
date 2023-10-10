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
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {
REGISTER_OP("AudioMicrofrontend")
    .Input("audio: int16")
    .Output("filterbanks: out_type")
    .Attr("sample_rate: int = 16000")
    .Attr("window_size: int = 25")
    .Attr("window_step: int = 10")
    .Attr("num_channels: int = 32")
    .Attr("upper_band_limit: float = 7500.0")
    .Attr("lower_band_limit: float = 125.0")
    .Attr("smoothing_bits: int = 10")
    .Attr("even_smoothing: float = 0.025")
    .Attr("odd_smoothing: float = 0.06")
    .Attr("min_signal_remaining: float = 0.05")
    .Attr("enable_pcan: bool = false")
    .Attr("pcan_strength: float = 0.95")
    .Attr("pcan_offset: float = 80.0")
    .Attr("gain_bits: int = 21")
    .Attr("enable_log: bool = true")
    .Attr("scale_shift: int = 6")
    .Attr("left_context: int = 0")
    .Attr("right_context: int = 0")
    .Attr("frame_stride: int = 1")
    .Attr("zero_padding: bool = false")
    .Attr("out_scale: int = 1")
    .Attr("out_type: {uint16, float} = DT_UINT16")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &input));

      int sample_rate;
      TF_RETURN_IF_ERROR(ctx->GetAttr("sample_rate", &sample_rate));
      int window_size;
      TF_RETURN_IF_ERROR(ctx->GetAttr("window_size", &window_size));
      window_size *= sample_rate / 1000;
      int window_step;
      TF_RETURN_IF_ERROR(ctx->GetAttr("window_step", &window_step));
      window_step *= sample_rate / 1000;

      int num_channels;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_channels", &num_channels));
      int left_context;
      TF_RETURN_IF_ERROR(ctx->GetAttr("left_context", &left_context));
      int right_context;
      TF_RETURN_IF_ERROR(ctx->GetAttr("right_context", &right_context));
      int frame_stride;
      TF_RETURN_IF_ERROR(ctx->GetAttr("frame_stride", &frame_stride));

      DimensionHandle num_frames = ctx->Dim(input, 0);
      if (ctx->Value(num_frames) < window_size) {
        num_frames = ctx->MakeDim(0);
      } else {
        TF_RETURN_IF_ERROR(ctx->Subtract(num_frames, window_size, &num_frames));
        TF_RETURN_IF_ERROR(
            ctx->Divide(num_frames, window_step, false, &num_frames));
        TF_RETURN_IF_ERROR(
            ctx->Divide(num_frames, frame_stride, false, &num_frames));
        TF_RETURN_IF_ERROR(ctx->Add(num_frames, 1, &num_frames));
      }

      int stack_size = 1 + left_context + right_context;
      DimensionHandle num_features = ctx->MakeDim(num_channels);
      TF_RETURN_IF_ERROR(
          ctx->Multiply(num_features, stack_size, &num_features));

      ShapeHandle output = ctx->MakeShape({num_frames, num_features});
      ctx->set_output(0, output);
      return OkStatus();
    })
    .Doc(R"doc(
Audio Microfrontend Op.

This Op converts a sequence of audio data into one or more
feature vectors containing filterbanks of the input. The
conversion process uses a lightweight library to perform:

1. A slicing window function
2. Short-time FFTs
3. Filterbank calculations
4. Noise reduction
5. PCAN Auto Gain Control
6. Logarithmic scaling

Arguments
  audio: 1D Tensor, int16 audio data in temporal ordering.
  sample_rate: Integer, the sample rate of the audio in Hz.
  window_size: Integer, length of desired time frames in ms.
  window_step: Integer, length of step size for the next frame in ms.
  num_channels: Integer, the number of filterbank channels to use.
  upper_band_limit: Float, the highest frequency included in the filterbanks.
  lower_band_limit: Float, the lowest frequency included in the filterbanks.
  smoothing_bits: Int, scale up signal by 2^(smoothing_bits) before reduction.
  even_smoothing: Float, smoothing coefficient for even-numbered channels.
  odd_smoothing: Float, smoothing coefficient for odd-numbered channels.
  min_signal_remaining: Float, fraction of signal to preserve in smoothing.
  enable_pcan: Bool, enable PCAN auto gain control.
  pcan_strength: Float, gain normalization exponent.
  pcan_offset: Float, positive value added in the normalization denominator.
  gain_bits: Int, number of fractional bits in the gain.
  enable_log: Bool, enable logarithmic scaling of filterbanks.
  scale_shift: Integer, scale filterbanks by 2^(scale_shift).
  left_context: Integer, number of preceding frames to attach to each frame.
  right_context: Integer, number of preceding frames to attach to each frame.
  frame_stride: Integer, M frames to skip over, where output[n] = frame[n*M].
  zero_padding: Bool, if left/right context is out-of-bounds, attach frame of
                zeroes. Otherwise, frame[0] or frame[size-1] will be copied.
  out_scale: Integer, divide all filterbanks by this number.
  out_type: DType, type of the output Tensor, defaults to UINT16.

Returns
  filterbanks: 2D Tensor, each row is a time frame, each column is a channel.
)doc");

template <typename T>
class AudioMicrofrontendOp : public OpKernel {
 public:
  explicit AudioMicrofrontendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_rate", &sample_rate_));

    int window_size;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size));
    config_.window.size_ms = window_size;

    int window_step;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_step", &window_step));
    config_.window.step_size_ms = window_step;

    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("num_channels", &config_.filterbank.num_channels));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("upper_band_limit",
                                     &config_.filterbank.upper_band_limit));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower_band_limit",
                                     &config_.filterbank.lower_band_limit));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("smoothing_bits",
                                     &config_.noise_reduction.smoothing_bits));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("even_smoothing",
                                     &config_.noise_reduction.even_smoothing));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("odd_smoothing",
                                     &config_.noise_reduction.odd_smoothing));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("min_signal_remaining",
                                &config_.noise_reduction.min_signal_remaining));

    bool enable_pcan;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_pcan", &enable_pcan));
    config_.pcan_gain_control.enable_pcan = enable_pcan;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("pcan_strength",
                                     &config_.pcan_gain_control.strength));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("pcan_offset", &config_.pcan_gain_control.offset));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("gain_bits", &config_.pcan_gain_control.gain_bits));

    bool enable_log;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_log", &enable_log));
    config_.log_scale.enable_log = enable_log;

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("scale_shift", &config_.log_scale.scale_shift));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_context", &left_context_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_context", &right_context_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frame_stride", &frame_stride_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_padding", &zero_padding_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_scale", &out_scale_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* audio;
    OP_REQUIRES_OK(ctx, ctx->input("audio", &audio));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(audio->shape()),
                InvalidArgument("audio is not a vector"));

    auto audio_data =
        reinterpret_cast<const int16_t*>(audio->tensor_data().data());
    int audio_size = audio->NumElements();

    Tensor* filterbanks = nullptr;
    int window_size = config_.window.size_ms * sample_rate_ / 1000;
    int window_step = config_.window.step_size_ms * sample_rate_ / 1000;
    int num_frames = 0;
    int sampled_frames = 0;
    if (audio_size >= window_size) {
      num_frames = (audio_size - window_size) / window_step + 1;
      sampled_frames = (num_frames - 1) / frame_stride_ + 1;
    }
    TensorShape filterbanks_shape{
        sampled_frames,
        config_.filterbank.num_channels * (1 + left_context_ + right_context_)};
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, filterbanks_shape, &filterbanks));
    auto filterbanks_flat = filterbanks->flat<T>();

    struct FrontendState state;
    if (!TF_PREDICT_TRUE(
            FrontendPopulateState(&config_, &state, sample_rate_))) {
      ctx->CtxFailure(__FILE__, __LINE__,
                      Internal("failed to populate frontend state"));
      FrontendFreeStateContents(&state);
      return;
    }

    std::vector<std::vector<T>> frame_buffer(num_frames);
    int frame_index = 0;
    while (audio_size > 0) {
      size_t num_samples_read;
      struct FrontendOutput output = FrontendProcessSamples(
          &state, audio_data, audio_size, &num_samples_read);
      audio_data += num_samples_read;
      audio_size -= num_samples_read;

      if (output.values != nullptr) {
        frame_buffer[frame_index].reserve(output.size);
        int i;
        for (i = 0; i < output.size; ++i) {
          frame_buffer[frame_index].push_back(static_cast<T>(output.values[i]) /
                                              out_scale_);
        }
        ++frame_index;
      }
    }
    FrontendFreeStateContents(&state);

    int index = 0;
    std::vector<T> pad(config_.filterbank.num_channels, 0);
    int anchor;
    for (anchor = 0; anchor < frame_buffer.size(); anchor += frame_stride_) {
      int frame;
      for (frame = anchor - left_context_; frame <= anchor + right_context_;
           ++frame) {
        std::vector<T>* feature;
        if (zero_padding_ && (frame < 0 || frame >= frame_buffer.size())) {
          feature = &pad;
        } else if (frame < 0) {
          feature = &frame_buffer[0];
        } else if (frame >= frame_buffer.size()) {
          feature = &frame_buffer[frame_buffer.size() - 1];
        } else {
          feature = &frame_buffer[frame];
        }
        for (auto f : *feature) {
          filterbanks_flat(index++) = f;
        }
      }
    }
  }

 protected:
  int sample_rate_;
  struct FrontendConfig config_;
  int left_context_;
  int right_context_;
  int frame_stride_;
  bool zero_padding_;
  int out_scale_;

  TF_DISALLOW_COPY_AND_ASSIGN(AudioMicrofrontendOp);
};

REGISTER_KERNEL_BUILDER(Name("AudioMicrofrontend")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<uint16>("out_type"),
                        AudioMicrofrontendOp<uint16>);
REGISTER_KERNEL_BUILDER(Name("AudioMicrofrontend")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<float>("out_type"),
                        AudioMicrofrontendOp<float>);
}  // namespace tensorflow
