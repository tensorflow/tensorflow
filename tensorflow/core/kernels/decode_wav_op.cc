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

// See docs in ../ops/audio_ops.cc

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/wav/wav_io.h"

namespace tensorflow {

// Decode the contents of a WAV file
class DecodeWavOp : public OpKernel {
 public:
  explicit DecodeWavOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_channels", &desired_channels_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_samples", &desired_samples_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));
    const string& wav_string = contents.scalar<tstring>()();
    OP_REQUIRES(context, wav_string.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("WAV contents are too large for int: ",
                                        wav_string.size()));

    std::vector<float> decoded_samples;
    uint32 decoded_sample_count;
    uint16 decoded_channel_count;
    uint32 decoded_sample_rate;
    OP_REQUIRES_OK(context,
                   wav::DecodeLin16WaveAsFloatVector(
                       wav_string, &decoded_samples, &decoded_sample_count,
                       &decoded_channel_count, &decoded_sample_rate));

    OP_REQUIRES(context, desired_channels_ >= -1,
                errors::InvalidArgument("desired_channels must be >= -1, got ",
                                        desired_channels_));
    OP_REQUIRES(context, desired_samples_ >= -1,
                errors::InvalidArgument("desired_samples must be >= -1, got ",
                                        desired_samples_));
    int32_t output_sample_count;
    if (desired_samples_ == -1) {
      output_sample_count = decoded_sample_count;
    } else {
      output_sample_count = desired_samples_;
    }
    int32_t output_channel_count;
    if (desired_channels_ == -1) {
      output_channel_count = decoded_channel_count;
    } else {
      output_channel_count = desired_channels_;
    }

    OP_REQUIRES(
        context, output_sample_count >= 0,
        errors::InvalidArgument("Output sample count must be >= 0, got ",
                                output_sample_count));
    OP_REQUIRES(
        context, output_channel_count >= 0,
        errors::InvalidArgument("Output channel count must be >= 0, got ",
                                output_channel_count));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({output_sample_count, output_channel_count}),
            &output));

    auto output_matrix = output->matrix<float>();
    for (int sample = 0; sample < output_sample_count; ++sample) {
      for (int channel = 0; channel < output_channel_count; ++channel) {
        float output_value;
        if (sample >= decoded_sample_count) {
          output_value = 0.0f;
        } else {
          int source_channel;
          if (channel < decoded_channel_count) {
            source_channel = channel;
          } else {
            source_channel = decoded_channel_count - 1;
          }
          const int decoded_index =
              (sample * decoded_channel_count) + source_channel;
          output_value = decoded_samples[decoded_index];
        }
        output_matrix(sample, channel) = output_value;
      }
    }

    Tensor* sample_rate_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &sample_rate_output));
    sample_rate_output->flat<int32>()(0) = decoded_sample_rate;
  }

 private:
  int32 desired_channels_;
  int32 desired_samples_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeWav").Device(DEVICE_CPU), DecodeWavOp);

}  // namespace tensorflow
