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
#include "tensorflow/core/kernels/mfcc.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Create a speech fingerpring from spectrogram data.
class MfccOp : public OpKernel {
 public:
  explicit MfccOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("upper_frequency_limit",
                                             &upper_frequency_limit_));
    OP_REQUIRES_OK(context, context->GetAttr("lower_frequency_limit",
                                             &lower_frequency_limit_));
    OP_REQUIRES_OK(context, context->GetAttr("filterbank_channel_count",
                                             &filterbank_channel_count_));
    OP_REQUIRES_OK(context, context->GetAttr("dct_coefficient_count",
                                             &dct_coefficient_count_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spectrogram = context->input(0);
    OP_REQUIRES(context, spectrogram.dims() == 3,
                errors::InvalidArgument("spectrogram must be 3-dimensional",
                                        spectrogram.shape().DebugString()));
    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample_rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const int32_t sample_rate = sample_rate_tensor.scalar<int32>()();

    const int spectrogram_channels = spectrogram.dim_size(2);
    const int spectrogram_samples = spectrogram.dim_size(1);
    const int audio_channels = spectrogram.dim_size(0);

    Mfcc mfcc;
    mfcc.set_upper_frequency_limit(upper_frequency_limit_);
    mfcc.set_lower_frequency_limit(lower_frequency_limit_);
    mfcc.set_filterbank_channel_count(filterbank_channel_count_);
    mfcc.set_dct_coefficient_count(dct_coefficient_count_);
    OP_REQUIRES(context, mfcc.Initialize(spectrogram_channels, sample_rate),
                errors::InvalidArgument(
                    "Mfcc initialization failed for channel count ",
                    spectrogram_channels, " and sample rate ", sample_rate));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       TensorShape({audio_channels, spectrogram_samples,
                                    dct_coefficient_count_}),
                       &output_tensor));

    const float* spectrogram_flat = spectrogram.flat<float>().data();
    float* output_flat = output_tensor->flat<float>().data();

    for (int audio_channel = 0; audio_channel < audio_channels;
         ++audio_channel) {
      for (int spectrogram_sample = 0; spectrogram_sample < spectrogram_samples;
           ++spectrogram_sample) {
        const float* sample_data =
            spectrogram_flat +
            (audio_channel * spectrogram_samples * spectrogram_channels) +
            (spectrogram_sample * spectrogram_channels);
        std::vector<double> mfcc_input(sample_data,
                                       sample_data + spectrogram_channels);
        std::vector<double> mfcc_output;
        mfcc.Compute(mfcc_input, &mfcc_output);
        DCHECK_EQ(dct_coefficient_count_, mfcc_output.size());
        float* output_data =
            output_flat +
            (audio_channel * spectrogram_samples * dct_coefficient_count_) +
            (spectrogram_sample * dct_coefficient_count_);
        for (int i = 0; i < dct_coefficient_count_; ++i) {
          output_data[i] = mfcc_output[i];
        }
      }
    }
  }

 private:
  float upper_frequency_limit_;
  float lower_frequency_limit_;
  int32 filterbank_channel_count_;
  int32 dct_coefficient_count_;
};
REGISTER_KERNEL_BUILDER(Name("Mfcc").Device(DEVICE_CPU), MfccOp);

}  // namespace tensorflow
