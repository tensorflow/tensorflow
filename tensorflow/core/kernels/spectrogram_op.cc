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
#include "tensorflow/core/kernels/spectrogram.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Create a spectrogram frequency visualization from audio data.
class AudioSpectrogramOp : public OpKernel {
 public:
  explicit AudioSpectrogramOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("magnitude_squared", &magnitude_squared_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional",
                                        input.shape().DebugString()));
    Spectrogram spectrogram;
    OP_REQUIRES(context, spectrogram.Initialize(window_size_, stride_),
                errors::InvalidArgument(
                    "Spectrogram initialization failed for window size ",
                    window_size_, " and stride ", stride_));

    const auto input_as_matrix = input.matrix<float>();

    const int64 sample_count = input.dim_size(0);
    const int64 channel_count = input.dim_size(1);

    const int64 output_width = spectrogram.output_frequency_channels();
    const int64 length_minus_window = (sample_count - window_size_);
    int64 output_height;
    if (length_minus_window < 0) {
      output_height = 0;
    } else {
      output_height = 1 + (length_minus_window / stride_);
    }
    const int64 output_slices = channel_count;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({output_slices, output_height, output_width}),
            &output_tensor));
    auto output_flat = output_tensor->flat<float>().data();

    std::vector<float> input_for_channel(sample_count);
    for (int64 channel = 0; channel < channel_count; ++channel) {
      OP_REQUIRES(context, spectrogram.Reset(),
                  errors::InvalidArgument("Failed to Reset()"));

      float* output_slice =
          output_flat + (channel * output_height * output_width);
      for (int i = 0; i < sample_count; ++i) {
        input_for_channel[i] = input_as_matrix(i, channel);
      }
      std::vector<std::vector<float>> spectrogram_output;
      OP_REQUIRES(context,
                  spectrogram.ComputeSquaredMagnitudeSpectrogram(
                      input_for_channel, &spectrogram_output),
                  errors::InvalidArgument("Spectrogram compute failed"));
      OP_REQUIRES(context, (spectrogram_output.size() == output_height),
                  errors::InvalidArgument(
                      "Spectrogram size calculation failed: Expected height ",
                      output_height, " but got ", spectrogram_output.size()));
      OP_REQUIRES(context,
                  spectrogram_output.empty() ||
                      (spectrogram_output[0].size() == output_width),
                  errors::InvalidArgument(
                      "Spectrogram size calculation failed: Expected width ",
                      output_width, " but got ", spectrogram_output[0].size()));
      for (int row_index = 0; row_index < output_height; ++row_index) {
        const std::vector<float>& spectrogram_row =
            spectrogram_output[row_index];
        DCHECK_EQ(spectrogram_row.size(), output_width);
        float* output_row = output_slice + (row_index * output_width);
        if (magnitude_squared_) {
          for (int i = 0; i < output_width; ++i) {
            output_row[i] = spectrogram_row[i];
          }
        } else {
          for (int i = 0; i < output_width; ++i) {
            output_row[i] = sqrtf(spectrogram_row[i]);
          }
        }
      }
    }
  }

 private:
  int32 window_size_;
  int32 stride_;
  bool magnitude_squared_;
};
REGISTER_KERNEL_BUILDER(Name("AudioSpectrogram").Device(DEVICE_CPU),
                        AudioSpectrogramOp);

}  // namespace tensorflow
