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

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/spectrogram.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

#include "flatbuffers/flexbuffers.h"  // flatbuffers

namespace tflite {
namespace ops {
namespace custom {
namespace audio_spectrogram {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
};

typedef struct {
  int window_size;
  int stride;
  bool magnitude_squared;
  int output_height;
  internal::Spectrogram* spectrogram;
} TfLiteAudioSpectrogramParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteAudioSpectrogramParams;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->window_size = m["window_size"].AsInt64();
  data->stride = m["stride"].AsInt64();
  data->magnitude_squared = m["magnitude_squared"].AsBool();

  data->spectrogram = new internal::Spectrogram;

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* params = reinterpret_cast<TfLiteAudioSpectrogramParams*>(buffer);
  delete params->spectrogram;
  delete params;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioSpectrogramParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  TF_LITE_ENSURE(context, params->spectrogram->Initialize(params->window_size,
                                                          params->stride));
  const int64_t sample_count = input->dims->data[0];
  const int64_t length_minus_window = (sample_count - params->window_size);
  if (length_minus_window < 0) {
    params->output_height = 0;
  } else {
    params->output_height = 1 + (length_minus_window / params->stride);
  }
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = input->dims->data[1];
  output_size->data[1] = params->output_height;
  output_size->data[2] = params->spectrogram->output_frequency_channels();

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioSpectrogramParams*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context, params->spectrogram->Initialize(params->window_size,
                                                          params->stride));

  const float* input_data = GetTensorData<float>(input);

  const int64_t sample_count = input->dims->data[0];
  const int64_t channel_count = input->dims->data[1];

  const int64_t output_width = params->spectrogram->output_frequency_channels();

  float* output_flat = GetTensorData<float>(output);

  std::vector<float> input_for_channel(sample_count);
  for (int64_t channel = 0; channel < channel_count; ++channel) {
    float* output_slice =
        output_flat + (channel * params->output_height * output_width);
    for (int i = 0; i < sample_count; ++i) {
      input_for_channel[i] = input_data[i * channel_count + channel];
    }
    std::vector<std::vector<float>> spectrogram_output;
    TF_LITE_ENSURE(context,
                   params->spectrogram->ComputeSquaredMagnitudeSpectrogram(
                       input_for_channel, &spectrogram_output));
    TF_LITE_ENSURE_EQ(context, spectrogram_output.size(),
                      params->output_height);
    TF_LITE_ENSURE(context, spectrogram_output.empty() ||
                                (spectrogram_output[0].size() == output_width));
    for (int row_index = 0; row_index < params->output_height; ++row_index) {
      const std::vector<float>& spectrogram_row = spectrogram_output[row_index];
      TF_LITE_ENSURE_EQ(context, spectrogram_row.size(), output_width);
      float* output_row = output_slice + (row_index * output_width);
      if (params->magnitude_squared) {
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
  return kTfLiteOk;
}

}  // namespace audio_spectrogram

TfLiteRegistration* Register_AUDIO_SPECTROGRAM() {
  static TfLiteRegistration r = {
      audio_spectrogram::Init, audio_spectrogram::Free,
      audio_spectrogram::Prepare,
      audio_spectrogram::Eval<audio_spectrogram::kReference>};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
