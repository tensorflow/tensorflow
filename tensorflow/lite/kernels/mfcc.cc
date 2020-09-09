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
#include "tensorflow/lite/kernels/internal/mfcc.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/mfcc_dct.h"
#include "tensorflow/lite/kernels/internal/mfcc_mel_filterbank.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace mfcc {

enum KernelType {
  kReference,
};

typedef struct {
  float upper_frequency_limit;
  float lower_frequency_limit;
  int filterbank_channel_count;
  int dct_coefficient_count;
} TfLiteMfccParams;

constexpr int kInputTensorWav = 0;
constexpr int kInputTensorRate = 1;
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteMfccParams;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->upper_frequency_limit = m["upper_frequency_limit"].AsInt64();
  data->lower_frequency_limit = m["lower_frequency_limit"].AsInt64();
  data->filterbank_channel_count = m["filterbank_channel_count"].AsInt64();
  data->dct_coefficient_count = m["dct_coefficient_count"].AsInt64();
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteMfccParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteMfccParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_wav = GetInput(context, node, kInputTensorWav);
  const TfLiteTensor* input_rate = GetInput(context, node, kInputTensorRate);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input_wav), 3);
  TF_LITE_ENSURE_EQ(context, NumElements(input_rate), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, input_wav->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input_rate->type, kTfLiteInt32);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = input_wav->dims->data[0];
  output_size->data[1] = input_wav->dims->data[1];
  output_size->data[2] = params->dct_coefficient_count;

  return context->ResizeTensor(context, output, output_size);
}

// Input is a single squared-magnitude spectrogram frame. The input spectrum
// is converted to linear magnitude and weighted into bands using a
// triangular mel filterbank, and a discrete cosine transform (DCT) of the
// values is taken. Output is populated with the lowest dct_coefficient_count
// of these values.
template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteMfccParams*>(node->user_data);

  const TfLiteTensor* input_wav = GetInput(context, node, kInputTensorWav);
  const TfLiteTensor* input_rate = GetInput(context, node, kInputTensorRate);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const int32 sample_rate = *GetTensorData<int>(input_rate);

  const int spectrogram_channels = input_wav->dims->data[2];
  const int spectrogram_samples = input_wav->dims->data[1];
  const int audio_channels = input_wav->dims->data[0];

  internal::Mfcc mfcc;
  mfcc.set_upper_frequency_limit(params->upper_frequency_limit);
  mfcc.set_lower_frequency_limit(params->lower_frequency_limit);
  mfcc.set_filterbank_channel_count(params->filterbank_channel_count);
  mfcc.set_dct_coefficient_count(params->dct_coefficient_count);

  mfcc.Initialize(spectrogram_channels, sample_rate);

  const float* spectrogram_flat = GetTensorData<float>(input_wav);
  float* output_flat = GetTensorData<float>(output);

  for (int audio_channel = 0; audio_channel < audio_channels; ++audio_channel) {
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
      TF_LITE_ENSURE_EQ(context, params->dct_coefficient_count,
                        mfcc_output.size());
      float* output_data = output_flat +
                           (audio_channel * spectrogram_samples *
                            params->dct_coefficient_count) +
                           (spectrogram_sample * params->dct_coefficient_count);
      for (int i = 0; i < params->dct_coefficient_count; ++i) {
        output_data[i] = mfcc_output[i];
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace mfcc

TfLiteRegistration* Register_MFCC() {
  static TfLiteRegistration r = {mfcc::Init, mfcc::Free, mfcc::Prepare,
                                 mfcc::Eval<mfcc::kReference>};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
