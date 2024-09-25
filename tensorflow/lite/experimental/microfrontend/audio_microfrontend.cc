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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace audio_microfrontend {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

typedef struct {
  int sample_rate;
  FrontendState* state;
  int left_context;
  int right_context;
  int frame_stride;
  bool zero_padding;
  int out_scale;
  bool out_float;
} TfLiteAudioMicrofrontendParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteAudioMicrofrontendParams;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  data->sample_rate = m["sample_rate"].AsInt32();

  struct FrontendConfig config;
  config.window.size_ms = m["window_size"].AsInt32();
  config.window.step_size_ms = m["window_step"].AsInt32();
  config.filterbank.num_channels = m["num_channels"].AsInt32();
  config.filterbank.upper_band_limit = m["upper_band_limit"].AsFloat();
  config.filterbank.lower_band_limit = m["lower_band_limit"].AsFloat();
  config.noise_reduction.smoothing_bits = m["smoothing_bits"].AsInt32();
  config.noise_reduction.even_smoothing = m["even_smoothing"].AsFloat();
  config.noise_reduction.odd_smoothing = m["odd_smoothing"].AsFloat();
  config.noise_reduction.min_signal_remaining =
      m["min_signal_remaining"].AsFloat();
  config.pcan_gain_control.enable_pcan = m["enable_pcan"].AsBool();
  config.pcan_gain_control.strength = m["pcan_strength"].AsFloat();
  config.pcan_gain_control.offset = m["pcan_offset"].AsFloat();
  config.pcan_gain_control.gain_bits = m["gain_bits"].AsInt32();
  config.log_scale.enable_log = m["enable_log"].AsBool();
  config.log_scale.scale_shift = m["scale_shift"].AsInt32();

  data->state = new FrontendState;
  FrontendPopulateState(&config, data->state, data->sample_rate);

  data->left_context = m["left_context"].AsInt32();
  data->right_context = m["right_context"].AsInt32();
  data->frame_stride = m["frame_stride"].AsInt32();
  data->zero_padding = m["zero_padding"].AsBool();
  data->out_scale = m["out_scale"].AsInt32();
  data->out_float = m["out_float"].AsBool();

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* data = reinterpret_cast<TfLiteAudioMicrofrontendParams*>(buffer);
  FrontendFreeStateContents(data->state);
  delete data->state;
  delete data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* data =
      reinterpret_cast<TfLiteAudioMicrofrontendParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt16);
  output->type = kTfLiteInt32;
  if (data->out_float) {
    output->type = kTfLiteFloat32;
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  int num_frames = 0;
  if (input->dims->data[0] >= data->state->window.size) {
    num_frames = (input->dims->data[0] - data->state->window.size) /
                     data->state->window.step / data->frame_stride +
                 1;
  }
  output_size->data[0] = num_frames;
  output_size->data[1] = data->state->filterbank.num_channels *
                         (1 + data->left_context + data->right_context);

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
void GenerateFeatures(TfLiteAudioMicrofrontendParams* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  const int16_t* audio_data = GetTensorData<int16_t>(input);
  int64_t audio_size = input->dims->data[0];

  T* filterbanks_flat = GetTensorData<T>(output);

  int num_frames = 0;
  if (audio_size >= data->state->window.size) {
    num_frames = (input->dims->data[0] - data->state->window.size) /
                     data->state->window.step +
                 1;
  }
  std::vector<std::vector<T>> frame_buffer(num_frames);

  int frame_index = 0;
  while (audio_size > 0) {
    size_t num_samples_read;
    struct FrontendOutput output = FrontendProcessSamples(
        data->state, audio_data, audio_size, &num_samples_read);
    audio_data += num_samples_read;
    audio_size -= num_samples_read;

    if (output.values != nullptr) {
      frame_buffer[frame_index].reserve(output.size);
      int i;
      for (i = 0; i < output.size; ++i) {
        frame_buffer[frame_index].push_back(static_cast<T>(output.values[i]) /
                                            data->out_scale);
      }
      ++frame_index;
    }
  }

  int index = 0;
  std::vector<T> pad(data->state->filterbank.num_channels, 0);
  int anchor;
  for (anchor = 0; anchor < frame_buffer.size(); anchor += data->frame_stride) {
    int frame;
    for (frame = anchor - data->left_context;
         frame <= anchor + data->right_context; ++frame) {
      std::vector<T>* feature;
      if (data->zero_padding && (frame < 0 || frame >= frame_buffer.size())) {
        feature = &pad;
      } else if (frame < 0) {
        feature = &frame_buffer[0];
      } else if (frame >= frame_buffer.size()) {
        feature = &frame_buffer[frame_buffer.size() - 1];
      } else {
        feature = &frame_buffer[frame];
      }
      for (auto f : *feature) {
        filterbanks_flat[index++] = f;
      }
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* data =
      reinterpret_cast<TfLiteAudioMicrofrontendParams*>(node->user_data);
  FrontendReset(data->state);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (data->out_float) {
    GenerateFeatures<float>(data, input, output);
  } else {
    GenerateFeatures<int32>(data, input, output);
  }

  return kTfLiteOk;
}

}  // namespace audio_microfrontend

TfLiteRegistration* Register_AUDIO_MICROFRONTEND() {
  static TfLiteRegistration r = {
      audio_microfrontend::Init, audio_microfrontend::Free,
      audio_microfrontend::Prepare, audio_microfrontend::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
