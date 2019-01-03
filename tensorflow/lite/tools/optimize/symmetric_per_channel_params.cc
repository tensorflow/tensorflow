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
#include "tensorflow/lite/tools/optimize/symmetric_per_channel_params.h"

#include "flatbuffers/flexbuffers.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace internal {

namespace {
const char* kCustomQuantizationScales = "scales";
const char* kCustomQuantizationChannelDimIndex = "channel_dim_index";

// Writes scales and dimensions to custom quantization.
std::vector<uint8_t> SerializeToFlexBuffer(const std::vector<float>& scales,
                                           int channel_dim_index) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Vector(kCustomQuantizationScales, [&]() {
      for (float scale : scales) {
        fbb.Float(scale);
      }
    });
    fbb.Int(kCustomQuantizationChannelDimIndex, channel_dim_index);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

}  // namespace

/*static*/ TfLiteStatus SymmetricPerChannelParams::ReadFromTensor(
    const TensorT& tensor, std::unique_ptr<SymmetricPerChannelParams>* params) {
  const auto* q_params = tensor.quantization.get();
  if (!q_params) return kTfLiteError;
  auto custom_quantization = q_params->details.AsCustomQuantization();

  auto buffer = flexbuffers::GetRoot(custom_quantization->custom.data(),
                                     custom_quantization->custom.size());
  if (!buffer.IsMap()) {
    return kTfLiteError;
  }

  auto custom_quantization_map = buffer.AsMap();
  auto quant_scales =
      custom_quantization_map[kCustomQuantizationScales].AsVector();
  auto channel_dim_index =
      custom_quantization_map[kCustomQuantizationChannelDimIndex].AsInt32();
  std::vector<float> scales;
  scales.reserve(quant_scales.size());
  for (int i = 0; i < quant_scales.size(); i++) {
    scales.push_back(quant_scales[i].AsFloat());
  }
  *params =
      absl::make_unique<SymmetricPerChannelParams>(scales, channel_dim_index);
  return kTfLiteOk;
}

TfLiteStatus SymmetricPerChannelParams::AddToTensor(TensorT* tensor) const {
  flatbuffers::FlatBufferBuilder fbb;
  CustomQuantizationT* custom_quantization = new CustomQuantizationT;
  custom_quantization->custom =
      SerializeToFlexBuffer(scales_, channel_dim_index_);
  tensor->quantization = absl::make_unique<QuantizationParametersT>();
  tensor->quantization->details.type = QuantizationDetails_CustomQuantization;
  tensor->quantization->details.value = custom_quantization;
  return kTfLiteOk;
}
}  // namespace internal
}  // namespace optimize
}  // namespace tflite
