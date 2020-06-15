/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <ethosu_driver.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/flatbuffers/flexbuffers.h"

namespace tflite {
namespace ops {
namespace micro {
namespace custom {
namespace ethosu {

constexpr uint8_t CO_TYPE_ETHOSU = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size > 0);
  TF_LITE_ENSURE(context, context->tensors);
  TF_LITE_ENSURE(context, node->custom_initial_data_size > 0);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Get base addresses
  TfLiteTensor* tensor;
  int num_base_addr = node->inputs->size + node->outputs->size;
  int i = 0;
  int num_tensors = 0;
  uint64_t base_addrs[num_base_addr];
  void* cms_data;
  int cms_data_size;
  uint8_t co_type;
  int result;

  const uint8_t* custom_data =
      static_cast<uint8_t const*>(node->custom_initial_data);
  auto root = flexbuffers::GetRoot(custom_data, node->custom_initial_data_size);
  co_type = root.AsInt8();
  if (co_type != CO_TYPE_ETHOSU) {
    TF_LITE_KERNEL_LOG(context, "CO_TYPE != ETHOSU");
    return kTfLiteError;
  }

  // Get command stream data address and size
  tensor = &(context->tensors[node->inputs->data[0]]);
  cms_data = reinterpret_cast<void*>(tensor->data.uint8);
  cms_data_size = tensor->bytes;

  // Get adresses to weights/scratch/input data
  for (i = 1; i < node->inputs->size; ++i) {
    tensor = &(context->tensors[node->inputs->data[i]]);
    base_addrs[num_tensors] = reinterpret_cast<uint64_t>(tensor->data.uint8);
    num_tensors++;
  }

  // Get adresses to output data
  for (i = 0; i < node->outputs->size; ++i) {
    tensor = &(context->tensors[node->outputs->data[i]]);
    base_addrs[num_tensors] = reinterpret_cast<uint64_t>(tensor->data.uint8);
    num_tensors++;
  }

  result = ethosu_invoke(cms_data, cms_data_size, base_addrs, num_tensors);
  if (-1 == result) {
    return kTfLiteError;
  } else {
    return kTfLiteOk;
  }
}

}  // namespace ethosu

TfLiteRegistration* Register_ETHOSU() {
  static TfLiteRegistration r = {ethosu::Init, ethosu::Free, ethosu::Prepare,
                                 ethosu::Eval};
  return &r;
}

const char* GetString_ETHOSU() { return "ethos-u"; }

}  // namespace custom
}  // namespace micro
}  // namespace ops
}  // namespace tflite
