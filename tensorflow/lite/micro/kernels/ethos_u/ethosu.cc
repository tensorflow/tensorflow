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

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr uint8_t CO_TYPE_ETHOSU = 1;

struct OpData {
  int cms_data_size;
  int base_addr_idx;
  int base_addr_size_idx;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(context != nullptr);
  TF_LITE_ENSURE(context, node->inputs->size > 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  TF_LITE_ENSURE(context, node->custom_initial_data_size > 0);

  OpData* data = static_cast<OpData*>(node->user_data);
  int num_base_addr = node->inputs->size + node->outputs->size;

  // Request arrays for the base address pointers and sizes
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, num_base_addr * sizeof(uint64_t), &data->base_addr_idx));
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, num_base_addr * sizeof(size_t), &data->base_addr_size_idx));

  // Get command stream data size
  TfLiteTensor* tensor = context->GetTensor(context, node->inputs->data[0]);
  data->cms_data_size = tensor->bytes;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  // Get base addresses
  TfLiteEvalTensor* tensor;
  int i = 0;
  int num_tensors = 0;
  void* cms_data;
  uint8_t co_type;
  int result;
  const OpData* data = static_cast<const OpData*>(node->user_data);
  uint64_t* base_addrs = static_cast<uint64_t*>(
      context->GetScratchBuffer(context, data->base_addr_idx));
  size_t* base_addrs_size = static_cast<size_t*>(
      context->GetScratchBuffer(context, data->base_addr_size_idx));

  const uint8_t* custom_data =
      static_cast<uint8_t const*>(node->custom_initial_data);
  auto root = flexbuffers::GetRoot(custom_data, node->custom_initial_data_size);
  co_type = root.AsInt8();
  if (co_type != CO_TYPE_ETHOSU) {
    TF_LITE_KERNEL_LOG(context, "CO_TYPE != ETHOSU");
    return kTfLiteError;
  }

  // Get command stream data address
  tensor = context->GetEvalTensor(context, node->inputs->data[0]);
  cms_data = reinterpret_cast<void*>(tensor->data.uint8);

  // Get addresses to weights/scratch/input data
  for (i = 1; i < node->inputs->size; ++i) {
    tensor = context->GetEvalTensor(context, node->inputs->data[i]);
    base_addrs[num_tensors] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor->data.uint8));
    size_t byte_size = 1;
    for (int k = 0; k < tensor->dims->size; k++) {
      byte_size = byte_size * tensor->dims->data[k];
    }
    base_addrs_size[num_tensors] = byte_size;
    num_tensors++;
  }

  // Get addresses to output data
  for (i = 0; i < node->outputs->size; ++i) {
    tensor = context->GetEvalTensor(context, node->outputs->data[i]);
    base_addrs[num_tensors] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor->data.uint8));
    size_t byte_size = 1;
    for (int k = 0; k < tensor->dims->size; k++) {
      byte_size = byte_size * tensor->dims->data[k];
    }
    base_addrs_size[num_tensors] = byte_size;
    num_tensors++;
  }

  // Ethos-U guarantees that the tensors that require a base pointer are among
  // the 8 first tensors
  num_tensors = std::min(num_tensors, 8);

  result = ethosu_invoke_v2(cms_data, data->cms_data_size, base_addrs,
                            base_addrs_size, num_tensors);
  if (-1 == result) {
    return kTfLiteError;
  } else {
    return kTfLiteOk;
  }
}

}  // namespace

TfLiteRegistration* Register_ETHOSU() {
  static TfLiteRegistration r = {Init,
                                 Free,
                                 Prepare,
                                 Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

const char* GetString_ETHOSU() { return "ethos-u"; }

}  // namespace tflite
