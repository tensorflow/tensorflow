/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/genai/kvcache.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace llm {

static const int kKeyTensor = 0;
static const int kValueTensor = 1;
static const int kFullKeyTensor = 0;
static const int kFullValueTensor = 1;
static const int kMaxNumCacheEntries = 1024;
static const int kRequiredNumDimensions = 4;

struct OpData {
  int num_entries;
};

void* KVCacheInit(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  // TODO(talumbau) Reset this value via ClearCaches in InternalBackendContext.
  op_data->num_entries = 0;
  return op_data;
}

TfLiteStatus KVCachePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  // Prepare the inputs.
  const TfLiteTensor* key;
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kKeyTensor, &key));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));

  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, value->type, kTfLiteFloat32);
  // Support only (B, S, N, H) for now.
  TF_LITE_ENSURE(context, NumDimensions(key) == kRequiredNumDimensions);
  // Enforce Batch == 1 for now.
  TF_LITE_ENSURE(context, GetTensorShape(key).Dims(0) == 1);
  TF_LITE_ENSURE(context, HaveSameShapes(key, value));

  // Create the key and value caches. Currently statically sized.
  TfLiteTensor* kfull;
  TfLiteTensor* vfull;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFullKeyTensor, &kfull));
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFullValueTensor, &vfull));

  kfull->type = kTfLiteFloat32;
  vfull->type = kTfLiteFloat32;

  TfLiteIntArray* input_dims = key->dims;
  TfLiteIntArray* kcache_dims = TfLiteIntArrayCopy(input_dims);
  TfLiteIntArray* vcache_dims = TfLiteIntArrayCopy(input_dims);
  kcache_dims->data[1] = kMaxNumCacheEntries;
  vcache_dims->data[1] = kMaxNumCacheEntries;

  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, kfull, kcache_dims));
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, vfull, vcache_dims));

  return kTfLiteOk;
}

void KVCacheFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus KVCacheEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kKeyTensor, &key));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));

  // Prepare the outputs.
  TfLiteTensor* kfull;
  TfLiteTensor* vfull;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFullKeyTensor, &kfull));
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFullValueTensor, &vfull));
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int current_num_entries = op_data->num_entries;

  // 0. Init the cache if there are no entries.
  if (current_num_entries == 0) {
    float* kfull_ptr = GetTensorData<float>(kfull);
    float* vfull_ptr = GetTensorData<float>(vfull);
    memset(kfull_ptr, 0, kfull->bytes);
    memset(vfull_ptr, 0, vfull->bytes);
  }

  // 1. Determine how many slots remain in the cache.
  const int num_slots_remaining = kMaxNumCacheEntries - current_num_entries;

  // 2. Determine how many slots these inputs take up.
  RuntimeShape shape(GetTensorShape(key));
  const int num_slots_needed = shape.Dims(1);
  const int elements_in_one_entry = shape.Dims(2) * shape.Dims(3);

  // 3. If we need more slots, make room in the cache by writing over oldest
  //    entries.
  if (num_slots_remaining < num_slots_needed) {
    char* k_ptr = reinterpret_cast<char*>(kfull->data.data);
    char* v_ptr = reinterpret_cast<char*>(vfull->data.data);
    const int slots_to_grow = num_slots_needed - num_slots_remaining;
    const int bytes_offset =
        sizeof(float) * elements_in_one_entry * slots_to_grow;

    const int num_bytes_to_shift =
        sizeof(float) * elements_in_one_entry * current_num_entries;
    // TODO(talumbau): This is O(cache_size) data motion. Consider optimizing
    // with a circular buffer or similar.
    memmove(k_ptr, k_ptr + bytes_offset, num_bytes_to_shift);
    memmove(v_ptr, v_ptr + bytes_offset, num_bytes_to_shift);
    // Just reduced the number of cache entries.
    current_num_entries -= slots_to_grow;
  }

  // 4. Put the key and value in their respective caches.
  const int64_t num_bytes_per_tensor = sizeof(float) * elements_in_one_entry;
  const int64_t offset = current_num_entries * num_bytes_per_tensor;
  memcpy((uint8_t*)(kfull->data.data) + offset, key->data.data, key->bytes);
  memcpy((uint8_t*)(vfull->data.data) + offset, value->data.data, value->bytes);

  // Update count.
  current_num_entries += num_slots_needed;
  op_data->num_entries = current_num_entries;

  return kTfLiteOk;
}

}  // namespace llm

TfLiteRegistration* Register_KV_CACHE() {
  static TfLiteRegistration r = {llm::KVCacheInit, llm::KVCacheFree,
                                 llm::KVCachePrepare, llm::KVCacheEval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
