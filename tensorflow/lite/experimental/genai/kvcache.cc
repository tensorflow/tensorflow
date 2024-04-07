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

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/cache_buffer.h"
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
static const int kRequiredNumDimensions = 4;
static const int kDefaultMaxNumCacheEntries = 2048;
static const int kDefaultNumTransformerLayers = 32;
static const int kDefaultTransformerLayerId = 0;

static const int KVCACHE_KEY_RESOURCE = 42;
static const int KVCACHE_VALUE_RESOURCE = 43;

struct OpData {
  int num_layers;
  int layer_index;
  int max_num_entries;
  // Pointers to the key and value cache buffers that this Op doesn't own
  // (and therefore does not free on destruction of this Op).
  resource::CacheBuffer* key_cache_buffer;
  resource::CacheBuffer* value_cache_buffer;
  bool is_initialized;
};

void* KVCacheInit(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  // TODO(talumbau) Reset this value via ClearCaches in InternalBackendContext.
  op_data->max_num_entries = -1;
  op_data->num_layers = -1;
  op_data->layer_index = -1;
  op_data->key_cache_buffer = nullptr;
  op_data->value_cache_buffer = nullptr;
  op_data->is_initialized = false;
  return op_data;
}

TfLiteStatus KVCachePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  if (!op_data->is_initialized) {
    const uint8_t* buffer =
        reinterpret_cast<const uint8_t*>(node->custom_initial_data);
    const size_t length = node->custom_initial_data_size;
    auto flexbuffer_map = flexbuffers::GetRoot(buffer, length).AsMap();
    int32_t max_num_entries = flexbuffer_map["kv_cache_max"].AsInt32();
    int32_t num_layers = flexbuffer_map["num_layers"].AsInt32();
    int32_t layer_index = flexbuffer_map["layer_index"].AsInt32();
    op_data->max_num_entries =
        max_num_entries > 0 ? max_num_entries : kDefaultMaxNumCacheEntries;
    op_data->num_layers =
        num_layers > 0 ? num_layers : kDefaultNumTransformerLayers;
    op_data->layer_index =
        layer_index > 0 ? layer_index : kDefaultTransformerLayerId;
    op_data->is_initialized = true;
  }

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
  // Need this to ensure memory remains persistent across invokes.
  kfull->allocation_type = kTfLiteArenaRwPersistent;
  vfull->allocation_type = kTfLiteArenaRwPersistent;

  kfull->type = kTfLiteFloat32;
  vfull->type = kTfLiteFloat32;

  TfLiteIntArray* input_dims = key->dims;
  TfLiteIntArray* kcache_dims = TfLiteIntArrayCopy(input_dims);
  TfLiteIntArray* vcache_dims = TfLiteIntArrayCopy(input_dims);
  kcache_dims->data[1] = op_data->max_num_entries;
  vcache_dims->data[1] = op_data->max_num_entries;

  TfLiteIntArray* kcache_buffer_dims = TfLiteIntArrayCreate(5);
  // Batch
  kcache_buffer_dims->data[0] = input_dims->data[0];
  // Number of layers
  kcache_buffer_dims->data[1] = op_data->num_layers;
  // Sequence Length
  kcache_buffer_dims->data[2] = op_data->max_num_entries;
  // Num heads
  kcache_buffer_dims->data[3] = input_dims->data[2];
  // Head dim
  kcache_buffer_dims->data[4] = input_dims->data[3];

  TfLiteIntArray* vcache_buffer_dims = TfLiteIntArrayCopy(kcache_buffer_dims);

  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, kfull, kcache_dims));
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, vfull, vcache_dims));

  // Get the pointer to the tensor for our buffer storage.
  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();

  if (resources.count(KVCACHE_KEY_RESOURCE) == 0) {
    auto* cbuffer = new resource::CacheBuffer();
    cbuffer->Initialize(*kcache_buffer_dims);
    resources.emplace(KVCACHE_KEY_RESOURCE, cbuffer);
    op_data->key_cache_buffer = cbuffer;
  }
  if (resources.count(KVCACHE_VALUE_RESOURCE) == 0) {
    auto* cbuffer = new resource::CacheBuffer();
    cbuffer->Initialize(*vcache_buffer_dims);
    resources.emplace(KVCACHE_VALUE_RESOURCE, cbuffer);
    op_data->value_cache_buffer = cbuffer;
  }
  TfLiteIntArrayFree(kcache_buffer_dims);
  TfLiteIntArrayFree(vcache_buffer_dims);
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

  float* key_cache_ptr = op_data->key_cache_buffer->GetBuffer();
  float* value_cache_ptr = op_data->value_cache_buffer->GetBuffer();
  int layer_index = op_data->layer_index;
  int current_num_entries =
      op_data->key_cache_buffer->GetNumEntries(layer_index);

  // 1. Determine how many slots remain in the cache.
  const int num_slots_remaining =
      op_data->max_num_entries - current_num_entries;

  // 2. Determine how many slots these inputs take up.
  RuntimeShape shape(GetTensorShape(key));
  const int num_slots_needed = shape.Dims(1);
  const int elements_in_one_entry = shape.Dims(2) * shape.Dims(3);
  const int elements_in_one_block =
      op_data->max_num_entries * elements_in_one_entry;

  // 3. If we need more slots, make room in the cache by writing over oldest
  //    entries.
  uint8_t* k_ptr = reinterpret_cast<uint8_t*>(key_cache_ptr);
  uint8_t* v_ptr = reinterpret_cast<uint8_t*>(value_cache_ptr);
  k_ptr = k_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;
  v_ptr = v_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;
  if (num_slots_remaining < num_slots_needed) {
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
  memcpy(k_ptr + offset, key->data.data, key->bytes);
  memcpy(v_ptr + offset, value->data.data, value->bytes);

  // 5. Set the output tensors with the relevant block's cache.
  memcpy((uint8_t*)(kfull->data.data), k_ptr, kfull->bytes);
  memcpy((uint8_t*)(vfull->data.data), v_ptr, vfull->bytes);

  // Update counts.
  current_num_entries += num_slots_needed;
  op_data->key_cache_buffer->SetNumEntries(layer_index, current_num_entries);
  op_data->value_cache_buffer->SetNumEntries(layer_index, current_num_entries);

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
