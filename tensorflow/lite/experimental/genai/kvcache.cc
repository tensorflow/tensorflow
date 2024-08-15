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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/cache_buffer.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace llm {

static const int kPositionTensor = 0;
static const int kKeyTensor = 1;
static const int kValueTensor = 2;
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
  int first_slot_index;
  // Pointers to the key and value cache buffers that this Op doesn't own
  // (and therefore does not free on destruction of this Op).
  resource::CacheBuffer* key_cache_buffer;
  resource::CacheBuffer* value_cache_buffer;
  bool is_initialized;
  uint8_t* key_cache_ptr;
  uint8_t* value_cache_ptr;
};

void* KVCacheInit(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  // TODO(b/333891673) Reset this value via ClearCaches in
  // InternalBackendContext.
  op_data->max_num_entries = -1;
  op_data->num_layers = -1;
  op_data->layer_index = -1;
  op_data->first_slot_index = -1;
  op_data->key_cache_buffer = nullptr;
  op_data->value_cache_buffer = nullptr;
  op_data->is_initialized = false;
  op_data->key_cache_ptr = nullptr;
  op_data->value_cache_ptr = nullptr;
  return op_data;
}

TfLiteStatus KVCachePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
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
    op_data->first_slot_index = 0;
    op_data->is_initialized = true;
  }

  // Prepare the inputs.
  const TfLiteTensor* position;
  const TfLiteTensor* key;
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kPositionTensor, &position));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kKeyTensor, &key));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));

  TF_LITE_ENSURE_EQ(context, position->type, kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, value->type, kTfLiteFloat32);
  // Ensure Positions correspond to KV sequence length.
  TF_LITE_ENSURE(context, NumDimensions(position) == 1);
  TF_LITE_ENSURE(
      context, GetTensorShape(position).Dims(0) == GetTensorShape(key).Dims(1));
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
  // Custom data pointer to the resource cache buffer.
  kfull->allocation_type = kTfLiteCustom;
  vfull->allocation_type = kTfLiteCustom;

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

  // Get the pointer to the tensor for our buffer storage.
  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();

  if (resources.count(KVCACHE_KEY_RESOURCE) == 0) {
    auto* cbuffer = new resource::CacheBuffer();
    cbuffer->Initialize(*kcache_buffer_dims);
    resources.emplace(KVCACHE_KEY_RESOURCE, cbuffer);
    op_data->key_cache_buffer = cbuffer;
  } else {
    resource::ResourceBase* resourcePtr =
        resources.at(KVCACHE_KEY_RESOURCE).get();
    resource::CacheBuffer* cbuffer = (resource::CacheBuffer*)(resourcePtr);
    op_data->key_cache_buffer = cbuffer;
  }
  if (resources.count(KVCACHE_VALUE_RESOURCE) == 0) {
    auto* cbuffer = new resource::CacheBuffer();
    cbuffer->Initialize(*vcache_buffer_dims);
    resources.emplace(KVCACHE_VALUE_RESOURCE, cbuffer);
    op_data->value_cache_buffer = cbuffer;
  } else {
    resource::ResourceBase* resourcePtr =
        resources.at(KVCACHE_VALUE_RESOURCE).get();
    resource::CacheBuffer* cbuffer = (resource::CacheBuffer*)(resourcePtr);
    op_data->value_cache_buffer = cbuffer;
  }

  // Get the pointers to the individual caches for a layer.
  RuntimeShape shape(GetTensorShape(key));
  const int elements_in_one_entry = shape.Dims(2) * shape.Dims(3);
  const int elements_in_one_block =
      op_data->max_num_entries * elements_in_one_entry;
  uint8_t* k_ptr =
      reinterpret_cast<uint8_t*>(op_data->key_cache_buffer->GetBuffer());
  uint8_t* v_ptr =
      reinterpret_cast<uint8_t*>(op_data->value_cache_buffer->GetBuffer());
  k_ptr = k_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;
  v_ptr = v_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;

  size_t kcache_dims_flatsize = kcache_dims->data[0] * kcache_dims->data[1] *
                                kcache_dims->data[2] * kcache_dims->data[3];
  size_t vcache_dims_flatsize = vcache_dims->data[0] * vcache_dims->data[1] *
                                vcache_dims->data[2] * vcache_dims->data[3];
  RuntimeShape kfull_shape(GetTensorShape(kfull));
  RuntimeShape vfull_shape(GetTensorShape(vfull));
  // Some testing utils don't fully set the output tensor shape
  if (kfull_shape.FlatSize() > 1 && vfull_shape.FlatSize() > 1) {
    TF_LITE_ENSURE_EQ(context, kfull_shape.FlatSize(), kcache_dims_flatsize);
    TF_LITE_ENSURE_EQ(context, vfull_shape.FlatSize(), vcache_dims_flatsize);
  }
  TF_LITE_ENSURE_EQ(context, elements_in_one_block, kcache_dims_flatsize);
  TF_LITE_ENSURE_EQ(context, elements_in_one_block, vcache_dims_flatsize);

  kfull->data.data = k_ptr;
  vfull->data.data = v_ptr;
  op_data->key_cache_ptr = k_ptr;
  op_data->value_cache_ptr = v_ptr;

  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, kfull, kcache_dims));
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, vfull, vcache_dims));

  TfLiteIntArrayFree(kcache_buffer_dims);
  TfLiteIntArrayFree(vcache_buffer_dims);
  return kTfLiteOk;
}

void KVCacheFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus KVCacheEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* position;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kPositionTensor, &position));
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
  const int layer_index = op_data->layer_index;
  const int64_t max_num_entries = op_data->max_num_entries;
  int current_num_entries =
      op_data->key_cache_buffer->GetNumEntries(layer_index);

  // Compute some constants for various pieces of the cache.
  RuntimeShape shape(GetTensorShape(key));
  const int64_t num_slots_needed = shape.Dims(1);
  const int elements_in_one_entry = shape.Dims(2) * shape.Dims(3);
  const int elements_in_one_block =
      op_data->max_num_entries * elements_in_one_entry;
  const int64_t num_bytes_per_tensor = sizeof(float) * elements_in_one_entry;

  // Get the pointers to the individual caches for a layer.
  uint8_t* k_ptr = reinterpret_cast<uint8_t*>(key_cache_ptr);
  uint8_t* v_ptr = reinterpret_cast<uint8_t*>(value_cache_ptr);
  k_ptr = k_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;
  v_ptr = v_ptr + sizeof(float) * op_data->layer_index * elements_in_one_block;

  // 0. Ensure output ptr is pointing to the cache data
  TF_LITE_ENSURE_EQ(context, k_ptr, op_data->key_cache_ptr);
  TF_LITE_ENSURE_EQ(context, v_ptr, op_data->value_cache_ptr);
  TF_LITE_ENSURE_EQ(context, k_ptr, kfull->data.data);
  TF_LITE_ENSURE_EQ(context, v_ptr, vfull->data.data);

  // 1. Determine which slots the inputs take up, and which slots are in the
  //    existing span of the cache.

  // Compute the span of the inputs.
  const int64_t input_first_idx = position->data.i64[0];
  const int64_t input_last_idx = input_first_idx + num_slots_needed - 1;

  // Compute the span of the cache.
  const int64_t cache_first_slot_idx = op_data->first_slot_index;
  const int64_t cache_last_slot_idx =
      cache_first_slot_idx + op_data->max_num_entries - 1;

  // Compute if a shift is needed.
  const int slots_to_shift = std::min(
      std::max(static_cast<int64_t>(0), input_last_idx - cache_last_slot_idx),
      max_num_entries);

  // These values determine how we will write to the output tensor:
  // first_slot := the first cache entry that we will write to in the output
  int64_t first_slot = input_first_idx - op_data->first_slot_index;
  if (first_slot < 0) {
    TF_LITE_KERNEL_LOG(
        context,
        "Can not specify a position before this cache's first slot index of %d",
        op_data->first_slot_index);
    return kTfLiteError;
  }

  // byte_offset_for_output := the byte offset for the first slot.
  int64_t byte_offset_for_output = first_slot * num_bytes_per_tensor;
  // num_slots_for_output := the number of slots we write in the output
  int64_t num_slots_for_output = num_slots_needed;

  // 3. If we need more slots, make room in the cache by writing over oldest
  //    entries.
  if (slots_to_shift > 0 && slots_to_shift < max_num_entries) {
    // If we are shifting the cache, we need to start writing from the
    // beginning.
    byte_offset_for_output = 0;
    // And we need to write the entire cache.
    num_slots_for_output = max_num_entries;
    const int bytes_offset =
        sizeof(float) * elements_in_one_entry * slots_to_shift;
    const int size_bytes_to_shift = sizeof(float) * elements_in_one_entry *
                                    (max_num_entries - slots_to_shift);
    // TODO(b/333893996): This is O(cache_size) data motion. Consider optimizing
    // with a circular buffer or similar.
    memmove(k_ptr, k_ptr + bytes_offset, size_bytes_to_shift);
    memmove(v_ptr, v_ptr + bytes_offset, size_bytes_to_shift);
  }

  // Update the first slot this cache now covers.
  op_data->first_slot_index = op_data->first_slot_index + slots_to_shift;

  // Recompute the first slot in case any shifting occurred.
  first_slot = input_first_idx - op_data->first_slot_index;
  const int64_t bytes_offset_for_cache = first_slot * num_bytes_per_tensor;

  // 4. Put the key and value in their respective caches.
  memcpy(k_ptr + bytes_offset_for_cache, key->data.data, key->bytes);
  memcpy(v_ptr + bytes_offset_for_cache, value->data.data, value->bytes);

  // Update counts.
  current_num_entries =
      std::min(first_slot + num_slots_needed, max_num_entries);
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
