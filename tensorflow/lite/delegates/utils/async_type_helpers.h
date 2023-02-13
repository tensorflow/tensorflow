/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_ASYNC_TYPE_HELPERS_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_ASYNC_TYPE_HELPERS_H_

#include <memory>
#include <optional>

#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/types.h"

namespace tflite::delegates::utils {

constexpr char kBufferTypeAHardwareBufferBlob[] = "ahardware_buffer_blob";
constexpr char kSyncTypeSyncFenceFd[] = "sync_fence_fd";

// RAII wrapper of TfLiteAttributeMap.
using ScopedTfLiteAttrMap =
    std::unique_ptr<TfLiteAttributeMap, decltype(&TfLiteAttributeMapDelete)>;

inline ScopedTfLiteAttrMap CreateScopedTfLiteAttrMap(TfLiteAttrMapType type) {
  return ScopedTfLiteAttrMap(TfLiteAttributeMapCreate(type),
                             TfLiteAttributeMapDelete);
}

// RAII wrapper of TfLiteBackendBuffer.
using ScopedTfLiteBackendBuffer =
    std::unique_ptr<TfLiteBackendBuffer, decltype(&TfLiteBackendBufferDelete)>;

inline ScopedTfLiteBackendBuffer CreateScopedTfLiteBackendBuffer() {
  return ScopedTfLiteBackendBuffer(TfLiteBackendBufferCreate(),
                                   TfLiteBackendBufferDelete);
}

// RAII wrapper of TfLiteSynchronization.
using ScopedTfLiteSynchronization =
    std::unique_ptr<TfLiteSynchronization,
                    decltype(&TfLiteSynchronizationDelete)>;

inline ScopedTfLiteSynchronization CreateScopedTfLiteSynchronization() {
  return ScopedTfLiteSynchronization(TfLiteSynchronizationCreate(),
                                     TfLiteSynchronizationDelete);
}

enum class BufferType { kUnknown, kAHardwareBufferBlob };

struct BufferAttributes {
  std::optional<BufferType> buffer_type;
  std::optional<size_t> alignment;
  std::optional<size_t> padding;
  std::optional<size_t> offset;
  std::optional<size_t> size;
};

// Converts TfLiteAttributeMap to BufferAttributes.
// Crashes if the input attr map is not a buffer attr map.
BufferAttributes ReadBufferAttrs(const TfLiteAttributeMap* attr_map);
BufferAttributes ReadBufferAttrs(const ScopedTfLiteAttrMap& attr_map);

// Converts BufferAttributes to TfLiteAttributeMap.
// Crashes if the input mutable attr map is not a buffer attr map.
void WriteBufferAttrs(const BufferAttributes& attrs,
                      TfLiteAttributeMap* attr_map);
ScopedTfLiteAttrMap WriteBufferAttrs(const BufferAttributes& attrs);

enum class SyncType { kUnknown, kNoSyncObj, kSyncFenceFd };

struct SyncAttributes {
  std::optional<SyncType> sync_type;
};

// Converts TfLiteAttributeMap to SyncAttributes.
// Crashes if the input attr map is not a sync attr map.
SyncAttributes ReadSyncAttrs(const TfLiteAttributeMap* attr_map);
SyncAttributes ReadSyncAttrs(const ScopedTfLiteAttrMap& attr_map);

// Converts SyncAttributes to TfLiteAttributeMap.
// Crashes if the input mutable attr map is not a sync attr map.
void WriteSyncAttrs(const SyncAttributes& attrs, TfLiteAttributeMap* attr_map);
ScopedTfLiteAttrMap WriteSyncAttrs(const SyncAttributes& attrs);

}  // namespace tflite::delegates::utils

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_ASYNC_TYPE_HELPERS_H_
