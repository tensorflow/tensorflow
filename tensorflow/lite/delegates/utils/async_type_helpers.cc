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
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

#include <cstring>
#include <string_view>

#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/delegates/utils/ret_macros.h"

// TODO(b/191883048): Cleanup this file when refactoring the attribute map
// accessors.

namespace tflite::delegates::utils {
namespace {

BufferType BufferTypeFromString(const char* buffer_type) {
  if (buffer_type == nullptr) {
    return BufferType::kUnknown;
  }
  if (std::strcmp(buffer_type, kBufferTypeAHardwareBufferBlob) == 0) {
    return BufferType::kAHardwareBufferBlob;
  }
  return BufferType::kUnknown;
}

const char* StringFromBufferType(BufferType buffer_type) {
  switch (buffer_type) {
    case BufferType::kAHardwareBufferBlob:
      return kBufferTypeAHardwareBufferBlob;
    case BufferType::kUnknown:
      return "<unknown buffer type>";
  }
}

SyncType SyncTypeFromString(const char* sync_type) {
  if (sync_type == nullptr) {
    return SyncType::kUnknown;
  }
  if (std::strcmp(sync_type, kTfLiteSyncTypeNoSyncObj) == 0) {
    return SyncType::kNoSyncObj;
  }
  if (std::strcmp(sync_type, kSyncTypeSyncFenceFd) == 0) {
    return SyncType::kSyncFenceFd;
  }
  return SyncType::kUnknown;
}

const char* StringFromSyncType(SyncType sync_type) {
  switch (sync_type) {
    case SyncType::kNoSyncObj:
      return kTfLiteSyncTypeNoSyncObj;
    case SyncType::kSyncFenceFd:
      return kSyncTypeSyncFenceFd;
    case SyncType::kUnknown:
      return "<unknown sync type>";
  }
}

}  // namespace

BufferAttributes ReadBufferAttrs(const TfLiteAttributeMap* attr_map) {
  TFLITE_ABORT_CHECK(TfLiteAttributeMapIsBufferAttributeMap(attr_map),
                     "");  // Crash OK
  BufferAttributes attrs{};
  const char* buffer_type = nullptr;
  if (TfLiteAttributeMapGetStringAttr(
          attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyResourceTypeName),
          &buffer_type)) {
    attrs.buffer_type = BufferTypeFromString(buffer_type);
  }
  size_t alignment = 0;
  if (TfLiteAttributeMapGetSizeTAttr(
          attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyAlignment),
          &alignment)) {
    attrs.alignment = alignment;
  }
  size_t padding = 0;
  if (TfLiteAttributeMapGetSizeTAttr(
          attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyPadding),
          &padding)) {
    attrs.padding = padding;
  }
  size_t offset = 0;
  if (TfLiteAttributeMapGetSizeTAttr(
          attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyOffset),
          &offset)) {
    attrs.offset = offset;
  }
  size_t size = 0;
  if (TfLiteAttributeMapGetSizeTAttr(
          attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeySize), &size)) {
    attrs.size = size;
  }
  return attrs;
}

BufferAttributes ReadBufferAttrs(const ScopedTfLiteAttrMap& attr_map) {
  return ReadBufferAttrs(attr_map.get());
}

void WriteBufferAttrs(const BufferAttributes& attrs,
                      TfLiteAttributeMap* attr_map) {
  TFLITE_ABORT_CHECK(TfLiteAttributeMapIsBufferAttributeMap(attr_map),
                     "");  // Crash OK
  if (attrs.buffer_type) {
    TfLiteAttributeMapSetStringAttr(
        attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyResourceTypeName),
        StringFromBufferType(attrs.buffer_type.value()));
  }
  if (attrs.alignment) {
    TfLiteAttributeMapSetSizeTAttr(
        attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyAlignment),
        attrs.alignment.value());
  }
  if (attrs.padding) {
    TfLiteAttributeMapSetSizeTAttr(
        attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyPadding),
        attrs.padding.value());
  }
  if (attrs.offset) {
    TfLiteAttributeMapSetSizeTAttr(
        attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeyOffset),
        attrs.offset.value());
  }
  if (attrs.size) {
    TfLiteAttributeMapSetSizeTAttr(
        attr_map, static_cast<uint32_t>(kTfLiteBufferAttrKeySize),
        attrs.size.value());
  }
}

ScopedTfLiteAttrMap WriteBufferAttrs(const BufferAttributes& attrs) {
  auto attr_map = CreateScopedTfLiteAttrMap(kTfLiteAttrMapTypeBuffer);
  WriteBufferAttrs(attrs, attr_map.get());
  return attr_map;
}

SyncAttributes ReadSyncAttrs(const TfLiteAttributeMap* attr_map) {
  TFLITE_ABORT_CHECK(TfLiteAttributeMapIsSyncAttributeMap(attr_map),
                     "");  // Crash OK
  SyncAttributes attrs{};
  const char* sync_type = nullptr;
  if (TfLiteAttributeMapGetStringAttr(
          attr_map,
          static_cast<uint32_t>(kTfLiteSynchronizationAttrKeyObjectTypeName),
          &sync_type)) {
    attrs.sync_type = SyncTypeFromString(sync_type);
  }
  return attrs;
}

SyncAttributes ReadSyncAttrs(const ScopedTfLiteAttrMap& attr_map) {
  return ReadSyncAttrs(attr_map.get());
}

void WriteSyncAttrs(const SyncAttributes& attrs, TfLiteAttributeMap* attr_map) {
  TFLITE_ABORT_CHECK(TfLiteAttributeMapIsSyncAttributeMap(attr_map),
                     "");  // Crash OK
  if (attrs.sync_type) {
    TfLiteAttributeMapSetStringAttr(
        attr_map,
        static_cast<uint32_t>(kTfLiteSynchronizationAttrKeyObjectTypeName),
        StringFromSyncType(attrs.sync_type.value()));
  }
}

ScopedTfLiteAttrMap WriteSyncAttrs(const SyncAttributes& attrs) {
  auto attr_map = CreateScopedTfLiteAttrMap(kTfLiteAttrMapTypeSync);
  WriteSyncAttrs(attrs, attr_map.get());
  return attr_map;
}

}  // namespace tflite::delegates::utils
