/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_KEYS_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_KEYS_H_

#include <cstdint>

const uint32_t kTfLiteAttributeKeyStart = 0;
const uint32_t kTfLiteAttributeKeyEnd = UINT32_MAX;

// General buffer attribute keys that are recognizable by TFLite.
enum class TfLiteBufferAttributeKey : uint32_t {
  kAttributeKeyStart = kTfLiteAttributeKeyStart,
  // Backing buffer resource. const char*
  // e.g. "AHardwareBuffer".
  kBufferResourceTypeName = 1,
  // Buffer alignment, size_t
  kAlignment = 2,
  // Buffer padding, size_t
  kPadding = 3,
  // Buffer offset, size_t
  kOffset = 4,
  // Buffer size (padded size if applicable), size_t
  kSize = 5,

  ATTRIBUTE_KEY_END = kTfLiteAttributeKeyEnd,
};

// General synchronization attribute keys that are recognizable by TFLite.
enum class TfLiteSyncAttributeKey : uint32_t {
  kAttributeKeyStart = kTfLiteAttributeKeyStart,
  // Synchronization type name. const char*
  // e.g. "ANeuralNetworksEvent"
  kSyncObjectTypeName = 1,

  ATTRIBUTE_KEY_END = kTfLiteAttributeKeyEnd,
};

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_ATTRIBUTE_KEYS_H_
