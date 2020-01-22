/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_HELPERS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_HELPERS_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Returns the next pointer address aligned to the given alignment.
uint8_t* AlignPointerUp(uint8_t* data, size_t alignment);

// Returns the previous pointer address aligned to the given alignment.
uint8_t* AlignPointerDown(uint8_t* data, size_t alignment);

// Returns an increased size that's a multiple of alignment.
size_t AlignSizeUp(size_t size, size_t alignment);

// Returns size in bytes for a given TfLiteType.
TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size,
                              ErrorReporter* reporter);

// How many bytes are needed to hold a tensor's contents.
TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size,
                                    ErrorReporter* error_reporter);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_HELPERS_H_
