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

#include "tensorflow/lite/experimental/resource/cache_buffer.h"

#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace resource {

constexpr char kCacheBufferTensorName[] = "CacheBuffer";

TfLiteStatus CacheBuffer::Initialize(const TfLiteIntArray &shape,
                                     const TfLiteType &type) {
  // Set basic parameters.
  tensor_.name = kCacheBufferTensorName;
  tensor_.allocation_type = kTfLiteDynamic;
  tensor_.type = type;

  // Set the shape and allocate the memory.
  tensor_.dims = TfLiteIntArrayCopy(&shape);
  const size_t num_bytes = TfLiteTypeGetSize(type) * NumElements(&tensor_);
  TfLiteTensorRealloc(num_bytes, &tensor_);

  memset(tensor_.data.raw, 0, tensor_.bytes);
  is_initialized_ = true;
  return kTfLiteOk;
}

size_t CacheBuffer::GetNumEntries() const { return num_entries_; }

void CacheBuffer::SetNumEntries(size_t count) {
  TFLITE_DCHECK(count <= tensor_.dims->data[2]);
  num_entries_ = count;
}

}  // namespace resource
}  // namespace tflite
