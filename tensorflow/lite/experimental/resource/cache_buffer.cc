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

TfLiteStatus CacheBuffer::Initialize(const TfLiteIntArray& shape) {
  // Set the dims and allocate the memory.
  dims_ = TfLiteIntArrayCopy(&shape);
  const size_t buf_size = NumElements(&shape);
  buffer_.reset(new float[buf_size]);
  memset(buffer_.get(), 0, sizeof(float) * buf_size);

  num_entries_.reset(new size_t[shape.data[1]]);
  memset(num_entries_.get(), 0, sizeof(size_t) * shape.data[1]);
  is_initialized_ = true;
  return kTfLiteOk;
}

size_t CacheBuffer::GetSize() { return sizeof(float) * NumElements(dims_); }

size_t CacheBuffer::GetNumEntries(int idx) const { return num_entries_[idx]; }

CacheBuffer::~CacheBuffer() { TfLiteIntArrayFree(dims_); }

float* CacheBuffer::GetBuffer() { return buffer_.get(); }

void CacheBuffer::SetNumEntries(int idx, size_t count) {
  TFLITE_DCHECK(count <= dims_->data[2]);
  num_entries_[idx] = count;
}

}  // namespace resource
}  // namespace tflite
