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
#include "tensorflow/lite/core/async/task_internal.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace async {

bool ExecutionTask::GetTensorIdx(TfLiteIoType io_type, const char* name,
                                 int* idx) const {
  const std::map<std::string, uint32_t>* map = nullptr;
  if (io_type == kTfLiteIoTypeInput) {
    map = input_name_to_idx_;
  } else {
    map = output_name_to_idx_;
  }
  if (!map) return false;
  if (auto it_idx = map->find(name); it_idx != map->end()) {
    *idx = it_idx->second;
    return true;
  }
  return false;
}

TfLiteBufferHandle ExecutionTask::GetBufferHandle(TfLiteIoType io_type,
                                                  const char* name) const {
  int index = 0;
  if (!GetTensorIdx(io_type, name, &index)) {
    return kTfLiteNullBufferHandle;
  }
  return GetBufferHandle(index);
}

TfLiteBufferHandle ExecutionTask::GetBufferHandle(int tensor_index) const {
  if (auto it = io_data_.find(tensor_index); it != io_data_.end()) {
    return it->second.buf;
  }
  return kTfLiteNullBufferHandle;
}

TfLiteStatus ExecutionTask::SetBufferHandle(TfLiteIoType io_type,
                                            const char* name,
                                            TfLiteBufferHandle handle) {
  int index = 0;
  if (!GetTensorIdx(io_type, name, &index)) {
    return kTfLiteError;
  }
  io_data_[index].buf = handle;
  return kTfLiteOk;
}

TfLiteSynchronization* ExecutionTask::GetSynchronization(
    TfLiteIoType io_type, const char* name) const {
  int index = 0;
  if (!GetTensorIdx(io_type, name, &index)) {
    return nullptr;
  }
  return GetSynchronization(index);
}

TfLiteSynchronization* ExecutionTask::GetSynchronization(
    int tensor_index) const {
  if (auto it = io_data_.find(tensor_index); it != io_data_.end()) {
    return it->second.sync;
  }
  return nullptr;
}

TfLiteStatus ExecutionTask::SetSynchronization(TfLiteIoType io_type,
                                               const char* name,
                                               TfLiteSynchronization* sync) {
  int index = 0;
  if (!GetTensorIdx(io_type, name, &index)) {
    return kTfLiteError;
  }
  io_data_[index].sync = sync;
  return kTfLiteOk;
}

}  // namespace async
}  // namespace tflite

TfLiteExecutionTask::TfLiteExecutionTask() {
  task = std::make_unique<tflite::async::ExecutionTask>();
}
