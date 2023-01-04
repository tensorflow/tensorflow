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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_TASK_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_ASYNC_TASK_INTERNAL_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/common.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

// Forward declaration
namespace tflite::async {
class ExecutionTask;
}  // namespace tflite::async

// TfLiteExecutionTask object holds the mapping from tensor index to
// the backend buffer and sync object tied to that tensor.
// It also holds the event handle that represents a specific scheduled
// async execution.
// The AsyncSignatureRunner that creates the task should out-live the life time
// of the TfLiteExecutionTask.
struct TfLiteExecutionTask {
  TfLiteExecutionTask();
  std::unique_ptr<tflite::async::ExecutionTask> task;
};

namespace tflite {
namespace async {

// Implementation of TfLiteExecutionTask.
// This class is not thread safe.
class ExecutionTask {
 public:
  // Returns the buffer handle for input / output tensor `name`.
  // If there's tensor `name` is not found, returns kTfLiteNullBufferHandle.
  TfLiteBufferHandle GetBufferHandle(TfLiteIoType io_type,
                                     const char* name) const;
  // Same as GetBufferHandle above, but uses tensor index as key.
  TfLiteBufferHandle GetBufferHandle(int tensor_index) const;

  // Sets the buffer handle for input / output `name`.
  // If there's tensor `name` is not found, do nothing.
  TfLiteStatus SetBufferHandle(TfLiteIoType io_type, const char* name,
                               TfLiteBufferHandle handle);

  // Returns the TfLiteSynchronization for input / output tensor `name`.
  // If there's tensor `name` is not found, returns nullptr.
  TfLiteSynchronization* GetSynchronization(TfLiteIoType io_type,
                                            const char* name) const;
  // Same as GetSynchronization above, but uses tensor index as key.
  TfLiteSynchronization* GetSynchronization(int tensor_index) const;

  // Sets the TfLiteSynchronization for input / output tensor `name`.
  // If there's tensor `name` is not found, do nothing.
  TfLiteStatus SetSynchronization(TfLiteIoType io_type, const char* name,
                                  TfLiteSynchronization* sync);

  using TensorNameMapT = std::map<std::string, uint32_t>;

  // Sets the mapping from signature input name to tensor index.
  void SetInputNameMap(const TensorNameMapT* input_name_to_idx) {
    input_name_to_idx_ = input_name_to_idx;
  }

  // Sets the mapping from signature output name to tensor index.
  void SetOutputNameMap(const TensorNameMapT* output_name_to_idx) {
    output_name_to_idx_ = output_name_to_idx;
  }

  // Returns the status of this task.
  // True if the task has been scheduled, false if idle.
  bool Scheduled() const { return scheduled_.load(); }

  // Exchanges the status of this task. Whether it's been scheduled or in idle
  // state.
  // Returns the previous value of the task.
  bool SetScheduled(bool scheduled) { return scheduled_.exchange(scheduled); }

  // Returns the latest status of this task.
  // Thread safe.
  TfLiteStatus Status() const { return status_.load(); }

  // Sets the status code for this task.
  // Thread safe.
  void SetStatus(TfLiteStatus status) { status_.store(status); }

  // Sets the delegate execution data for this task.
  void SetDelegateExecutionData(TfLiteAsyncKernel* kernel, void* data) {
    data_ = data;
  }

  // Returns the delegate execution data for this task.
  // Returns nullptr if not set.
  void* GetDelegateExecutionData(TfLiteAsyncKernel* kernel) const {
    return data_;
  }

 private:
  struct IOData {
    TfLiteBufferHandle buf = kTfLiteNullBufferHandle;
    TfLiteSynchronization* sync = nullptr;
  };

  // Finds the tensor index for input / output name.
  // Returns false if the tensor is not found.
  bool GetTensorIdx(TfLiteIoType io_type, const char* name, int* idx) const;

  // Mapping from tensor index to buffer handle and sync object ptr.
  // Set by the application.
  std::map<int, IOData> io_data_;

  // The status of the task. Whether the task has been scheduled or not.
  // The bit is set when calling InvokeAsync to this task, and resets on Wait.
  std::atomic_bool scheduled_ = false;

  // The latest status of this task. Default value will be kTfLiteOk.
  std::atomic<TfLiteStatus> status_ = kTfLiteOk;

  // Mapping from signature name to tensor index.
  // Not owned. Set and owned by AsyncSignatureRunner.
  const TensorNameMapT* input_name_to_idx_ = nullptr;
  const TensorNameMapT* output_name_to_idx_ = nullptr;

  // Delegate owned data.
  // NOTE: Currently we only support one delegate. If we are to support multiple
  // backends, we might need to change this to a map.
  void* data_ = nullptr;
};

}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_TASK_INTERNAL_H_
