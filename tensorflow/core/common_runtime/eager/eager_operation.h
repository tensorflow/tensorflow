/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
class EagerOperation {
 public:
  EagerOperation(tensorflow::EagerContext* ctx, const char* op,
                 bool is_function, const tensorflow::AttrTypeMap* t,
                 EagerExecutor* executor = nullptr)
      : ctx_(ctx),
        name_(op),
        attrs_(op),
        attr_types_(t),
        device_(nullptr),
        is_function_(is_function),
        executor_(executor ? *executor : ctx->Executor()) {}

  ~EagerOperation() {
    for (tensorflow::TensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  bool is_function() const { return is_function_; }

  tensorflow::EagerContext* EagerContext() { return ctx_; }

  tensorflow::AttrBuilder* MutableAttrs() { return &attrs_; }
  const tensorflow::AttrBuilder& Attrs() const { return attrs_; }

  const tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4>& Inputs()
      const {
    return inputs_;
  }
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4>*
  MutableInputs() {
    return &inputs_;
  }

  void AddInput(tensorflow::TensorHandle* h);
  void UpdateInput(int i, tensorflow::TensorHandle* h);
  void ConsumeInput(tensorflow::TensorHandle* h);

  const tensorflow::string& Name() const { return name_; }
  const tensorflow::AttrTypeMap* AttrTypes() const { return attr_types_; }

  tensorflow::Device* Device() const { return device_; }
  void SetDevice(tensorflow::Device* device) {
    device_ = device;
    device_name_ = device->parsed_name();
  }
  const DeviceNameUtils::ParsedName& GetDeviceName() const {
    return device_name_;
  }
  tensorflow::Status SetDeviceName(const char* device);

  void SetUseXla(bool use_xla) { use_xla_ = use_xla; }

  CancellationManager* GetCancellationManager() const {
    return cancellation_manager_;
  }
  void SetCancellationManager(CancellationManager* cancellation_manager) {
    cancellation_manager_ = cancellation_manager;
  }

  EagerExecutor& Executor() { return executor_; }

  string DebugString() const;

 private:
  tensorflow::EagerContext* ctx_;  // Must outlive the EagerOperation.
  const tensorflow::string name_;
  tensorflow::AttrBuilder attrs_;
  const tensorflow::AttrTypeMap* attr_types_;
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4> inputs_;
  tensorflow::Device* device_;
  DeviceNameUtils::ParsedName device_name_;
  bool use_xla_ = false;
  const bool is_function_;
  CancellationManager* cancellation_manager_ = nullptr;  // Not owned.
  EagerExecutor& executor_;                              // Not owned.
};

inline void EagerOperation::AddInput(tensorflow::TensorHandle* h) {
  h->Ref();
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}

inline void EagerOperation::UpdateInput(int i, tensorflow::TensorHandle* h) {
  tensorflow::TensorHandle** slot = &inputs_[i];
  tensorflow::TensorHandle* existing = *slot;
  if (existing != h) {
    h->Ref();
    existing->Unref();
    *slot = h;  // Update inputs_[i] to h
  }
}

inline void EagerOperation::ConsumeInput(tensorflow::TensorHandle* h) {
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
