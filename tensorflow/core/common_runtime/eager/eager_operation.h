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

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class EagerOperation {
 public:
  explicit EagerOperation(tensorflow::EagerContext* ctx) : ctx_(*ctx) {}
  ~EagerOperation() {
    for (tensorflow::TensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  // An EagerOperation object can be reused for a different op by calling
  // Clear(), and then Reset(...) with the same arguments that would have
  // been provided to the constructor.
  void Clear() {
    for (tensorflow::TensorHandle* h : inputs_) {
      h->Unref();
    }
    inputs_.clear();
    ClearInferenceState();
  }

  tensorflow::Status Reset(const char* op, const char* raw_device_name,
                           bool remote, EagerExecutor* executor,
                           const absl::optional<EagerRemoteFunctionParams>
                               remote_func_params = absl::nullopt);

  bool is_function() const { return is_function_; }

  tensorflow::EagerContext& EagerContext() { return ctx_; }

  tensorflow::AttrBuilder* MutableAttrs() { return &attrs_; }
  const tensorflow::AttrBuilder& Attrs() const { return attrs_; }
  const tensorflow::OpDef* OpDef() const { return op_def_; }

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

  const tensorflow::string& Name() const { return attrs_.op_name(); }
  const tensorflow::AttrTypeMap* AttrTypes() const { return attr_types_; }

  tensorflow::Device* Device() const { return device_; }
  void SetDevice(tensorflow::Device* device) {
    device_ = device;
    raw_device_name_.clear();
    device_name_ = device->name();
    device_parsed_name_ = device->parsed_name();
  }

  const string& GetDeviceName() const { return device_name_; }
  const DeviceNameUtils::ParsedName& GetDeviceParsedName() const {
    return device_parsed_name_;
  }
  tensorflow::Status SetDeviceName(const char* device,
                                   const bool reset = false);

  // Indicates whether the op is assigned to a device that is local to the
  // current host.
  bool IsLocal() const;

  void SetUseXla(bool use_xla) { use_xla_ = use_xla; }

  CancellationManager* GetCancellationManager() const {
    return cancellation_manager_;
  }
  void SetCancellationManager(CancellationManager* cancellation_manager) {
    cancellation_manager_ = cancellation_manager;
  }

  EagerExecutor& Executor() { return *executor_; }

  string DebugString() const;

  const absl::optional<EagerRemoteFunctionParams>& remote_func_params() const {
    return remote_func_params_;
  }

#ifdef TENSORFLOW_MEM_DEBUG
  const char* op_name() const { return op_name_; }
  const char* op_name_ = nullptr;
#endif

  Status MaybeInferSingleInputAttrs(tensorflow::TensorHandle* handle);
  Status InferInputListAttrs(int num_inputs);

 private:
  void ClearInferenceState() {
    op_def_ = nullptr;
    inference_arg_idx_ = 0;
    inference_attrs_.clear_no_resize();
  }
  void InferSingleTypeInputListAttrs(const tensorflow::OpDef::ArgDef& input_def,
                                     const tensorflow::DataType dtype,
                                     int num_inputs);
  void InferMixedTypeInputListAttrs(
      const tensorflow::OpDef::ArgDef& input_def,
      const std::vector<tensorflow::DataType>& dtypes);

  tensorflow::EagerContext& ctx_;
  tensorflow::AttrBuilder attrs_;
  const tensorflow::AttrTypeMap* attr_types_;
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4> inputs_;
  tensorflow::Device* device_;
  string raw_device_name_;
  string device_name_;
  DeviceNameUtils::ParsedName device_parsed_name_;
  bool use_xla_ = false;
  bool is_function_;  // Conceptually const, but can't be because of Reset
  CancellationManager* cancellation_manager_ = nullptr;  // Not owned.
  EagerExecutor* executor_;                              // Not owned.
  absl::optional<EagerRemoteFunctionParams> remote_func_params_;

  // Inference information
  const tensorflow::OpDef* op_def_;  // op definition from protobuf
  int inference_arg_idx_;  // arg definition index for the next input to be
                           // added
  tensorflow::gtl::FlatSet<std::string>
      inference_attrs_;  // attributes inferred so far
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
