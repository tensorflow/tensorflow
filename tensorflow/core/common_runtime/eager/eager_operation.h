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

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class EagerOperation : public AbstractOperationInterface {
 public:
  explicit EagerOperation(tensorflow::EagerContext* ctx) : ctx_(*ctx) {}
  ~EagerOperation() override {
    for (TensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  void Release() override { delete this; }

  void Clear() override;
  Status Reset(const char* op, const char* raw_device_name) override {
    return Reset(op, raw_device_name, false, nullptr);
  }

  const string& Name() const override { return attrs_.op_name(); }
  const string& DeviceName() const override;
  Status SetDeviceName(const char* name) override;

  Status AddInput(AbstractTensorHandleInterface* input) override;
  Status AddInputList(
      absl::Span<AbstractTensorHandleInterface*> inputs) override;
  Status Execute(absl::Span<AbstractTensorHandleInterface*> retvals,
                 int* num_retvals) override;
  const tensorflow::OpDef* OpDef() const override { return op_def_; };

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name, DataType value) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperationInterface* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* data,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name, const DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperationInterface*> values) override;

  Status InputLength(const char* input_name, int* length) override;
  Status OutputLength(const char* output_name, int* length) override;

  Status SetUseXla(bool enable) override;

  Status Reset(const char* op, const char* raw_device_name, bool remote,
               EagerExecutor* executor,
               const absl::optional<EagerRemoteFunctionParams>
                   remote_func_params = absl::nullopt);

  bool is_function() const { return is_function_; }
  bool colocation_exempt() const { return colocation_exempt_; }

  tensorflow::EagerContext& EagerContext() { return ctx_; }
  const tensorflow::EagerContext& EagerContext() const { return ctx_; }

  AttrBuilder* MutableAttrs() { return &attrs_; }
  const AttrBuilder& Attrs() const { return attrs_; }

  const absl::InlinedVector<TensorHandle*, 4>& Inputs() const {
    return inputs_;
  }
  absl::InlinedVector<TensorHandle*, 4>* MutableInputs() { return &inputs_; }

  void AddInput(TensorHandle* h);
  void UpdateInput(int i, TensorHandle* h);

  const AttrTypeMap* AttrTypes() const { return attr_types_; }

  // Like TensorHandles, EagerOperations may be placed either on a virtual
  // CustomDevice or on a physical Device.
  absl::variant<tensorflow::Device*, tensorflow::CustomDevice*> Device() const {
    return device_;
  }

  void SetDevice(tensorflow::Device* device) {
    device_ = device;
    raw_device_name_.clear();
    device_name_ = device->name();
    device_parsed_name_ = device->parsed_name();
  }

  void SetDevice(tensorflow::CustomDevice* device) {
    device_ = device;
    raw_device_name_.clear();
    device_name_ = device->name();
    DeviceNameUtils::ParseFullName(device_name_, &device_parsed_name_);
  }

  const string& GetDeviceName() const { return device_name_; }
  const DeviceNameUtils::ParsedName& GetDeviceParsedName() const {
    return device_parsed_name_;
  }

  // Indicates whether the op is assigned to a device that is local to the
  // current host.
  bool IsLocal() const;

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

  // Op name recorded for memory debugging purpose.
  const char* op_name() const { return op_name_; }
  const char* op_name_ = nullptr;

  Status MaybeInferSingleInputAttrs(TensorHandle* handle);
  Status InferInputListAttrs(int num_inputs);

 private:
  const tensorflow::OpDef* GetOpDef(Status* status);

  void ClearInferenceState() {
    op_def_ = nullptr;
    inference_arg_idx_ = 0;
    inference_attrs_.clear_no_resize();
  }
  void InferSingleTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                     const DataType dtype, int num_inputs);
  void InferMixedTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                    const std::vector<DataType>& dtypes);

  tensorflow::EagerContext& ctx_;
  AttrBuilder attrs_;
  const AttrTypeMap* attr_types_;
  absl::InlinedVector<TensorHandle*, 4> inputs_;
  absl::variant<tensorflow::Device*, tensorflow::CustomDevice*> device_;
  string raw_device_name_;
  string device_name_;
  DeviceNameUtils::ParsedName device_parsed_name_;
  bool use_xla_ = false;
  bool is_function_;  // Conceptually const, but can't be because of Reset
  bool colocation_exempt_;
  CancellationManager* cancellation_manager_ = nullptr;  // Not owned.
  EagerExecutor* executor_;                              // Not owned.
  absl::optional<EagerRemoteFunctionParams> remote_func_params_;

  // Inference information
  const tensorflow::OpDef* op_def_;  // op definition from protobuf
  int inference_arg_idx_;  // arg definition index for the next input to be
                           // added
  gtl::FlatSet<std::string> inference_attrs_;  // attributes inferred so far
};

inline void EagerOperation::AddInput(TensorHandle* h) {
  h->Ref();
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}

inline void EagerOperation::UpdateInput(int i, TensorHandle* h) {
  TensorHandle** slot = &inputs_[i];
  TensorHandle* existing = *slot;
  if (existing != h) {
    h->Ref();
    existing->Unref();
    *slot = h;  // Update inputs_[i] to h
  }
}

inline EagerOperation* OperationFromInterface(
    AbstractOperationInterface* operation) {
  return down_cast<EagerOperation*>(operation);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
