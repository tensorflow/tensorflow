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
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

class EagerOperation : public ImmediateExecutionOperation {
 public:
  explicit EagerOperation(tensorflow::EagerContext* ctx)
      : ImmediateExecutionOperation(kEager), ctx_(*ctx), is_function_(false) {}
  ~EagerOperation() override {
    for (ImmediateExecutionTensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  void Release() override { delete this; }

  void Clear() override;
  Status Reset(const char* op, const char* raw_device_name) override {
    return Reset(op, raw_device_name, false, nullptr);
  }

  const string& Name() const override { return attrs_.op_name(); }

  const string& DeviceName() const override { return device_name_; }

  ImmediateExecutionContext* GetContext() const override { return &ctx_; }

  const DeviceNameUtils::ParsedName& GetDeviceParsedName() const {
    return device_parsed_name_;
  }

  // Replaces the previous device name with the given one (see
  // AbstractOperation::SetDeviceName for more details).
  //
  // This also resets the internal device pointer, unless the given name refers
  // to a known custom device, in which case the internal device pointer is
  // updated to that device.
  Status SetDeviceName(const char* name) override;

  void SetDevice(VariantDevice device) {
    device_ = device;
    device_name_ = absl::visit(
        [](auto* device) { return device == nullptr ? "" : device->name(); },
        device);
    DeviceNameUtils::ParseFullName(device_name_, &device_parsed_name_);
    // TODO(b/154133594): Due to intricacies of external logic, we can not
    // set this do device_name_ as it would be natural, because we need the
    // next call to SetDeviceName to reset the device pointer.
    last_set_device_name_ = "\177";  // DEL (an invalid value)
  }

  Status SetAttrValue(const char* attr_name, const AttrValue& value);

  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
  Status SetInput(size_t index, ImmediateExecutionTensorHandle* input) override;
  absl::Span<ImmediateExecutionTensorHandle* const> GetInputs() const override;
  bool HasCustomDeviceInput() const override {
    return custom_device_tensor_handles_count_ > 0;
  }
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
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
                         const AbstractOperation* value) override;
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
      absl::Span<const AbstractOperation*> values) override;

  Status InputLength(const char* input_name, int* length) override;
  Status OutputLength(const char* output_name, int* length) override;

  const AbstractOpAttrs* GetOpAttrs() const override;
  void AddAttrs(const AbstractOpAttrs* op_attrs) override;

  void SetStackTrace(ManagedStackTrace stack_trace) override {
    stack_trace_ = stack_trace;
  }

  absl::optional<ManagedStackTrace> GetStackTrace() override {
    return stack_trace_;
  }

  Status Reset(const char* op, const char* device_name, bool remote,
               EagerExecutor* executor,
               const absl::optional<EagerFunctionParams> remote_func_params =
                   absl::nullopt);

  bool is_function() const { return is_function_; }
  bool colocation_exempt() const { return colocation_exempt_; }

  tensorflow::EagerContext& EagerContext() const { return ctx_; }

  AttrBuilder* MutableAttrs() { return &attrs_; }
  const AttrBuilder& Attrs() const { return attrs_; }

  // TensorHandleInputs and MutableTensorHandleInputs first check that all
  // inputs are TensorHandles, i.e. that there are no custom device inputs. They
  // return a bad status otherwise.
  Status TensorHandleInputs(
      const absl::InlinedVector<TensorHandle*, 4>** inputs) const;
  Status MutableTensorHandleInputs(
      absl::InlinedVector<TensorHandle*, 4>** inputs);

  const absl::InlinedVector<ImmediateExecutionTensorHandle*, 4>& Inputs()
      const {
    return inputs_;
  }

  void UpdateInput(int i, TensorHandle* h);

  // This is useful if we want the EagerOperation to point to a different
  // function.
  void UpdateName(const string& name) {
    op_name_ = name.c_str();
    attrs_.set_op_name(name);
  }

  // Like TensorHandles, EagerOperations may be placed either on a virtual
  // CustomDevice or on a physical Device.
  VariantDevice Device() const { return device_; }

  // Indicates whether the op is assigned to a device that is local to the
  // current host.
  bool IsLocal() const;

  CancellationManager* GetCancellationManager() const {
    return cancellation_manager_;
  }
  void SetCancellationManager(
      CancellationManager* cancellation_manager) override {
    cancellation_manager_ = cancellation_manager;
  }

  // Assign step_id value only if op has valid step id.
  // When eager_func_params.has_value() returns true, we can directly overwrite
  // its step id according to Op's step id (if not default value). However, when
  // eager_func_params.has_value() returns false, we need to first create a new
  // EagerFuncParams object for it before assigning step_id; otherwise,
  // directly assigning step_id in this case leaves eager_func_params to be
  // in a weird state where:
  // (1) eager_func_params.has_value() returns false, but
  // (2) eager_func_params->step_id.has_value() returns true.
  void SetStepId(int64_t step_id) override {
    assert(is_function());
    if (step_id != EagerContext::kGlobalRendezvousId) {
      if (eager_func_params_.has_value()) {
        eager_func_params_->step_id = step_id;
      } else {
        eager_func_params_ = EagerFunctionParams{
            kInvalidOpId, /*is_component_function=*/false, step_id};
      }
    } else {
      LOG(WARNING) << "SetStepId() should not receive a gloabl rendezvous id.";
    }
  }

  EagerExecutor& Executor() { return *executor_; }

  string DebugString() const;

  const absl::optional<EagerFunctionParams>& eager_func_params() const {
    return eager_func_params_;
  }

  // Op name recorded for memory debugging purpose.
  const char* op_name() const { return op_name_; }

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
    return ptr->getKind() == kEager;
  }

 private:
  void AddTensorHandle(ImmediateExecutionTensorHandle* h);

  const tensorflow::OpDef* GetOpDef(Status* status);

  void ClearInferenceState() {
    op_def_ = nullptr;
    inference_arg_idx_ = 0;
    inference_attrs_.clear_no_resize();
  }

  Status MaybeInferSingleInputAttrs(ImmediateExecutionTensorHandle* handle);
  Status InferInputListAttrs(int num_inputs);

  void InferSingleTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                     const DataType dtype, int num_inputs);
  void InferMixedTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                    const std::vector<DataType>& dtypes);

  tensorflow::EagerContext& ctx_;
  const char* op_name_ = nullptr;
  AttrBuilder attrs_;
  const AttrTypeMap* attr_types_;

  // The number of custom device TensorHandle inputs. These inputs need to be
  // processed by CustomDeviceOpHandler first.
  int custom_device_tensor_handles_count_ = 0;
  absl::InlinedVector<ImmediateExecutionTensorHandle*, 4> inputs_;

  // The last device name given to SetDeviceName.
  // This is used to avoid having to re-process the same device in repeated
  // calls to SetDeviceName.
  string last_set_device_name_;

  // The operation's device name.
  // This contains the named passed to SetDeviceName until device_ is set,
  // at which point it contains the device_ name.
  string device_name_;

  // The parsed device name.
  // This will always contain the result of
  // DeviceNameUtils::ParseFullName(device_name_).
  DeviceNameUtils::ParsedName device_parsed_name_;

  // The operation's device.
  // This is set by the execution device placement logic, and should conform
  // with the contents of device_name_. Once it is set, the device_name_ is
  // updated accordingly.
  VariantDevice device_;

  absl::optional<ManagedStackTrace> stack_trace_;
  bool is_function_;  // Conceptually const, but can't be because of Reset
  bool colocation_exempt_;
  CancellationManager* cancellation_manager_ = nullptr;  // Not owned.
  EagerExecutor* executor_;                              // Not owned.

  absl::optional<EagerFunctionParams> eager_func_params_;

  // Inference information
  const tensorflow::OpDef* op_def_;  // op definition from protobuf
  int inference_arg_idx_;  // arg definition index for the next input to be
                           // added
  gtl::FlatSet<std::string> inference_attrs_;  // attributes inferred so far
};

inline void EagerOperation::UpdateInput(int i, TensorHandle* h) {
  ImmediateExecutionTensorHandle** slot = &inputs_[i];
  ImmediateExecutionTensorHandle* existing = *slot;
  if (existing != h) {
    h->Ref();
    existing->Unref();
    *slot = h;  // Update inputs_[i] to h
  }
}

inline EagerOperation* OperationFromInterface(
    ImmediateExecutionOperation* operation) {
  return down_cast<EagerOperation*>(operation);
}

inline const EagerOperation* OperationFromInterface(
    const ImmediateExecutionOperation* operation) {
  return down_cast<const EagerOperation*>(operation);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
