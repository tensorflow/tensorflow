/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_

#include <string>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class TensorHandle;
class EagerOperation;
class CustomDeviceTensorHandle;

// Custom devices intercept the execution of operations (the `Execute` method),
// typically implemented with one or more of the custom device's own executions.
class CustomDevice {
 public:
  virtual ~CustomDevice() {}
  virtual const string& name() = 0;
  virtual Status CopyTensorToDevice(
      ImmediateExecutionTensorHandle* tensor,
      ImmediateExecutionTensorHandle** result) = 0;

  virtual Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* tensor, const string& target_device_name,
      ImmediateExecutionTensorHandle** result) = 0;

  virtual Status Execute(const ImmediateExecutionOperation* op,
                         ImmediateExecutionTensorHandle** retvals,
                         int* num_retvals) = 0;

  // Creates a packed TensorHandle from a group of custom device TensorHandles,
  // one of which is on this custom device.
  virtual Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
                      ImmediateExecutionTensorHandle** result) = 0;

  // Returns true signifying to pin to the current custom device.
  // Returns false to pin to the physical device.
  virtual StatusOr<bool> ShallPinToThisDevice(
      const ImmediateExecutionOperation* op) = 0;
};

// Custom devices do many of the same things as physical Devices, but have a
// much more restricted interface. We pass around ambiguous pointers since
// operations may be placed either on custom or physical devices.
using VariantDevice = absl::variant<Device*, CustomDevice*>;

// Indicates either HostCPU or an unset physical device. We never set a null
// CustomDevice*.
const VariantDevice kVariantDeviceNull = static_cast<Device*>(nullptr);

// A tensor handle produced by a custom device. Generally they can only be
// consumed by executing an operation on the same custom device that produced it
// originally, or by attempting to copy the handle off the custom device.
//
// TODO(allenl): Currently custom devices are tied to the eager C API. They
// should be renamed op handlers and subclass AbstractTensorHandle instead so
// they are eager/graph agnostic.
//
// full_type_ is not set by the constructor (because it is not currently
// needed). If full type information is needed in the future, the constructor
// could use map_dtype_to_child_of_tensor() from core/framework/types.h to set
// it based on dtype. Update test CustomDevice.TestTensorHandle in
// custom_device_test.cc if this changes.
class CustomDeviceTensorHandle : public ImmediateExecutionTensorHandle {
 public:
  CustomDeviceTensorHandle(ImmediateExecutionContext* context,
                           CustomDevice* device, tensorflow::DataType dtype)
      : ImmediateExecutionTensorHandle(kCustomDevice),
        context_(context),
        device_(device),
        dtype_(dtype) {}

  // TODO(allenl): Should this be a generic method of
  // ImmediateExecutionTensorHandle to support TFE_TensorHandleDevicePointer?
  virtual void* DevicePointer() const = 0;

  tensorflow::DataType DataType() const override { return dtype_; }
  tensorflow::FullTypeDef FullType() const override { return full_type_; }
  Status Shape(PartialTensorShape* shape) const override;
  Status NumElements(int64_t* num_elements) const override;

  const char* DeviceName(Status* status) const override {
    return device_->name().c_str();
  }
  const char* BackingDeviceName(Status* status) const override {
    return device_->name().c_str();
  }
  CustomDevice* device() const { return device_; }
  const char* DeviceType(Status* status) const override;
  int DeviceId(Status* status) const override;

  AbstractTensorInterface* Resolve(Status* status) override;

  ImmediateExecutionTensorHandle* Copy() override {
    Ref();
    return this;
  }
  void Release() override { Unref(); }

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kCustomDevice;
  }

 protected:
  const DeviceNameUtils::ParsedName* ParsedName(Status* status) const;

  ImmediateExecutionContext* const context_;
  CustomDevice* const device_;
  const tensorflow::DataType dtype_;
  tensorflow::FullTypeDef full_type_;

  mutable absl::optional<DeviceNameUtils::ParsedName> parsed_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_
