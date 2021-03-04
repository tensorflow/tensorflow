/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

void CustomDeviceOpHandler::Clear() { custom_devices_.clear(); }

Status CustomDeviceOpHandler::RegisterCustomDevice(
    const string& device_name, std::unique_ptr<CustomDevice> device) {
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed) ||
      !parsed.has_job || !parsed.has_replica || !parsed.has_task ||
      !parsed.has_type || !parsed.has_id) {
    return errors::InvalidArgument(
        device_name,
        " could not be parsed as a device name. Use the full "
        "/job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num> "
        "format.");
  }

  if (!custom_devices_.emplace(device_name, std::move(device)).second) {
    return errors::AlreadyExists(device_name,
                                 " already registered as a custom device.");
  }
  return Status::OK();
}

bool CustomDeviceOpHandler::FindCustomDeviceFromName(
    const string& name, CustomDevice** device) const {
  auto dev_it = custom_devices_.find(name);
  if (dev_it == custom_devices_.end()) {
    return false;
  }
  *device = dev_it->second.get();
  return true;
}

Status CustomDeviceOpHandler::Execute(ImmediateExecutionOperation* op,
                                      ImmediateExecutionTensorHandle** retvals,
                                      int* num_retvals) {
  tensorflow::CustomDevice* custom_device = nullptr;

  TF_RETURN_IF_ERROR(MaybePinToCustomDevice(&custom_device, *op));

  if (custom_device != nullptr) {
    return custom_device->Execute(op, retvals, num_retvals);
  }

  // The op will be placed on physical device. However, it contains custom
  // device tensor handles. The tensor handles will be copy to physical device
  // first.
  if (op->HasCustomDeviceInput()) {
    auto inputs = op->GetInputs();
    for (int i = 0; i < inputs.size(); ++i) {
      auto target_device = op->DeviceName();
      if (target_device.empty()) {
        target_device = op->GetContext()->HostCPUName();
      }
      // TODO(b/175427838): It would be nice to be able to use tensorflow::isa
      // here.
      if (tensorflow::CustomDeviceTensorHandle::classof(inputs[i])) {
        tensorflow::CustomDeviceTensorHandle* previous =
            tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
                inputs[i]);
        tensorflow::ImmediateExecutionTensorHandle* new_tesnor;
        TF_RETURN_IF_ERROR(previous->device()->CopyTensorFromDevice(
            previous, target_device, &new_tesnor));
        Status s = op->SetInput(i, new_tesnor);
        new_tesnor->Unref();
        TF_RETURN_IF_ERROR(s);
      }
    }
  }

  return op->Execute(
      absl::MakeSpan(
          reinterpret_cast<tensorflow::AbstractTensorHandle**>(retvals),
          *num_retvals),
      num_retvals);
}

ImmediateExecutionTensorHandle* CustomDeviceOpHandler::CopyTensorHandleToDevice(
    ImmediateExecutionContext* context, ImmediateExecutionTensorHandle* handle,
    const char* device_name, Status* status) {
  *status = Status::OK();
  ImmediateExecutionTensorHandle* result = nullptr;
  tensorflow::CustomDevice* dev;

  if (FindCustomDeviceFromName(device_name, &dev)) {
    *status = dev->CopyTensorToDevice(handle, &result);
    if (status->ok()) {
      return result;
    }
    return nullptr;
  }

  // Target device is regular device. Check if the input is on custom
  // device
  const char* handle_device_name = handle->DeviceName(status);
  if (!status->ok()) {
    return nullptr;
  }
  if (FindCustomDeviceFromName(handle_device_name, &dev)) {
    *status = dev->CopyTensorFromDevice(handle, device_name, &result);
    if (status->ok()) {
      return result;
    }
    return nullptr;
  }

  // Both source and target device are regular device.
  return context->CopyTensorHandleToDevice(handle, device_name, status);
}

Status CustomDeviceOpHandler::MaybePinToCustomDevice(
    CustomDevice** device, const ImmediateExecutionOperation& op) const {
  *device = nullptr;
  if (!FindCustomDeviceFromName(op.DeviceName(), device) &&
      !op.HasCustomDeviceInput()) {
    return Status::OK();
  }

  // Ops are placed on a custom device if there's no other explicit requested
  // placement and there is only one custom device in the op
  // inputs.
  //
  // Resource-dtype inputs take precedence over non-resource inputs and explicit
  // placements; this function pins ops with a resource-dtype custom device
  // input to that custom device.
  CustomDevice* first = nullptr;
  if (!op.GetInputs().empty()) {
    for (const ImmediateExecutionTensorHandle* generic_input : op.GetInputs()) {
      // TODO(b/175427838): It would be nice to be able to use tensorflow::isa
      // here.
      if (CustomDeviceTensorHandle::classof(generic_input)) {
        const CustomDeviceTensorHandle* input =
            down_cast<const CustomDeviceTensorHandle*>(generic_input);
        CustomDevice* current = input->device();
        if (first == nullptr) {
          first = current;
        } else if (first != current) {
          return errors::InvalidArgument(absl::StrCat(
              "If an operation has one of its inputs in a custom device, then "
              "all inputs should be on that same custom device or another "
              "physical device. Operation ",
              op.Name(),
              " has one input in custom "
              "device ",
              first->name(),
              " and at least one input in a different custom device ",
              current->name()));
        }
      }
    }
    for (const ImmediateExecutionTensorHandle* generic_input : op.GetInputs()) {
      if (generic_input->DataType() == DT_RESOURCE) {
        if (CustomDeviceTensorHandle::classof(generic_input)) {
          const CustomDeviceTensorHandle* input =
              down_cast<const CustomDeviceTensorHandle*>(generic_input);
          // There's only one custom device input, and it's a resource input, so
          // we'll force-place the op on to that custom device. As with physical
          // devices, this overrides any explicit placement for the op.
          *device = input->device();
          return Status::OK();
        } else {
          // Don't set a custom device if there's a physical-device resource
          // input.
          return Status::OK();
        }
      }
    }
  }
  // Since there are no resource-dtype inputs, we'll respect explicit placements
  // before considering input-based placement.
  if (*device == nullptr && op.DeviceName().empty() && first != nullptr) {
    // If there are non-resource inputs on a custom device we will default the
    // op to that custom device, but not override an explicit op placement.
    *device = first;
    return Status::OK();
  }
  return Status::OK();
}

}  // namespace tensorflow
