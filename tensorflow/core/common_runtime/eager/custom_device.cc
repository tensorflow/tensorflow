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

#include "tensorflow/core/common_runtime/eager/custom_device.h"

#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"

namespace tensorflow {

Status CustomDeviceTensorHandle::Shape(PartialTensorShape* shape) const {
  int num_dims;
  TF_RETURN_IF_ERROR(NumDims(&num_dims));
  std::vector<int64_t> dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    TF_RETURN_IF_ERROR(Dim(i, &dims[i]));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), num_dims, shape);
}

Status CustomDeviceTensorHandle::NumElements(int64_t* num_elements) const {
  *num_elements = 1;
  int num_dims;
  TF_RETURN_IF_ERROR(NumDims(&num_dims));
  for (int i = 0; i < num_dims; ++i) {
    int64_t dim;
    TF_RETURN_IF_ERROR(Dim(i, &dim));
    if (dim < 0) {
      return errors::InvalidArgument(
          absl::StrCat("Tried to compute the number of elements of a tensor "
                       "representing varying shapes. ",
                       DebugString()));
    }
    *num_elements *= dim;
  }
  return absl::OkStatus();
}

const char* CustomDeviceTensorHandle::DeviceType(Status* status) const {
  const DeviceNameUtils::ParsedName* parsed = ParsedName(status);
  if (!status->ok()) {
    return "";
  }
  return parsed->type.c_str();
}

int CustomDeviceTensorHandle::DeviceId(Status* status) const {
  const DeviceNameUtils::ParsedName* parsed = ParsedName(status);
  if (!status->ok()) {
    return 0;
  }
  return parsed->id;
}

AbstractTensorInterface* CustomDeviceTensorHandle::Resolve(Status* status) {
  core::RefCountPtr<ImmediateExecutionTensorHandle> copied_off(
      context_->GetCustomDeviceOpHandler().CopyTensorHandleToDevice(
          context_, this,
          DeviceNameUtils::ParsedNameToString(context_->HostCPUParsedName())
              .c_str(),
          status));
  if (!status->ok()) {
    return nullptr;
  }
  return copied_off->Resolve(status);
}

const DeviceNameUtils::ParsedName* CustomDeviceTensorHandle::ParsedName(
    Status* status) const {
  if (!parsed_name_.has_value()) {
    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullOrLocalName(device_->name(), &parsed_name)) {
      *status = errors::InvalidArgument(
          absl::StrCat("Invalid custom device name ", device_->name()));
      return nullptr;
    }
    parsed_name_.emplace(std::move(parsed_name));
  }
  return &*parsed_name_;
}

}  // namespace tensorflow
