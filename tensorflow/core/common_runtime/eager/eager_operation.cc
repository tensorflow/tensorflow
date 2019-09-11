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
#include "tensorflow/core/common_runtime/eager/eager_operation.h"

namespace tensorflow {

tensorflow::Status EagerOperation::SetDeviceName(const char* device) {
  if (device != nullptr && strlen(device) > 0) {
    if (!DeviceNameUtils::ParseFullName(device, &device_name_)) {
      return errors::InvalidArgument("Malformed device specification '", device,
                                     "' in eager op: ", DebugString());
    }
  }
  return Status::OK();
}

string EagerOperation::DebugString() const {
  string out;
  VLOG(1) << "EagerOperation::DebugString() over " << this;

  strings::StrAppend(&out, "Name: ", name_, "\n");
  strings::StrAppend(&out, "Device Name: [",
                     DeviceNameUtils::ParsedNameToString(device_name_), "]\n");
  strings::StrAppend(
      &out, "Device: ", Device() ? Device()->DebugString() : "[]", "\n");
  for (const auto& input : inputs_) {
    VLOG(1) << "Input ptr: " << input;
    strings::StrAppend(&out, "Input: ", input->DebugString(), "\n");
  }

  NodeDef ndef;
  Attrs().FillAttrValueMap(ndef.mutable_attr());
  strings::StrAppend(&out, "Attrs: ", ndef.DebugString(), "\n");
  return out;
}

}  // namespace tensorflow
