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

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"

namespace tensorflow {

std::string ImmediateExecutionTensorHandle::DebugString() const {
  PartialTensorShape shape;
  std::string shape_string;
  if (Shape(&shape).ok()) {
    shape_string = shape.DebugString();
  } else {
    shape_string = "<error computing shape>";
  }
  std::string value_string;
  if (!SummarizeValue(value_string).ok()) {
    value_string = "<error computing value>";
  }
  if (value_string.length() > 100) {
    // The default NumPy-style output can be distractingly long in error
    // messages.
    value_string = absl::StrCat(value_string.substr(0, 100), " [...]");
  }
  absl::Status s;
  const char* device_name = DeviceName(&s);
  if (!s.ok()) {
    device_name = "<error fetching device name>";
  }
  return absl::StrCat("TensorHandle(", value_string, ", shape=", shape_string,
                      ", dtype=", DataType_Name(DataType()), ", device=\"",
                      device_name, "\")");
}

absl::Status ImmediateExecutionTensorHandle::SummarizeValue(
    std::string& summary) const {
  absl::Status status;
  AbstractTensorPtr resolved(
      // TODO(allenl): Resolve should be const, and the caches that get updated
      // marked mutable.
      const_cast<ImmediateExecutionTensorHandle*>(this)->Resolve(&status));
  if (!status.ok()) {
    return status;
  }
  summary = resolved->SummarizeValue();
  return absl::OkStatus();
}

}  // namespace tensorflow
