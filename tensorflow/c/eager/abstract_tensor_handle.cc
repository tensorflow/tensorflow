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

#include "tensorflow/c/eager/abstract_tensor_handle.h"

namespace tensorflow {

std::string AbstractTensorHandle::DebugString() const {
  PartialTensorShape shape;
  Status s = Shape(&shape);
  std::string shape_string;
  if (!s.ok()) {
    shape_string = "<error computing shape>";
  } else {
    shape_string = shape.DebugString();
  }
  return absl::StrCat("TensorHandle(shape=", shape_string,
                      ", dtype=", DataType_Name(DataType()),
                      ", type=", FullType().DebugString(), ")");
}

Status AbstractTensorHandle::TensorHandleStatus() const {
  // Tensor handles in current runtime don't carry error info and this method
  // should always return OK status.
  return OkStatus();
}

}  // namespace tensorflow
