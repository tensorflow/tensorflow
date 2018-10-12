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
tensorflow::Status EagerOperation::SetDevice(const char* device) {
  auto status = Status::OK();
  tensorflow::Device* d = nullptr;
  if (device != nullptr && strlen(device) > 0) {
    status.Update(ctx_->FindDeviceByName(device, &d));
  }
  device_ = d;
  return status;
}

void EagerOperation::AddInput(tensorflow::TensorHandle* h) {
  h->Ref();
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}
}  // namespace tensorflow
