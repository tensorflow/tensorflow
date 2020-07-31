/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/add.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

Add::Add(const OperationDef& definition, const std::vector<int>& channels,
         int dst_channels)
    : ElementwiseOperation(definition),
      dst_depth_(DivideRoundUp(dst_channels, 4)) {
  src_depthes_.resize(channels.size());
  for (int i = 0; i < channels.size(); ++i) {
    src_depthes_[i] = DivideRoundUp(channels[i], 4);
  }
  if (src_depthes_[0] < dst_depth_) {
    check_src_channels_size_ = true;
  }
  for (int i = 1; i < definition_.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_data_", i);
    auto src_desc =
        absl::make_unique<TensorDescriptor>(definition_.src_tensors[i]);
    if (definition_.IsBatchSupported()) {
      src_desc->SetStateVar("BatchedWidth", "true");
    }
    args_.AddObjectRef(tensor_name, AccessType::READ, std::move(src_desc));
    code_ += "if (S_COORD < args." + tensor_name + ".Slices()) {\n";
    code_ += "  in_out_value += args." + tensor_name +
             ".Read(X_COORD, Y_COORD, S_COORD);\n";
    code_ += "}\n";
  }
}

Add::Add(Add&& operation)
    : ElementwiseOperation(std::move(operation)),
      link_index_(operation.link_index_),
      src_depthes_(std::move(operation.src_depthes_)),
      dst_depth_(operation.dst_depth_) {}

Add& Add::operator=(Add&& operation) {
  if (this != &operation) {
    link_index_ = operation.link_index_;
    src_depthes_ = std::move(operation.src_depthes_);
    dst_depth_ = operation.dst_depth_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Add::SetArgs(const std::string& unique_postfix, Arguments* args) {
  for (int i = 1; i < definition_.src_tensors.size(); ++i) {
    std::string tensor_name = absl::StrCat("src_data_", i, unique_postfix);
    RETURN_IF_ERROR(args->SetObjectRef(tensor_name, src_[i]));
  }
  return absl::OkStatus();
}

Add CreateAdd(const OperationDef& definition, const std::vector<int>& channels,
              int dst_channels) {
  Add operation(definition, channels, dst_channels);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
