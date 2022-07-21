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

#include "tensorflow/lite/delegates/gpu/common/tasks/add.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

GPUOperation CreateAdd(const OperationDef& definition,
                       const std::vector<int>& channels, int dst_channels) {
  GPUOperation add(definition);
  int dst_depth = DivideRoundUp(dst_channels, 4);
  int src0_depth = DivideRoundUp(channels[0], 4);
  add.elementwise_ = true;
  add.linkable_ = dst_depth == src0_depth;
  if (src0_depth < dst_depth) {
    add.check_src_channels_size_ = true;
  }
  add.code_ += "  out_value = in_value;\n";
  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_data_", i);
    auto src_desc = definition.src_tensors[i];
    if (definition.IsBatchSupported()) {
      src_desc.SetStateVar("BatchedWidth", "true");
    }
    add.AddSrcTensor(tensor_name, src_desc);
    add.code_ += "if (S_COORD < args." + tensor_name + ".Slices()) {\n";
    add.code_ += "  out_value += args." + tensor_name +
                 ".Read(X_COORD, Y_COORD, S_COORD);\n";
    add.code_ += "}\n";
  }
  return add;
}

}  // namespace gpu
}  // namespace tflite
