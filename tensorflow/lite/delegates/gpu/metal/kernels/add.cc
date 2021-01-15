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

#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

ComputeTaskDescriptor Add(const OperationDef& definition) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;

  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = "src_tensor_" + std::to_string(i);
    desc.AddSrcTensor(tensor_name, definition.src_tensors[i]);
    desc.shader_source += "  in_out_value += args." + tensor_name +
                          ".Read(X_COORD, Y_COORD, S_COORD);\n";
  }

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
