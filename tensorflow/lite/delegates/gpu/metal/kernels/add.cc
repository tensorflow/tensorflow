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
namespace {

std::string GetAddTableCodeFused(int src_count) {
  std::string code = "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid";
  for (int i = 0; i < src_count; ++i) {
    code += ", device FLT4* const src_buf" + std::to_string(i);
  }
  code += ") {\n";
  for (int i = 0; i < src_count; ++i) {
    code += "  value += src_buf" + std::to_string(i) + "[linear_index];\n";
    code += "  return value;\n";
  }
  code += "}\n";
  return code;
}
}  // namespace

ComputeTaskDescriptor Add(const OperationDef& definition) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  desc.is_associative_op = true;
  desc.shader_source = GetAddTableCodeFused(definition.src_tensors.size() - 1);

  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    desc.AddSrcTensor("", definition.src_tensors[i]);
  }
  desc.AddDstTensor("", definition.dst_tensors[0]);

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
