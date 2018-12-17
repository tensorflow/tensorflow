/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/custom_ops.h"
#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/poplibs_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace xla {
namespace poplarplugin {
namespace {

// Every pop* has a OpInfoMap function which maps a function name to
// CustomPoplibOpInfo which contains a creator and allocator function.
typedef const absl::flat_hash_map<std::string, CustomPoplibOpInfo>& (
    *GetOpInfoMapFn)();

absl::flat_hash_map<std::string, GetOpInfoMapFn> poplibs_info_map = {
    {"poplin", GetPoplinOpInfoMap},
    {"popnn", GetPopnnOpInfoMap},
    {"poprand", GetPoprandOpInfoMap},
};

StatusOr<const CustomPoplibOpInfo> GetCustomPoplibOpInfo(
    const HloInstruction* inst) {
  // Mapping of getting the right Create function given the metadata:
  // * op_type - poplibs library name
  // * op_name - function inside given library
  std::vector<std::string> op_info =
      absl::StrSplit(inst->custom_call_target(), "::");
  if (op_info.size() != 2) {
    return xla::FailedPrecondition("Invalid custom poplibs call info: ",
                                   inst->custom_call_target());
  }
  // First find the right poplibs library map
  auto it_type = poplibs_info_map.find(op_info[0]);
  if (it_type == poplibs_info_map.end()) {
    return xla::FailedPrecondition("Unknown poplibs library: %s.", op_info[0]);
  }
  // Then find the right Create function
  auto lib_info_map = it_type->second();
  auto it_name = lib_info_map.find(op_info[1]);
  if (it_name == lib_info_map.end()) {
    return xla::FailedPrecondition("Unknown custom poplibs function: %s.",
                                   op_info[1]);
  }
  return it_name->second;
}
}  // namespace

StatusOr<poplar::Tensor> AllocatePoplibsOpTensor(poplar::Graph& graph,
                                                 CompilerResources& res,
                                                 const std::string& name,
                                                 const HloInstruction* inst,
                                                 const int64 target_idx,
                                                 const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(auto op_info, GetCustomPoplibOpInfo(inst));
  auto allocator_function = op_info.first;
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor out,
      allocator_function(graph, res, name, inst, target_idx, attribute_map));
  return out;
}

StatusOr<poplar::program::Program> CreatePoplibsOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(auto op_info, GetCustomPoplibOpInfo(inst));
  auto creator_function = op_info.second;
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      creator_function(graph, res, inst, output_shape,
                                       tensor_map, attribute_map));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla