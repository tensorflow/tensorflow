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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/custom_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
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

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {

// Every pop* has a OpInfoMap function which maps a function name to
// CustomPoplibOpInfo which contains a creator and allocator function.
typedef const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>& (
    *GetOpInfoMapFn)();

StatusOr<const CustomPoplibOpInfo> GetCustomPoplibOpInfo(
    const HloInstruction* inst) {
  // Mapping of getting the right Create function given the metadata.
  PoplibsLib poplibs_lib;
  PoplibsOp poplibs_op;
  auto ret = GetPoplibsCustomOp(inst);
  if (ret == absl::nullopt) {
    return xla::FailedPrecondition("Unknown poplibs library on %s.",
                                   inst->name());
  }
  std::tie(poplibs_lib, poplibs_op) = ret.value();

  static absl::flat_hash_map<PoplibsLib, GetOpInfoMapFn> poplibs_info_map = {
      {PoplibsLib::Poplin, GetPoplinOpInfoMap},
      {PoplibsLib::Popnn, GetPopnnOpInfoMap},
      {PoplibsLib::Popops, GetPopopsOpInfoMap},
      {PoplibsLib::Poprand, GetPoprandOpInfoMap},
  };

  // First find the right poplibs library map
  auto it_type = poplibs_info_map.find(poplibs_lib);
  if (it_type == poplibs_info_map.end()) {
    return xla::FailedPrecondition("Unknown poplibs library: %s.",
                                   PoplibsLibToString(poplibs_lib).c_str());
  }
  // Then find the right CustomPoplibOpInfo
  auto lib_info_map = it_type->second();
  auto it_name = lib_info_map.find(poplibs_op);
  if (it_name == lib_info_map.end()) {
    return xla::FailedPrecondition("Unknown custom poplibs function: %s.",
                                   PoplibsOpToString(poplibs_op).c_str());
  }
  return it_name->second;
}
}  // namespace

StatusOr<poplar::Tensor> AllocatePoplibsOpTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const TensorTarget& tensor_target, const xla::Shape& shape,
    const TensorMap& tensor_map) {
  const HloInstruction* inst = tensor_target.tgt;
  TF_ASSIGN_OR_RETURN(auto op_info, GetCustomPoplibOpInfo(inst));
  auto allocator_function = op_info.first;
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      allocator_function(graph, res, name, tensor_target,
                                         attribute_map, tensor_map));
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