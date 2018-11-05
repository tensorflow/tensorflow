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

namespace xla {
namespace poplarplugin {
namespace {

// Every pop* has a CallMap function which maps a function name to a op creater
// function.
typedef const absl::flat_hash_map<std::string, CustomPoplibsCallFn>& (
    *GetCallMapFn)();

absl::flat_hash_map<std::string, GetCallMapFn> poplibs_call_map = {
    {"poplin", GetPoplinCallMap},
    {"popnn", GetPopnnCallMap},
    {"poprand", GetPoprandCallMap},
};
}

StatusOr<poplar::program::Program> CreatePoplibsOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  auto metadata = inst->metadata();
  const std::string op_type = metadata.op_type();
  const std::string op_name = metadata.op_name();

  // First find the right poplibs library map
  auto it_type = poplibs_call_map.find(op_type);
  if (it_type == poplibs_call_map.end()) {
    return xla::FailedPrecondition("Unknown poplibs library: %s.", op_type);
  }
  // Then find the right Create function
  auto lib_call_map = it_type->second();
  auto it_name = lib_call_map.find(op_name);
  if (it_name == lib_call_map.end()) {
    return xla::FailedPrecondition("Unknown custom poplibs function: %s.",
                                   op_name);
  }
  auto function_to_call = it_name->second;
  poplar::program::Program prog;
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
  TF_ASSIGN_OR_RETURN(prog, function_to_call(graph, res, inst, output_shape,
                                             tensor_map, attribute_map));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla