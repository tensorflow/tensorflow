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
#include "tensorflow/compiler/xla/service/optimize_input_output_buffer_alias.h"

#include <queue>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Returns true if the given shape is a non-nested tuple.
bool IsNonNestedTuple(const Shape& shape) {
  return shape.IsTuple() && !ShapeUtil::IsNestedTuple(shape);
}

}  // namespace

StatusOr<bool> OptimizeInputOutputBufferAlias::Build(
    const Shape& input_shape, const Shape& output_shape,
    HloInputOutputAliasConfig* alias_config) {
  bool changed = false;
  TF_RET_CHECK(LayoutUtil::HasLayout(input_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(output_shape));
  VLOG(1) << "input_shape:" << input_shape.ToString();
  VLOG(1) << "output_shape:" << output_shape.ToString();

  // For all buffers defined by the parameter, build a map from the byte
  // size to the list of the buffers of that size.
  absl::flat_hash_map<int64, std::queue<ShapeIndex>> size_to_input_index;
  ShapeUtil::ForEachSubshape(
      input_shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsTuple()) {
          return;
        }
        int64 bytes = size_func_(subshape);
        size_to_input_index[bytes].push(index);
      });

  // For each result buffer shape index, take the first unused parameter
  // buffer that matches the size.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      output_shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsTuple()) {
          return Status::OK();
        }
        int64 bytes = size_func_(subshape);

        auto it = size_to_input_index.find(bytes);
        if (it != size_to_input_index.end() && !it->second.empty()) {
          changed = true;
          const ShapeIndex& input_index = it->second.front();
          const ShapeIndex& output_index = index;
          if (!alias_config->ParameterHasAlias(0, input_index) &&
              !alias_config->OutputHasAlias(output_index)) {
            TF_RETURN_IF_ERROR(alias_config->SetUpAlias(
                output_index, 0, input_index,
                HloInputOutputAliasConfig::AliasKind::kSystemAlias));
          }
          VLOG(3) << "Set up alias from with param index "
                  << it->second.front().ToString() << ", shape size " << bytes
                  << " and result subshape "
                  << ShapeUtil::HumanStringWithLayout(subshape) << " at index "
                  << index.ToString();
          it->second.pop();
        }
        return Status::OK();
      }));
  return changed;
}

StatusOr<bool> OptimizeInputOutputBufferAlias::Run(HloModule* module) {
  // User buffer alias only work for modules with 1 parameter.
  if (module->entry_computation()->num_parameters() != 1) {
    return false;
  }

  HloInputOutputAliasConfig* alias_config =
      &module->input_output_alias_config();

  return Build(module->entry_computation()->parameter_instruction(0)->shape(),
               module->entry_computation()->root_instruction()->shape(),
               alias_config);
}

}  // namespace xla
