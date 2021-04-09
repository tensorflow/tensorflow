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

#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include <atomic>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

using absl::StrAppend;

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape,
                                 bool ignore_layouts)
    : entry_computation_layout_(
          ComputationLayout(program_shape, ignore_layouts)) {}

HloModuleConfig::HloModuleConfig(ComputationLayout entry_computation_layout)
    : entry_computation_layout_(std::move(entry_computation_layout)) {}

void HloModuleConfig::SetDefaultComputationLayout(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape);
}

void HloModuleConfig::SetComputationLayoutIfExists(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape,
                                                /*ignore_layouts=*/false);
}

string HloModuleConfig::compilation_cache_key() const {
  string key = absl::StrCat("profiling=", hlo_profiling_enabled());
  StrAppend(&key, "::(");
  std::vector<string> params;
  if (entry_computation_layout_.has_value()) {
    for (const ShapeLayout& param_layout :
         entry_computation_layout_->parameter_layouts()) {
      params.push_back(param_layout.shape().DebugString());
    }
    StrAppend(&key, absl::StrJoin(params, ", "), ") => ",
              entry_computation_layout_->result_shape().SerializeAsString());
  }
  if (seed() != 0) {
    // TODO(b/32083678): force recompilation to reset global state.
    static std::atomic<int> counter{0};
    StrAppend(&key, "forcing recompile ", counter++);
  }
  if (replica_count() != 1) {
    StrAppend(&key, "::replica_count=", replica_count());
  }
  StrAppend(&key, debug_options_.DebugString());
  if (intra_op_parallelism_threads() > 0) {
    StrAppend(&key, "::intra_op_parallelism_threads=",
              intra_op_parallelism_threads());
  }
  StrAppend(&key, "::alias_passthrough_params=", alias_passthrough_params_);
  return key;
}

}  // namespace xla
