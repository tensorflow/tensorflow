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

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

using tensorflow::strings::StrAppend;

HloModuleConfig::HloModuleConfig() {}

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape)
    : host_entry_computation_layout_(program_shape),
      device_entry_computation_layout_(program_shape) {}

void HloModuleConfig::SetDefaultComputationLayout(
    const ProgramShape& program_shape) {
  host_entry_computation_layout_ = ComputationLayout(program_shape);
  device_entry_computation_layout_ = ComputationLayout(program_shape);
}

string HloModuleConfig::compilation_cache_key() const {
  string key =
      tensorflow::strings::StrCat("profiling=", hlo_profiling_enabled());
  StrAppend(&key, "::(");
  std::vector<string> params;
  for (const ShapeLayout& param_layout :
       host_entry_computation_layout_->parameter_layouts()) {
    params.push_back(param_layout.shape().DebugString());
  }
  StrAppend(&key, tensorflow::str_util::Join(params, ", "), ") => ",
            host_entry_computation_layout_->result_shape().SerializeAsString());
  for (const ShapeLayout& param_layout :
       device_entry_computation_layout_->parameter_layouts()) {
    params.push_back(param_layout.shape().DebugString());
  }
  StrAppend(
      &key, tensorflow::str_util::Join(params, ", "), ") => ",
      device_entry_computation_layout_->result_shape().SerializeAsString());
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
  return key;
}

}  // namespace xla
