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

#include <vector>

#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape)
    : entry_computation_layout_(program_shape) {}

string HloModuleConfig::compilation_cache_key() const {
  string key = tensorflow::strings::StrCat("profiling=", hlo_profiling_enabled_,
                                           "::", "hybrid=", has_hybrid_result_);
  tensorflow::strings::StrAppend(&key, "::(");
  std::vector<string> params;
  for (const ShapeLayout& param_layout :
       entry_computation_layout_.parameter_layouts()) {
    params.push_back(param_layout.shape().SerializeAsString());
  }
  tensorflow::strings::StrAppend(
      &key, tensorflow::str_util::Join(params, ", "), ") => ",
      entry_computation_layout_.result_shape().SerializeAsString());
  if (seed_ != 0) {
    // TODO(b/32083678): force recompilation to reset global state.
    static int counter = 0;
    tensorflow::strings::StrAppend(&key, "forcing recompile ", counter++);
  }
  if (replica_count() != 1) {
    tensorflow::strings::StrAppend(&key, "::replica_count=", replica_count());
  }
  return key;
}

}  // namespace xla
