/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"

#include <algorithm>
#include <vector>

#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {

StatusOr<CompileOptionsProto> CompileOptions::ToProto() const {
  CompileOptionsProto output;
  if (argument_layouts.has_value()) {
    for (const auto& layout : *argument_layouts) {
      *output.add_argument_layouts() = layout.ToProto();
    }
  }
  output.set_parameter_is_tupled_arguments(parameter_is_tupled_arguments);
  TF_ASSIGN_OR_RETURN(*output.mutable_executable_build_options(),
                      executable_build_options.ToProto());
  output.set_compile_portable_executable(compile_portable_executable);
  output.set_profile_version(profile_version);
  // TODO(b/240299401): Serialize "multi_slice_config".
  return output;
}

StatusOr<CompileOptions> CompileOptionsFromProto(
    const CompileOptionsProto& input) {
  CompileOptions output;
  if (!input.argument_layouts().empty()) {
    output.argument_layouts = std::vector<Shape>();
    output.argument_layouts->reserve(input.argument_layouts_size());
    for (const auto& layout : input.argument_layouts()) {
      output.argument_layouts->emplace_back(Shape(layout));
    }
  }
  output.parameter_is_tupled_arguments = input.parameter_is_tupled_arguments();
  TF_ASSIGN_OR_RETURN(
      output.executable_build_options,
      ExecutableBuildOptionsFromProto(input.executable_build_options()));
  output.compile_portable_executable = input.compile_portable_executable();
  output.profile_version = input.profile_version();
  // TODO(b/240299401): Set "multi_slice_config".
  return output;
}

}  // namespace xla
