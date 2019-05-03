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

#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

namespace tflite {
namespace gpu {
namespace gl {

ShaderCodegen::ShaderCodegen(const CompilationOptions& options,
                             const GpuInfo& gpu_info)
    : options_(options), gpu_type_(gpu_info.type) {}

Status ShaderCodegen::Build(CompiledNodeAttributes attr,
                            ShaderCode* shader_code) const {
  ParameterAccessor parameters(options_.inline_parameters);
  ObjectAccessor objects(gpu_type_ == GpuType::MALI, &parameters);

  auto add_object = [&](const std::string& name, Object&& object) {
    if (!objects.AddObject(name, std::forward<Object>(object))) {
      return InternalError("There is an object with the same name");
    }
    return OkStatus();
  };

  auto add_parameter = [&](UniformParameter&& param) {
    if (!parameters.AddParameter(std::forward<UniformParameter>(param))) {
      return InternalError("There is a parameter with the same name");
    }
    return OkStatus();
  };

  for (auto&& param : attr.code.parameters) {
    RETURN_IF_ERROR(add_parameter(std::move(param)));
  }

  for (auto&& object : attr.code.objects) {
    RETURN_IF_ERROR(add_object(object.first, std::move(object.second)));
  }

  int index = 0;
  for (auto&& input : attr.inputs) {
    RETURN_IF_ERROR(
        add_object(absl::StrCat("input_data_", index++), std::move(input)));
  }
  index = 0;
  for (auto&& output : attr.outputs) {
    RETURN_IF_ERROR(
        add_object(absl::StrCat("output_data_", index++), std::move(output)));
  }

  // TODO(akulik): workload params need to go away and be replaced with
  // output_data_0_w
  RETURN_IF_ERROR(add_parameter(
      {"workload_x", static_cast<int32_t>(attr.code.workload.x)}));
  RETURN_IF_ERROR(add_parameter(
      {"workload_y", static_cast<int32_t>(attr.code.workload.y)}));
  RETURN_IF_ERROR(add_parameter(
      {"workload_z", static_cast<int32_t>(attr.code.workload.z)}));

  std::string source_code = R"(
  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
  if (gid.x >= $workload_x$ || gid.y >= $workload_y$ || gid.z >= $workload_z$) {
    return;
  }
)";

  switch (attr.code.input) {
    case IOStructure::ONLY_DEFINITIONS:
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&source_code, "  highp vec4 value_", i,
                        " = vec4(0);\n");
      }
      break;
    case IOStructure::AUTO: {
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&source_code, "  highp vec4 value_", i,
                        " = $input_data_", i, "[gid.x, gid.y, gid.z]$;\n");
      }
      break;
    }
  }

  source_code.append(attr.code.source_code);

  if (attr.code.output == IOStructure::AUTO) {
    for (int i = 0; i < attr.outputs.size(); ++i) {
      absl::StrAppend(&source_code, "  $output_data_", i,
                      "[gid.x, gid.y, gid.z] = value_", i, "$;\n");
    }
  }

  // At this point main function is already generated. Now we need to process
  // object and parameter accessors.

  // process objects first. Object accessor may introduce new uniform
  // parameters that need to be rewritten in the subsequent pass.
  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/true);
    preprocessor.AddRewrite(&objects);
    RETURN_IF_ERROR(preprocessor.Rewrite(source_code, &source_code));
  }

  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/false);
    preprocessor.AddRewrite(&parameters);
    RETURN_IF_ERROR(preprocessor.Rewrite(source_code, &source_code));
  }

  if (options_.inline_parameters) {
    source_code = absl::StrCat(parameters.GetConstDeclarations(), source_code);
  }

  std::string declarations = absl::StrCat(
      objects.GetFunctionsDeclarations(), "\n", objects.GetObjectDeclarations(),
      "\n", parameters.GetUniformDeclarations());
  *shader_code = ShaderCode(
      parameters.GetUniformParameters(), objects.GetObjects(),
      attr.code.workload, attr.code.workgroup,
      absl::StrCat("layout(std430) buffer;\nprecision ",
                   (options_.allow_precision_loss ? "mediump" : "highp"),
                   " float;\n", declarations, "\nvoid main() {\n", source_code,
                   "\n}"),
      attr.node_indices);
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
