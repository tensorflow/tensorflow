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
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

namespace tflite {
namespace gpu {
namespace gl {

ShaderCodegen::ShaderCodegen(const CompilationOptions& options,
                             const GpuInfo& gpu_info)
    : options_(options),
      gpu_type_(gpu_info.vendor),
      inline_parameters_(options.inline_parameters) {
#ifdef __ANDROID__
  if (gpu_info.IsAdreno() &&
      gpu_info.adreno_info.adreno_gpu == AdrenoGpu::kAdreno730) {
    char sdk_version[PROP_VALUE_MAX];
    __system_property_get("ro.build.version.sdk", sdk_version);
    if (!strcmp(sdk_version, "31")) inline_parameters_ = false;
  } else if (gpu_info.IsPowerVR() &&
             !gpu_info.powervr_info.IsBetterThan(PowerVRGpu::kRogueGm9xxx)) {
    inline_parameters_ = false;
  }
#endif  // __ANDROID__
}

absl::Status ShaderCodegen::Build(CompiledNodeAttributes attr,
                                  ShaderCode* shader_code) const {
  VariableAccessor variable_accessor(inline_parameters_,
                                     options_.vulkan_support);
  ObjectAccessor object_accessor(gpu_type_ == GpuVendor::kMali,
                                 options_.sampler_textures, &variable_accessor);

  const auto add_object = [&](const std::string& name, Object&& object) {
    if (!object_accessor.AddObject(name, std::forward<Object>(object))) {
      return absl::AlreadyExistsError(absl::StrCat("Object \"", name, "\""));
    }
    return absl::OkStatus();
  };

  const auto add_uniform_parameter = [&](Variable&& variable) {
    const std::string name = variable.name;
    const Variable& const_ref = variable;
    if (variable_accessor.IsEmptyVariableLength(const_ref)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Empty uniform vector value \"", name, "\""));
    }
    if (!variable_accessor.AddUniformParameter(std::move(variable))) {
      return absl::AlreadyExistsError(
          absl::StrCat("Uniform parameter \"", name, "\""));
    }
    return absl::OkStatus();
  };

  for (auto&& object : attr.code.objects) {
    RETURN_IF_ERROR(add_object(object.first, std::move(object.second)));
  }

  for (auto&& variable : attr.code.shared_variables) {
    const std::string name = variable.name;
    if (!variable_accessor.AddSharedVariable(std::move(variable))) {
      return absl::AlreadyExistsError(
          absl::StrCat("Shared variable \"", name, "\""));
    }
  }

  for (auto&& variable : attr.code.parameters) {
    RETURN_IF_ERROR(add_uniform_parameter(std::move(variable)));
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
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_x", static_cast<int32_t>(attr.code.workload.x)}));
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_y", static_cast<int32_t>(attr.code.workload.y)}));
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_z", static_cast<int32_t>(attr.code.workload.z)}));

  // NOTE: If the shader has shared variables it will have to use barriers,
  //       which will conflict with a return at this stage.
  // Let the user deal with the geometry constraints.
  const bool has_shared_variables = !attr.code.shared_variables.empty();
  std::string main_source_code = has_shared_variables ? R"(
  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
)"
                                                      : R"(
  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
  if (gid.x >= $workload_x$ || gid.y >= $workload_y$ || gid.z >= $workload_z$) {
    return;
  }
)";

  switch (attr.code.input) {
    case IOStructure::ONLY_DEFINITIONS:
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&main_source_code, "  highp vec4 value_", i,
                        " = vec4(0);\n");
      }
      break;
    case IOStructure::AUTO: {
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&main_source_code, "  highp vec4 value_", i,
                        " = $input_data_", i, "[gid.x, gid.y, gid.z]$;\n");
      }
      break;
    }
  }

  main_source_code.append(attr.code.source_code);

  if (attr.code.output == IOStructure::AUTO) {
    for (int i = 0; i < attr.outputs.size(); ++i) {
      absl::StrAppend(&main_source_code, "  $output_data_", i,
                      "[gid.x, gid.y, gid.z] = value_", i, "$;\n");
    }
  }

  // At this point main function is already generated. Now we need to process
  // object and variable accessors.

  // process objects first. Object accessor may introduce new uniform
  // parameters that need to be rewritten in the subsequent pass.
  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/true);
    preprocessor.AddRewrite(&object_accessor);
    RETURN_IF_ERROR(preprocessor.Rewrite(main_source_code, &main_source_code));
  }

  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/false);
    preprocessor.AddRewrite(&variable_accessor);
    RETURN_IF_ERROR(preprocessor.Rewrite(main_source_code, &main_source_code));
  }

  if (inline_parameters_) {
    main_source_code = absl::StrCat(variable_accessor.GetConstDeclarations(),
                                    main_source_code);
  }

  // partial_source_code is only missing the following which is added later:
  // #version 310 es
  // layout(local_size_x = ..., local_size_y = ..., local_size_z = ...) in;
  const char* precision = options_.allow_precision_loss ? "mediump" : "highp";
  const std::string partial_source_code = absl::StrCat(
      "layout(std430) buffer;\n",                                 //
      "precision ", precision, " float;\n",                       //
      object_accessor.GetFunctionsDeclarations(), "\n",           //
      object_accessor.GetObjectDeclarations(), "\n",              //
      variable_accessor.GetUniformParameterDeclarations(), "\n",  //
      variable_accessor.GetSharedVariableDeclarations(), "\n",    //
      "void main() {\n",                                          //
      main_source_code,                                           //
      "}");
  *shader_code =
      ShaderCode(variable_accessor.GetUniformParameters(),
                 object_accessor.GetObjects(), attr.code.workload,
                 attr.code.workgroup, partial_source_code, attr.node_indices);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
