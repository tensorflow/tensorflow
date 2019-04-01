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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PARAMETER_ACCESSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PARAMETER_ACCESSOR_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/uniform_parameter.h"

namespace tflite {
namespace gpu {
namespace gl {

// This rewrite handles access to parameters. It may rewrite a parameter with
// actual values if inline_values is set to true.
//
// The following syntax is supported to access parameters:
//  - simple parameter: name
//  - parameter with field: name.(x|y|z|w)
//  - parameter with index: name[i]
//  - parameter with index and field: name[i].(x|y|z|w)
//
// If 'inline_values' is set to true, non variable-length parameters will be
// inlined. For example, 'base.x' will be replaced with value of 'x' field from
// 'base'. Variable-length are declared as const and accessed via index.
// These declarations are returned by GetConstDeclarations.
//
// If 'inline_values' is set to false, all parameters will be declared as
// uniforms. Uniform declarations are returned by GetUniformDeclarations.
class ParameterAccessor : public InlineRewrite {
 public:
  explicit ParameterAccessor(bool inline_values)
      : inline_values_(inline_values) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final;

  // Return true if parameter was successfully added.
  bool AddParameter(UniformParameter param);

  // Returns const parameters that need to be inlined in the a shader's code.
  std::string GetConstDeclarations() const;

  // Returns uniforms declarations that need to be inlined in a shader's code.
  std::string GetUniformDeclarations() const;

  // Returns a collection of uniform parameters.
  std::vector<UniformParameter> GetUniformParameters() const;

 private:
  const bool inline_values_;
  // Unique parameter index used for obfuscation.
  uint32_t unique_param_index_ = 0;

  std::unordered_map<std::string, UniformParameter> name_to_param_;
};

// Implementation details below.

namespace parameter_accessor_internal {

struct ParameterReference {
  absl::string_view name;
  absl::string_view index;
  absl::string_view field;
};

// Parse the following regex manually
// name(\[index\])?(\.field)?
ParameterReference Parse(absl::string_view input);

}  // namespace parameter_accessor_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PARAMETER_ACCESSOR_H_
