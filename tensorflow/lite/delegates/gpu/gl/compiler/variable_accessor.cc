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

#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"

#include <string>
#include <utility>
#include <variant>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace variable_accessor_internal {

// Parse the following regex manually
// name(\[index\])?(\.field)?
VariableReference Parse(absl::string_view input) {
  VariableReference ref;
  auto start_index = input.find('[');
  if (start_index != std::string::npos) {
    auto end_index = input.rfind(']');
    if (end_index == std::string::npos) {
      return ref;
    }
    ref.index = input.substr(start_index + 1, end_index - start_index - 1);
    ref.name = input.substr(0, start_index);
    ref.field = input.substr(end_index + 1);
  } else {
    auto dot = input.find('.');
    if (dot != std::string::npos) {
      ref.name = input.substr(0, dot);
      ref.field = input.substr(dot);
    } else {
      ref.name = input;
    }
  }
  return ref;
}

}  // namespace variable_accessor_internal

namespace {

struct VariableTypeGetter {
  std::string operator()(int) const { return "int"; }
  std::string operator()(const int2&) const { return "ivec2"; }
  std::string operator()(const std::vector<int2>&) const { return "ivec2"; }
  std::string operator()(const int4&) const { return "ivec4"; }
  std::string operator()(unsigned int) const { return "uint"; }
  std::string operator()(const uint4&) const { return "uvec4"; }
  std::string operator()(float) const { return "float"; }
  std::string operator()(const float2&) const { return "vec2"; }
  std::string operator()(const float4&) const { return "vec4"; }
  std::string operator()(const std::vector<float4>&) const { return "vec4"; }
};

// Returns GLSL uniform type of the given variable.
std::string GetVariableType(const Variable::ValueType& value) {
  return std::visit(VariableTypeGetter(), value);
}

struct LengthGetter {
  template <typename T>
  int operator()(const T& param) const {
    return 1;
  }
  template <typename T>
  int operator()(const std::vector<T>& param) const {
    return param.size();
  }
};

int GetLength(const Variable::ValueType& value) {
  return std::visit(LengthGetter(), value);
}

template <typename T>
void FormatValue(std::string* result, T t) {
  absl::StrAppend(result, t);
}

template <>
void FormatValue(std::string* result, float t) {
  absl::StrAppend(result, absl::StrFormat("%.9ff", t));
}

// Unfortunately absl::StrJoin with custom formatter requires formatter to use
// string, not std::string. Therefore, due to this compatibility issue data
// needs to be converted to string representation first and then joined.
template <typename T, int N>
std::vector<std::string> ToString(const std::array<T, N>& data) {
  std::vector<std::string> result(N);
  for (int i = 0; i < N; ++i) {
    FormatValue(&result[i], data[i]);
  }
  return result;
}

struct ConstGenerator {
  template <typename T>
  void operator()(T t) const {
    FormatValue(result, t);
  }

  template <typename T>
  void operator()(const Vec2<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 2>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec3<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 3>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec4<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 4>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    std::string type = VariableTypeGetter()(v);
    absl::StrAppend(result, type, "[", v.size(), "](");
    bool first = true;
    for (const auto& i : v) {
      if (first) {
        first = false;
      } else {
        absl::StrAppend(result, ",");
      }
      (*this)(i);
    }
    absl::StrAppend(result, ")");
  }

  std::string* result;
};

// Appends string representation of a variable value.
void GetValue(const Variable::ValueType& value, std::string* result) {
  std::visit(ConstGenerator{result}, value);
}

struct SharedVariableDeclarationGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "shared highp ", GetVariableType(variable.value),
                    " ", variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "shared highp ", GetVariableType(variable.value),
                    " ", variable.name);
    if (v.empty()) {
      // Normalize the size of the shared array to that of the WorkGroupSize
      absl::StrAppend(
          result,
          "[gl_WorkGroupSize.z * gl_WorkGroupSize.y * gl_WorkGroupSize.x];\n");
    } else {
      // Use the specified size
      absl::StrAppend(result, "[", v.size(), "];\n");
    }
  }

  const Variable& variable;
  std::string* result;
};

void GenerateSharedVariableDeclaration(const Variable& variable,
                                       std::string* result) {
  std::visit(SharedVariableDeclarationGenerator{variable, result},
             variable.value);
}

struct UniformParameterDeclarationGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "uniform ", GetVariableType(variable.value), " ",
                    variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "uniform ", GetVariableType(variable.value), " ",
                    variable.name, "[", v.size(), "];\n");
  }

  const Variable& variable;
  std::string* result;
};

void GenerateUniformParameterDeclaration(const Variable& variable,
                                         std::string* result) {
  std::visit(UniformParameterDeclarationGenerator{variable, result},
             variable.value);
}

struct VulkanPushConstantGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "  ", GetVariableType(variable.value), " ",
                    variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "  ", GetVariableType(variable.value), " ",
                    variable.name, "[", v.size(), "];\n");
  }

  const Variable& variable;
  std::string* result;
};

void GenerateVulkanPushConstant(const Variable& variable, std::string* result) {
  std::visit(VulkanPushConstantGenerator{variable, result}, variable.value);
}

struct VariableLengthGetter {
  template <typename T>
  bool operator()(const T&) const {
    return false;
  }
  template <typename T>
  bool operator()(const std::vector<T>&) const {
    return true;
  }
};

struct VulkanConstantGenerator {
  template <typename T>
  void operator()(const T&) const {
    const std::string variable_type = GetVariableType(variable.value);

    // Vulkan specialization constants are used for scalar types, all other
    // types go in push (uniform) constants.
    if (variable_type == "int" || variable_type == "uint" ||
        variable_type == "float") {
      absl::StrAppend(result, "layout(constant_id = ", *constant_id, ") const ",
                      variable_type, " ", variable.name, " = ");
      // Always set the default values to zero to generate generic cacheable
      // shaders.
      absl::StrAppend(result, (variable_type == "float" ? "0.0" : "0"), ";\n");
      (*constant_id)++;
    } else {
      non_scalar_variables->push_back(variable);
    }
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    non_scalar_variables->push_back(variable);
  }

  const Variable& variable;
  int* const constant_id;
  std::vector<Variable>* non_scalar_variables;
  std::string* result;
};

void GenerateVulkanConstant(const Variable& variable, int* constant_id,
                            std::vector<Variable>* non_scalar_variables,
                            std::string* result) {
  std::visit(VulkanConstantGenerator{variable, constant_id,
                                     non_scalar_variables, result},
             variable.value);
}

class VulkanConstantsProcessor {
 public:
  void ProcessVulkanConstant(const Variable& variable, std::string* result) {
    GenerateVulkanConstant(variable, &constant_id_, &non_scalar_variables_,
                           result);
  }

  void GeneratePushConstantsDeclarations(std::string* result) {
    if (!non_scalar_variables_.empty()) {
      *result += "\nlayout(push_constant) uniform pushConstants {\n";
      for (const auto& variable : non_scalar_variables_) {
        GenerateVulkanPushConstant(variable, result);
      }
      *result += "};\n";
    }
  }

 protected:
  // Reserve the first three specialization constants slots for the
  // workgroup size.
  int constant_id_ = 3;
  std::vector<Variable> non_scalar_variables_;
};

// Returns true if value is a vector
bool IsVariableLength(const Variable::ValueType& value) {
  return std::visit(VariableLengthGetter(), value);
}

enum Field : uint8_t { UNKNOWN = 4, X = 0, Y = 1, Z = 2, W = 3 };

Field ToField(absl::string_view field_name) {
  if (field_name.size() == 2 && field_name[0] == '.') {
    switch (field_name[1]) {
      case 'x':
        return Field::X;
      case 'y':
        return Field::Y;
      case 'z':
        return Field::Z;
      case 'w':
        return Field::W;
    }
  }
  return Field::UNKNOWN;
}

struct FieldAccessor {
  template <typename T>
  void operator()(const T&) const {}

  template <typename T>
  void operator()(const Vec2<T>& v) const {
    FormatValue(result, v[field]);
  }

  template <typename T>
  void operator()(const Vec3<T>& v) const {
    FormatValue(result, v[field]);
  }

  template <typename T>
  void operator()(const Vec4<T>& v) const {
    FormatValue(result, v[field]);
  }

  Field field;
  std::string* result;
};

// Appends formatted value of the given field.
void GetValue(const Variable::ValueType& value, Field field,
              std::string* result) {
  std::visit(FieldAccessor{field, result}, value);
}

struct FieldChecker {
  // For trivial as well as variable-length types indexed access is not allowed.
  template <typename T>
  bool operator()(const T&) const {
    return false;
  }

  template <typename T>
  bool operator()(const Vec2<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const Vec3<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const Vec4<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const std::vector<T>&) const {
    // technically accessing [0] element of an empty vector is UB, but we need
    // only type information for this check. Therefore, construct default T and
    // use it instead.
    T t;
    return (*this)(t);
  }

  Field field;
};

// Returns true if field has field access and field is not out of bounds.
bool HasField(const Variable::ValueType& value, Field field) {
  return std::visit(FieldChecker{field}, value);
}

void AssembleAccessor(absl::string_view name, absl::string_view index,
                      absl::string_view field, std::string* result) {
  if (index.empty()) {
    absl::StrAppend(result, name, field);
  } else {
    absl::StrAppend(result, name, "[", index, "]", field);
  }
}

}  // namespace

RewriteStatus VariableAccessor::Rewrite(absl::string_view input,
                                        std::string* output) {
  auto ref = variable_accessor_internal::Parse(input);
  if (ref.name.empty()) {
    absl::StrAppend(output, "INVALID_SYNTAX");
    return RewriteStatus::ERROR;
  }

  auto it =
      name_to_variable_.find(std::string(ref.name.data(), ref.name.size()));
  if (it == name_to_variable_.end()) {
    // Uniform with this name is not registered.
    return RewriteStatus::NOT_RECOGNIZED;
  }
  const auto& value = it->second.value;

  if (!ref.index.empty() && !IsVariableLength(value)) {
    // Trying to access variable by index, but it is not variable-length.
    absl::StrAppend(output, "INVALID_ACCESS_BY_INDEX");
    return RewriteStatus::ERROR;
  }

  Field f = ToField(ref.field);
  if (!ref.field.empty() && !HasField(value, f)) {
    // Trying to access a variable by field, but it does not have it.
    absl::StrAppend(output, "INVALID_ACCESS_BY_FIELD");
    return RewriteStatus::ERROR;
  }

  // Error checks are complete now.

  // All variable-length variables are encoded as-is without inlining.
  if (!inline_values_ || IsVariableLength(value)) {
    AssembleAccessor(it->second.name, ref.index, ref.field, output);
  } else {
    // Parameter + field is replaced with field value.
    if (f != Field::UNKNOWN) {
      GetValue(value, f, output);
    } else {
      // Parameter is accessed directly.
      GetValue(value, output);
    }
  }
  return RewriteStatus::SUCCESS;
}

bool VariableAccessor::AddSharedVariable(Variable&& variable) {
  const std::string name = variable.name;
  if (!name_to_variable_.insert({name, std::move(variable)}).second) {
    return false;
  }
  shared_variables_.insert(name);
  return true;
}

bool VariableAccessor::AddUniformParameter(Variable&& variable) {
  const std::string name = variable.name;
  if (!name_to_variable_.insert({name, std::move(variable)}).second) {
    return false;
  }
  uniform_parameters_.insert(name);
  return true;
}

bool VariableAccessor::IsEmptyVariableLength(const Variable& variable) const {
  const auto& value = variable.value;
  return IsVariableLength(value) && GetLength(value) == 0;
}

std::string VariableAccessor::GetConstDeclarations() const {
  // Variable length variables are declared as const and accessed via variable
  // with index.
  std::string declarations;
  for (const auto& variable : name_to_variable_) {
    // Skip shared variables.
    const std::string& variable_name = variable.second.name;
    if (shared_variables_.find(variable_name) != shared_variables_.end()) {
      continue;
    }

    const auto& value = variable.second.value;
    if (IsVariableLength(value)) {
      absl::StrAppend(&declarations, "const ", GetVariableType(value), " ",
                      variable_name, "[] = ");
      GetValue(value, &declarations);
      absl::StrAppend(&declarations, ";\n");
    }
  }
  return declarations;
}

std::string VariableAccessor::GetSharedVariableDeclarations() const {
  std::string declarations;
  for (const auto& name : shared_variables_) {
    const auto& variable = name_to_variable_.at(name);
    GenerateSharedVariableDeclaration(variable, &declarations);
  }
  return declarations;
}

std::string VariableAccessor::GetUniformParameterDeclarations() const {
  std::string declarations;
  if (!inline_values_) {
    if (vulkan_support_) {
      VulkanConstantsProcessor processor;
      for (const auto& name : uniform_parameters_) {
        const auto& variable = name_to_variable_.at(name);
        processor.ProcessVulkanConstant(variable, &declarations);
      }
      processor.GeneratePushConstantsDeclarations(&declarations);
    } else {
      for (const auto& name : uniform_parameters_) {
        const auto& variable = name_to_variable_.at(name);
        GenerateUniformParameterDeclaration(variable, &declarations);
      }
    }
  }
  return declarations;
}

std::vector<Variable> VariableAccessor::GetUniformParameters() const {
  std::vector<Variable> variables;
  if (!inline_values_) {
    variables.reserve(name_to_variable_.size());
    // Keep the order of the variables consistent with that of the declarations
    for (const auto& name : uniform_parameters_) {
      const auto& variable = name_to_variable_.at(name);
      variables.push_back(variable);
    }
  }
  return variables;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
