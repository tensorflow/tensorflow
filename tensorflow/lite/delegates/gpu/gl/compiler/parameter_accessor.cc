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

#include "tensorflow/lite/delegates/gpu/gl/compiler/parameter_accessor.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace parameter_accessor_internal {

// Parse the following regex manually
// name(\[index\])?(\.field)?
ParameterReference Parse(absl::string_view input) {
  ParameterReference ref;
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

}  // namespace parameter_accessor_internal

namespace {

struct UniformTypeGetter {
  std::string operator()(int) const { return "int"; }
  std::string operator()(const int2&) const { return "ivec2"; }
  std::string operator()(const std::vector<int2>&) const { return "ivec2"; }
  std::string operator()(const int4&) const { return "ivec4"; }
  std::string operator()(unsigned int) const { return "uint"; }
  std::string operator()(const uint4&) const { return "uvec4"; }
  std::string operator()(float) const { return "float"; }
  std::string operator()(const float2&) const { return "vec2"; }
  std::string operator()(const float4&) const { return "vec4"; }
};

// Returns GLSL uniform type of the given parameter.
std::string GetUniformType(const UniformParameter::ValueType& value) {
  return absl::visit(UniformTypeGetter(), value);
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
    absl::StrAppend(result, UniformTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 2>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec3<T>& v) const {
    absl::StrAppend(result, UniformTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 3>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec4<T>& v) const {
    absl::StrAppend(result, UniformTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 4>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    std::string type = UniformTypeGetter()(v);
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

// Appends string representation of a parameter value.
void GetValue(const UniformParameter::ValueType& value, std::string* result) {
  absl::visit(ConstGenerator{result}, value);
}

struct UniformDeclarationGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "uniform ", GetUniformType(param.value), " ",
                    param.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "uniform ", GetUniformType(param.value), " ",
                    param.name, "[", v.size(), "];\n");
  }

  const UniformParameter& param;
  std::string* result;
};

void GenerateUniformDeclaration(const UniformParameter& parameter,
                                std::string* result) {
  absl::visit(UniformDeclarationGenerator{parameter, result}, parameter.value);
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

// Returns true if value is a vector
bool IsVariableLength(const UniformParameter::ValueType& value) {
  return absl::visit(VariableLengthGetter(), value);
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
void GetValue(const UniformParameter::ValueType& value, Field field,
              std::string* result) {
  absl::visit(FieldAccessor{field, result}, value);
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
bool HasField(const UniformParameter::ValueType& value, Field field) {
  return absl::visit(FieldChecker{field}, value);
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

RewriteStatus ParameterAccessor::Rewrite(absl::string_view input,
                                         std::string* output) {
  auto ref = parameter_accessor_internal::Parse(input);
  if (ref.name.empty()) {
    absl::StrAppend(output, "INVALID_SYNTAX");
    return RewriteStatus::ERROR;
  }

  auto it = name_to_param_.find(std::string(ref.name.data(), ref.name.size()));
  if (it == name_to_param_.end()) {
    // Uniform with this name is not registered.
    return RewriteStatus::NOT_RECOGNIZED;
  }
  const auto& value = it->second.value;

  if (!ref.index.empty() && !IsVariableLength(value)) {
    // Trying to access parameter by index, but it is not variable-length.
    absl::StrAppend(output, "INVALID_ACCESS_BY_INDEX");
    return RewriteStatus::ERROR;
  }

  Field f = ToField(ref.field);
  if (!ref.field.empty() && !HasField(value, f)) {
    // Trying to access a parameter by field, but it does not have it.
    absl::StrAppend(output, "INVALID_ACCESS_BY_FIELD");
    return RewriteStatus::ERROR;
  }

  // Error checks are complete now.

  // All variable-length parameters are encoded as-is without inlining.
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

bool ParameterAccessor::AddParameter(UniformParameter param) {
  std::string name = param.name;
  return name_to_param_.insert({name, std::move(param)}).second;
}

std::string ParameterAccessor::GetConstDeclarations() const {
  // Variable length parameters are declared as const and accessed via variable
  // with index.
  std::string declarations;
  for (auto& param : name_to_param_) {
    const auto& value = param.second.value;
    if (IsVariableLength(value)) {
      absl::StrAppend(&declarations, "const ", GetUniformType(value), " ",
                      param.second.name, "[] = ");
      GetValue(value, &declarations);
      absl::StrAppend(&declarations, ";\n");
    }
  }
  return declarations;
}

std::string ParameterAccessor::GetUniformDeclarations() const {
  std::string declarations;
  if (!inline_values_) {
    for (auto& param : name_to_param_) {
      GenerateUniformDeclaration(param.second, &declarations);
    }
  }
  return declarations;
}

std::vector<UniformParameter> ParameterAccessor::GetUniformParameters() const {
  std::vector<UniformParameter> params;
  if (!inline_values_) {
    for (auto& param : name_to_param_) {
      params.push_back(param.second);
    }
  }
  return params;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
