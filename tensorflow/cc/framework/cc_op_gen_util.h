/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_FRAMEWORK_CC_OP_GEN_UTIL_H_
#define TENSORFLOW_CC_FRAMEWORK_CC_OP_GEN_UTIL_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace cc_op {

absl::StatusOr<ApiDefMap> LoadOpsAndApiDefs(
    OpList& ops, bool include_internal,
    const std::vector<string>& api_def_dirs);

// Converts:
//   bazel-out/.../(bin|genfiles)/(external/YYY/)?XX
// to: XX.
string GetPath(absl::string_view dot_h_fname);

// Converts: some/path/to/file.xx
// to: file
// (note that suffix is removed)
string GetFilename(absl::string_view path);

// Converts:
//   cc/ops/gen_foo_ops.h
// to:
//   CC_OPS_GEN_FOO_OPS_H_
string ToGuard(absl::string_view path);

// Converts: some_name_xyz
// to: Some Name Xyz
string ToTitle(absl::string_view name);

// Change:     Into:
//   ABC         /// ABC
//               ///
//   DEF         /// DEF
string MakeComment(absl::string_view text, absl::string_view indent);

string PrintString(absl::string_view str);

string PrintTensorShape(const TensorShapeProto& shape_proto);

template <typename T>
string PrintArray(int64_t num_elts, const T* array) {
  string ret;
  for (int64_t i = 0; i < num_elts; ++i) {
    if (i > 0) absl::StrAppend(&ret, ", ");
    absl::StrAppend(&ret, strings::LegacyPrecision(array[i]));
  }
  return ret;
}

string PrintTensor(const TensorProto& tensor_proto);

string PrintTensorProto(const TensorProto& proto);

string PrintAttrValue(absl::string_view, const AttrValue& attr_value);

bool IsEmptyList(const AttrValue::ListValue& list);

string ToCamelCase(absl::string_view str);

string SeparateNamespaces(absl::string_view str);

// Returns a <string, bool> pair. The string is the C++ type name to be used for
// attr_type when defining an object of that type. The bool is a flag to
// indicate whether to treat the type as const when accepting the C++ type as an
// argument to a function.
std::pair<absl::string_view, bool> AttrTypeName(absl::string_view attr_type);

absl::string_view ListElementTypeName(absl::string_view attr_type);

bool IsCPPKeyword(absl::string_view name);

string AvoidCPPKeywords(absl::string_view name);

void InferArgAttributes(const OpDef::ArgDef& arg,
                        std::unordered_map<string, string>* inferred_attrs);

void InferOpAttributes(
    const OpDef& op_def,
    std::unordered_map<string, string>* inferred_input_attrs);

bool ArgIsList(const OpDef::ArgDef& arg);

bool HasOptionalAttrs(
    const ApiDef& api_def,
    const std::unordered_map<string, string>& inferred_input_attrs);

struct OpInfo {
  // graph_op_def: The OpDef used by the runtime, has the names that
  //   must be used when calling NodeBuilder.
  // interface_op_def: The OpDef used in the interface in the generated
  //   code, with possibly overridden names and defaults.
  OpInfo(const OpDef& graph_op_def, const ApiDef& api_def,
         const std::vector<string>& aliases);
  OpInfo(const OpDef& graph_op_def, const ApiDef& api_def);
  string GetOpAttrStruct() const;
  string GetConstructorDecl(absl::string_view op_name_prefix,
                            bool include_attr) const;

  string op_name;
  std::vector<string> arg_types;
  std::vector<string> arg_names;
  std::vector<string> output_types;
  std::vector<string> output_names;
  std::vector<bool> is_list_output;
  bool has_optional_attrs;
  string comment;

  const OpDef& graph_op_def;
  const ApiDef& api_def;
  const std::vector<string>& aliases;
  // Map from type attribute to corresponding original argument name.
  std::unordered_map<string, string> inferred_input_attrs;
};

}  // namespace cc_op
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_CC_OP_GEN_UTIL_H_
