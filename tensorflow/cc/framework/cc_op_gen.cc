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

#include "tensorflow/cc/framework/cc_op_gen.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

const int kRightMargin = 79;

// Converts:
//   bazel-out/.../(bin|genfiles)/(external/YYY/)?XX
// to: XX.
string GetPath(const string& dot_h_fname) {
  auto pos = dot_h_fname.find("/bin/");
  string result = dot_h_fname;
  if (pos != string::npos) {
    // - 1 account for the terminating null character (\0) in "/genfiles/".
    result = dot_h_fname.substr(pos + sizeof("/bin/") - 1);
  } else {
    pos = dot_h_fname.find("/genfiles/");
    if (pos != string::npos) {
      result = dot_h_fname.substr(pos + sizeof("/genfiles/") - 1);
    }
  }
  if (result.size() > sizeof("external/") &&
      result.compare(0, sizeof("external/") - 1, "external/") == 0) {
    result = result.substr(sizeof("external/") - 1);
    pos = result.find('/');
    if (pos != string::npos) {
      result = result.substr(pos + 1);
    }
  }
  return result;
}

// Converts: some/path/to/file.xx
// to: file
// (note that suffix is removed)
string GetFilename(const string& path) {
  size_t slash_pos = path.rfind('/');
  if (slash_pos == path.npos) slash_pos = -1;
  size_t dot_pos = path.rfind('.');
  return path.substr(slash_pos + 1, dot_pos - (slash_pos + 1));
}

// Converts:
//   cc/ops/gen_foo_ops.h
// to:
//   CC_OPS_GEN_FOO_OPS_H_
string ToGuard(const string& path) {
  string guard;
  guard.reserve(path.size() + 1);  // + 1 -> trailing _
  for (const char c : path) {
    if (c >= 'A' && c <= 'Z') {
      guard += c;
    } else if (c >= 'a' && c <= 'z') {
      guard += c + 'A' - 'a';
    } else {
      guard += '_';
    }
  }
  guard += '_';
  return guard;
}

// Converts: some_name_xyz
// to: Some Name Xyz
string ToTitle(const string& name) {
  string title = name;
  for (int i = 0; i < title.size(); ++i) {
    if (title[i] == '_') title[i] = ' ';
  }
  str_util::TitlecaseString(&title, " ");
  return title;
}

// Change:     Into:
//   ABC         /// ABC
//               ///
//   DEF         /// DEF
string MakeComment(StringPiece text, StringPiece indent) {
  string ret;
  while (!text.empty()) {
    int last_non_space = -1;
    int newline;
    for (newline = 0; newline < static_cast<int>(text.size()); ++newline) {
      if (text[newline] == '\n') break;
      if (text[newline] != ' ') last_non_space = newline;
    }
    if (last_non_space == -1) {
      strings::StrAppend(&ret, indent, "///\n");
    } else {
      strings::StrAppend(&ret, indent, "/// ",
                         text.substr(0, last_non_space + 1), "\n");
    }
    text.remove_prefix(newline + 1);
  }
  return ret;
}

string PrintString(const string& str) {
  return strings::StrCat("\"", absl::CEscape(str), "\"");
}

string PrintTensorShape(const TensorShapeProto& shape_proto) {
  PartialTensorShape shape(shape_proto);
  if (shape.IsIdenticalTo(PartialTensorShape())) {
    return "::tensorflow::PartialTensorShape() /* unknown */";
  }
  string ret = "{";
  for (int d = 0; d < shape.dims(); ++d) {
    if (d > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, shape.dim_size(d));
  }
  strings::StrAppend(&ret, "}");
  return ret;
}

template <typename T>
string PrintArray(int64_t num_elts, const T* array) {
  string ret;
  for (int64_t i = 0; i < num_elts; ++i) {
    if (i > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, array[i]);
  }
  return ret;
}

string PrintTensor(const TensorProto& tensor_proto) {
  Tensor t(tensor_proto.dtype());
  CHECK(t.FromProto(tensor_proto));
  const int64 num_elts = t.NumElements();
  switch (t.dtype()) {
    case DT_FLOAT:
      return PrintArray(num_elts, t.flat<float>().data());
    case DT_DOUBLE:
      return PrintArray(num_elts, t.flat<double>().data());
    case DT_INT32:
      return PrintArray(num_elts, t.flat<int32>().data());
    case DT_UINT8:
    case DT_QUINT8:
      return PrintArray(num_elts, t.flat<uint8>().data());
    case DT_UINT16:
    case DT_QUINT16:
      return PrintArray(num_elts, t.flat<uint16>().data());
    case DT_INT16:
    case DT_QINT16:
      return PrintArray(num_elts, t.flat<int16>().data());
    case DT_INT8:
    case DT_QINT8:
      return PrintArray(num_elts, t.flat<int8>().data());
    case DT_INT64:
      return PrintArray(num_elts, t.flat<int64>().data());
    case DT_BOOL:
      return PrintArray(num_elts, t.flat<bool>().data());
    case DT_STRING: {
      string ret;
      for (int64_t i = 0; i < num_elts; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        strings::StrAppend(&ret, absl::CEscape(t.flat<tstring>()(i)));
      }
      return ret;
    }
    default: {
      LOG(FATAL) << "Not handling type " << DataType_Name(t.dtype());
      return string();
    }
  }
}

string PrintTensorProto(const TensorProto& proto) {
  return strings::StrCat("Input::Initializer(", "{", PrintTensor(proto), "}, ",
                         PrintTensorShape(proto.tensor_shape()),
                         ").AsTensorProto()");
}

string PrintAttrValue(const string& op, const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return PrintString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF: {
      const float f = attr_value.f();
      return strings::StrCat(attr_value.f(), floorf(f) == f ? ".0" : "", "f");
    }
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return DataType_Name(attr_value.type());
    case AttrValue::kShape:
      return PrintTensorShape(attr_value.shape());
    case AttrValue::kTensor:
      return PrintTensorProto(attr_value.tensor());
    case AttrValue::kList: {
      string ret = "{";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, PrintString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().i(i));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          const float f = attr_value.list().f(i);
          strings::StrAppend(&ret, f, floorf(f) == f ? ".0" : "", "f");
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, DataType_Name(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             PrintTensorShape(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             PrintTensorProto(attr_value.list().tensor(i)));
        }
      }
      strings::StrAppend(&ret, "}");
      return ret;
    }
    default:
      LOG(FATAL) << "Unsupported Attr type: " << op << " "
                 << attr_value.value_case();
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

bool IsEmptyList(const AttrValue::ListValue& list) {
  return list.s_size() == 0 && list.i_size() == 0 && list.f_size() == 0 &&
         list.b_size() == 0 && list.type_size() == 0 &&
         list.shape_size() == 0 && list.tensor_size() == 0;
}

string ToCamelCase(const string& str) {
  string result;
  const char joiner = '_';
  size_t i = 0;
  bool cap = true;
  while (i < str.size()) {
    const char c = str[i++];
    if (c == '>') {
      cap = true;
    } else if (c == joiner) {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

string SeparateNamespaces(const string& str) {
  string result;
  const char joiner = '_';
  size_t i = 0;
  while (i < str.size()) {
    const char c = str[i++];
    if (c == '>') {
      result += joiner;
    } else {
      result += c;
    }
  }
  return result;
}

// Returns a <string, bool> pair. The string is the C++ type name to be used for
// attr_type when defining an object of that type. The bool is a flag to
// indicate whether to treat the type as const when accepting the C++ type as an
// argument to a function.
std::pair<const char*, bool> AttrTypeName(StringPiece attr_type) {
  static const auto* attr_type_map =
      new std::unordered_map<StringPiece, std::pair<const char*, bool>,
                             StringPieceHasher>{
          {"string", {"StringPiece", false}},
          {"list(string)", {"gtl::ArraySlice<::tensorflow::tstring>", true}},
          {"int", {"int64", false}},
          {"list(int)", {"gtl::ArraySlice<int>", true}},
          {"float", {"float", false}},
          {"list(float)", {"gtl::ArraySlice<float>", true}},
          {"bool", {"bool", false}},
          {"list(bool)", {"gtl::ArraySlice<bool>", true}},
          {"type", {"DataType", false}},
          {"list(type)", {"DataTypeSlice", true}},
          {"shape", {"PartialTensorShape", false}},
          {"list(shape)", {"gtl::ArraySlice<PartialTensorShape>", true}},
          {"tensor", {"TensorProto", true}},
          {"list(tensor)", {"gtl::ArraySlice<TensorProto>", true}},
          {"func", {"NameAttrList", true}},
          {"list(func)", {"gtl::ArraySlice<NameAttrList>", true}},
      };

  auto entry = attr_type_map->find(attr_type);
  if (entry == attr_type_map->end()) {
    LOG(FATAL) << "Unsupported Attr type: " << attr_type;
    return {"", false};
  }
  return entry->second;
}

const char* ListElementTypeName(StringPiece attr_type) {
  static const auto* attr_list_type_map =
      new std::unordered_map<StringPiece, const char*, StringPieceHasher>{
          {"list(string)", "string"},
          {"list(int)", "int"},
          {"list(float)", "float"},
          {"list(bool)", "bool"},
          {"list(type)", "DataType"},
          {"list(shape)", "PartialTensorShape"},
          {"list(tensor)", "TensorProto"},
      };

  auto entry = attr_list_type_map->find(attr_type);
  if (entry == attr_list_type_map->end()) {
    LOG(FATAL) << "Unsupported or non-list Attr type: " << attr_type;
    return "";
  }
  return entry->second;
}

bool IsCPPKeyword(StringPiece name) {
  static const std::unordered_set<StringPiece, StringPieceHasher>
      // Keywords obtained from http://en.cppreference.com/w/cpp/keyword
      kCPPReserved{
          "alignas",
          "alignof",
          "and",
          "and_eq",
          "asm",
          "atomic_cancel",
          "atomic_commit",
          "atomic_noexcept",
          "auto",
          "bitand",
          "bitor",
          "bool",
          "break",
          "case",
          "catch",
          "char",
          "char16_t",
          "char32_t",
          "class",
          "compl",
          "concept",
          "const",
          "const_cast",
          "constexpr",
          "continue",
          "decltype",
          "default",
          "delete",
          "do",
          "double",
          "dynamic_cast",
          "else",
          "enum",
          "explicit",
          "export",
          "extern",
          "false",
          "final",
          "float",
          "for",
          "friend",
          "goto",
          "if",
          "import",
          "inline",
          "int",
          "long",
          "module",
          "mutable",
          "namespace",
          "new",
          "noexcept",
          "not",
          "not_eq",
          "nullptr",
          "operator",
          "or",
          "or_eq",
          "override",
          "private",
          "protected",
          "public",
          "register",
          "reinterpret_cast",
          "requires",
          "return",
          "short",
          "signed",
          "sizeof",
          "static",
          "static_assert",
          "static_cast",
          "struct",
          "switch",
          "synchronized",
          "template",
          "this",
          "thread_local",
          "throw",
          "true",
          "try",
          "typedef",
          "typeid",
          "typename",
          "union",
          "unsigned",
          "using",
          "virtual",
          "void",
          "volatile",
          "wchar_t",
          "while",
          "xor",
          "xor_eq",

          // The following are not C++ keywords, but names of local variables
          // and parameters used in the op constructor. Treating them as
          // keywords, so that other parameter names don't conflict with these.
          "builder",
          "node",
          "ret",
          "scope",
          "unique_name",
      };
  return kCPPReserved.count(name) > 0;
}

string AvoidCPPKeywords(StringPiece name) {
  if (IsCPPKeyword(name)) {
    return strings::StrCat(name, "_");
  }
  return string(name);
}

void InferArgAttributes(const OpDef::ArgDef& arg,
                        std::unordered_map<string, string>* inferred_attrs) {
  if (!arg.type_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.type_attr(), arg.name());
  } else if (!arg.type_list_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.type_list_attr(), arg.name());
  }
  if (!arg.number_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.number_attr(), arg.name());
  }
}

void InferOpAttributes(
    const OpDef& op_def,
    std::unordered_map<string, string>* inferred_input_attrs) {
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    InferArgAttributes(arg, inferred_input_attrs);
  }
}

bool ArgIsList(const OpDef::ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

bool HasOptionalAttrs(
    const ApiDef& api_def,
    const std::unordered_map<string, string>& inferred_input_attrs) {
  for (int i = 0; i < api_def.attr_size(); ++i) {
    const auto& attr(api_def.attr(i));
    if ((inferred_input_attrs.find(attr.name()) ==
         inferred_input_attrs.end()) &&
        attr.has_default_value()) {
      return true;
    }
  }
  return false;
}

struct OpInfo {
  // graph_op_def: The OpDef used by the runtime, has the names that
  //   must be used when calling NodeBuilder.
  // interface_op_def: The OpDef used in the interface in the generated
  //   code, with possibly overridden names and defaults.
  explicit OpInfo(const OpDef& graph_op_def, const ApiDef& api_def,
                  const std::vector<string>& aliases);
  string GetOpAttrStruct() const;
  string GetConstructorDecl(StringPiece op_name_prefix,
                            bool include_attr) const;
  void WriteClassDecl(WritableFile* h) const;
  void GetOutput(string* out) const;
  string GetConstructorBody() const;
  void WriteClassDef(WritableFile* cc) const;

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

OpInfo::OpInfo(const OpDef& graph_op_def, const ApiDef& api_def,
               const std::vector<string>& aliases)
    : graph_op_def(graph_op_def), api_def(api_def), aliases(aliases) {
  op_name = SeparateNamespaces(api_def.endpoint(0).name());
  InferOpAttributes(graph_op_def, &inferred_input_attrs);
  has_optional_attrs = HasOptionalAttrs(api_def, inferred_input_attrs);
  arg_types.push_back("const ::tensorflow::Scope&");
  arg_names.push_back("scope");

  if (graph_op_def.has_deprecation()) {
    if (!api_def.summary().empty()) {
      comment = strings::StrCat(api_def.summary(), "\n");
    }
    strings::StrAppend(&comment, "DEPRECATED at GraphDef version ",
                       graph_op_def.deprecation().version(), ":\n",
                       graph_op_def.deprecation().explanation(), ".\n");
  } else if (api_def.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(api_def.summary(), "\n");
  }
  if (!api_def.description().empty()) {
    strings::StrAppend(&comment, "\n", api_def.description(), "\n");
  }
  strings::StrAppend(&comment, "\nArgs:\n* scope: A Scope object\n");

  // Process inputs
  for (int i = 0; i < api_def.arg_order_size(); ++i) {
    const auto& arg = *FindInputArg(api_def.arg_order(i), graph_op_def);
    const auto& api_def_arg = *FindInputArg(api_def.arg_order(i), api_def);
    arg_types.push_back(strings::StrCat(
        "::tensorflow::", ArgIsList(arg) ? "InputList" : "Input"));
    arg_names.push_back(AvoidCPPKeywords(api_def_arg.rename_to()));

    // TODO(keveman): Include input type information.
    StringPiece description = api_def_arg.description();
    if (!description.empty()) {
      ConsumeEquals(&description);
      strings::StrAppend(&comment, "* ",
                         AvoidCPPKeywords(api_def_arg.rename_to()), ": ",
                         api_def_arg.description(), "\n");
    }
  }

  // Process attrs
  string required_attrs_comment;
  string optional_attrs_comment;
  for (int i = 0; i < graph_op_def.attr_size(); ++i) {
    // ApiDef attributes must be in the same order as in OpDef since
    // we initialize ApiDef based on OpDef.
    const auto& attr(graph_op_def.attr(i));
    const auto& api_def_attr(api_def.attr(i));
    CHECK_EQ(attr.name(), api_def_attr.name());
    // Skip inferred arguments
    if (inferred_input_attrs.count(attr.name()) > 0) continue;

    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;
    string attr_name = AvoidCPPKeywords(api_def_attr.rename_to());

    string attr_comment;
    if (!api_def_attr.description().empty()) {
      // TODO(keveman): Word wrap and indent this, to handle multi-line
      // descriptions.
      strings::StrAppend(&attr_comment, "* ", attr_name, ": ",
                         api_def_attr.description(), "\n");
    }
    if (api_def_attr.has_default_value()) {
      strings::StrAppend(&optional_attrs_comment, attr_comment);
    } else {
      strings::StrAppend(&required_attrs_comment, attr_comment);
      arg_types.push_back(strings::StrCat(
          use_const ? "const " : "", attr_type_name, use_const ? "&" : ""));
      arg_names.push_back(attr_name);
    }
  }

  strings::StrAppend(&comment, required_attrs_comment);

  if (!optional_attrs_comment.empty()) {
    strings::StrAppend(&comment, "\nOptional attributes (see `Attrs`):\n");
    strings::StrAppend(&comment, optional_attrs_comment);
  }

  // Process outputs
  for (int i = 0; i < graph_op_def.output_arg_size(); ++i) {
    // ApiDef arguments must be in the same order as in OpDef since
    // we initialize ApiDef based on OpDef.
    const auto& arg = graph_op_def.output_arg(i);
    const auto& api_def_arg(api_def.out_arg(i));
    CHECK_EQ(arg.name(), api_def_arg.name());

    bool is_list = ArgIsList(arg);
    output_types.push_back(
        strings::StrCat("::tensorflow::", is_list ? "OutputList" : "Output"));
    output_names.push_back(AvoidCPPKeywords(api_def_arg.rename_to()));
    is_list_output.push_back(is_list);
  }

  strings::StrAppend(&comment, "\nReturns:\n");
  if (graph_op_def.output_arg_size() == 0) {  // No outputs.
    strings::StrAppend(&comment, "* the created `Operation`\n");
  } else if (graph_op_def.output_arg_size() == 1) {  // One output
    if (is_list_output[0]) {
      strings::StrAppend(&comment, "* `OutputList`: ");
    } else {
      strings::StrAppend(&comment, "* `Output`: ");
    }
    if (api_def.out_arg(0).description().empty()) {
      strings::StrAppend(&comment, "The ", api_def.out_arg(0).name(),
                         " tensor.\n");
    } else {
      // TODO(josh11b): Word wrap this.
      strings::StrAppend(&comment, api_def.out_arg(0).description(), "\n");
    }
  } else {  // Multiple outputs.
    for (int i = 0; i < graph_op_def.output_arg_size(); ++i) {
      if (is_list_output[i]) {
        strings::StrAppend(&comment, "* `OutputList`");
      } else {
        strings::StrAppend(&comment, "* `Output`");
      }
      strings::StrAppend(&comment, " ", output_names[i]);
      if (api_def.out_arg(i).description().empty()) {
        strings::StrAppend(&comment, "\n");
      } else {
        // TODO(josh11b): Word wrap this.
        strings::StrAppend(&comment, ": ", api_def.out_arg(i).description(),
                           "\n");
      }
    }
  }

  if (!aliases.empty()) {
    strings::StrAppend(&comment, "\nAliases:\n");
    for (const auto& alias : aliases) {
      strings::StrAppend(&comment, "* ", alias, "\n");
    }
  }
  comment = MakeComment(comment, "");
}

string OpInfo::GetOpAttrStruct() const {
  string struct_fields;
  string setters;
  string defaults_static_storage;

  for (int i = 0; i < graph_op_def.attr_size(); ++i) {
    const auto& attr(graph_op_def.attr(i));
    const auto& api_def_attr(api_def.attr(i));
    // If attr will be inferred or it doesn't have a default value, don't
    // add it to the struct.
    if ((inferred_input_attrs.find(attr.name()) !=
         inferred_input_attrs.end()) ||
        !api_def_attr.has_default_value()) {
      continue;
    }
    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;
    const string camel_case_name = ToCamelCase(api_def_attr.rename_to());
    const string suffix =
        (camel_case_name == op_name || camel_case_name == "Attrs") ? "_" : "";
    const string attr_func_def =
        strings::StrCat(camel_case_name, suffix, "(", use_const ? "const " : "",
                        attr_type_name, use_const ? "&" : "");

    string attr_comment;
    if (!api_def_attr.description().empty()) {
      strings::StrAppend(&attr_comment, api_def_attr.description(), "\n\n");
    }
    strings::StrAppend(&attr_comment, "Defaults to ",
                       SummarizeAttrValue(api_def_attr.default_value()), "\n");
    attr_comment = MakeComment(attr_comment, "    ");

    strings::StrAppend(&setters, attr_comment);
    strings::StrAppend(&setters, "    TF_MUST_USE_RESULT Attrs ", attr_func_def,
                       " x) {\n");
    strings::StrAppend(&setters, "      Attrs ret = *this;\n");
    strings::StrAppend(&setters, "      ret.", api_def_attr.rename_to(),
                       "_ = x;\n");
    strings::StrAppend(&setters, "      return ret;\n    }\n\n");

    string field_initiliazer;
    auto& default_value = api_def_attr.default_value();
    if (default_value.value_case() == AttrValue::kList &&
        !IsEmptyList(default_value.list())) {
      // Non-empty lists need static storage for their defaults. Define a
      // function with static local variable that stores the array.
      strings::StrAppend(&defaults_static_storage, "    static ",
                         attr_type_name, " Default_", api_def_attr.rename_to(),
                         "() {\n");
      strings::StrAppend(
          &defaults_static_storage, "      static const ",
          ListElementTypeName(attr.type()), " kStorage[] = ",
          PrintAttrValue(graph_op_def.name(), api_def_attr.default_value()),
          ";\n");
      strings::StrAppend(&defaults_static_storage, "      return ",
                         attr_type_name, "(kStorage);\n    }\n");
      // Set the field_initializer to call the defined function.
      strings::StrAppend(&field_initiliazer, "Default_",
                         api_def_attr.rename_to(), "()");
    } else {
      field_initiliazer =
          PrintAttrValue(graph_op_def.name(), api_def_attr.default_value());
    }
    strings::StrAppend(&struct_fields, "    ", attr_type_name, " ",
                       api_def_attr.rename_to(), "_ = ", field_initiliazer,
                       ";\n");
  }

  if (struct_fields.empty()) {
    return "";
  }

  string attrs_comment =
      strings::StrCat("Optional attribute setters for ", op_name, "\n");
  string struct_decl = MakeComment(attrs_comment, "  ");
  strings::StrAppend(&struct_decl, "  struct Attrs {\n");
  strings::StrAppend(&struct_decl, setters, struct_fields);
  if (!defaults_static_storage.empty()) {
    strings::StrAppend(&struct_decl, "  private:\n", defaults_static_storage);
  }
  strings::StrAppend(&struct_decl, "  };\n");

  return struct_decl;
}

string OpInfo::GetConstructorDecl(StringPiece op_name_prefix,
                                  bool include_attr) const {
  const string prefix = strings::StrCat(op_name_prefix, op_name, "(");
  string c_decl;
  for (int i = 0; i < arg_types.size(); ++i) {
    if (i > 0) strings::StrAppend(&c_decl, ", ");
    strings::StrAppend(&c_decl, arg_types[i], " ", arg_names[i]);
  }
  if (include_attr && has_optional_attrs) {
    strings::StrAppend(&c_decl, ", const ", op_name, "::Attrs& attrs");
  }
  strings::StrAppend(&c_decl, ")");
  return WordWrap(prefix, c_decl, kRightMargin);
}

void OpInfo::WriteClassDecl(WritableFile* h) const {
  string class_decl = comment;
  strings::StrAppend(&class_decl, "class ", op_name, " {\n");
  strings::StrAppend(&class_decl, " public:\n");
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, GetOpAttrStruct());
  }
  strings::StrAppend(&class_decl, "  ",
                     GetConstructorDecl("", /* include_attr */ false), ";\n");
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, "  ",
                       GetConstructorDecl("", /* include_attr */ true), ";\n");
  }
  if (output_types.empty()) {
    // Allow casting this class to Operation.
    strings::StrAppend(&class_decl,
                       "  operator ::tensorflow::Operation() const { "
                       "return operation; }\n");
  } else if (output_types.size() == 1) {
    if (is_list_output[0]) {
      // Write the subscript operator, allowing out[i] for the list-typed
      // output.
      strings::StrAppend(&class_decl,
                         "  ::tensorflow::Output operator[](size_t index) "
                         "const { return ",
                         output_names[0], "[index]; }\n\n");

    } else {
      // Write type cast functions, allowing casting this class to Input and
      // Output.
      strings::StrAppend(&class_decl,
                         "  operator ::tensorflow::Output() const { return ",
                         output_names[0], "; }\n");
      strings::StrAppend(&class_decl,
                         "  operator ::tensorflow::Input() const { return ",
                         output_names[0], "; }\n");
      // Write node() to get the Node* directly.
      strings::StrAppend(&class_decl,
                         "  ::tensorflow::Node* node() const { return ",
                         output_names[0], ".node(); }\n");
    }
  }
  // Add the static functions to set optional attrs
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, "\n");
    for (int i = 0; i < graph_op_def.attr_size(); ++i) {
      const auto& attr(graph_op_def.attr(i));
      const auto& api_def_attr(api_def.attr(i));
      if ((inferred_input_attrs.find(attr.name()) !=
           inferred_input_attrs.end()) ||
          !api_def_attr.has_default_value()) {
        continue;
      }
      const auto entry = AttrTypeName(attr.type());
      const auto attr_type_name = entry.first;
      const bool use_const = entry.second;
      const string camel_case_name = ToCamelCase(api_def_attr.rename_to());
      const string suffix =
          (camel_case_name == op_name || camel_case_name == "Attrs") ? "_" : "";
      const string attr_func_def = strings::StrCat(
          camel_case_name, suffix, "(", use_const ? "const " : "",
          attr_type_name, use_const ? "&" : "");
      strings::StrAppend(&class_decl, "  static Attrs ", attr_func_def,
                         " x) {\n");
      strings::StrAppend(&class_decl, "    return Attrs().", camel_case_name,
                         suffix, "(x);\n");
      strings::StrAppend(&class_decl, "  }\n");
    }
  }

  strings::StrAppend(&class_decl, "\n  Operation operation;\n");
  for (int i = 0; i < output_types.size(); ++i) {
    strings::StrAppend(&class_decl, "  ", output_types[i], " ", output_names[i],
                       ";\n");
  }

  strings::StrAppend(&class_decl, "};\n");
  if (!aliases.empty()) {
    for (const auto& alias : aliases) {
      strings::StrAppend(&class_decl, "typedef ", op_name, " ", alias, ";\n");
    }
  }
  strings::StrAppend(&class_decl, "\n");
  TF_CHECK_OK(h->Append(class_decl));
}

void OpInfo::GetOutput(string* out) const {
  const string scope_str = arg_names[0];
  string return_on_error =
      strings::StrCat("if (!", scope_str, ".ok()) return;");

  strings::StrAppend(out, "  this->operation = Operation(ret);\n");

  // No outputs.
  if (graph_op_def.output_arg_size() == 0) {
    strings::StrAppend(out, "  return;\n");
    return;
  }
  if (graph_op_def.output_arg_size() == 1) {
    // One output, no need for NameRangeMap
    if (is_list_output[0]) {
      strings::StrAppend(out,
                         "  for (int32 i = 0; i < ret->num_outputs(); ++i)\n");
      strings::StrAppend(out, "    this->", output_names[0],
                         ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", output_names[0],
                         " = Output(ret, 0);\n");
    }
    return;
  }
  strings::StrAppend(out, "  ::tensorflow::NameRangeMap _outputs_range;\n");
  strings::StrAppend(out,
                     "  ::tensorflow::Status _status_ = "
                     "::tensorflow::NameRangesForNode(*ret, ret->op_def(), "
                     "nullptr, &_outputs_range);\n");
  strings::StrAppend(out, "  if (!_status_.ok()) {\n", "    ", scope_str,
                     ".UpdateStatus(_status_);\n", "    return;\n");
  strings::StrAppend(out, "  }\n\n");

  for (int i = 0; i < graph_op_def.output_arg_size(); ++i) {
    const string arg_range = strings::StrCat(
        "_outputs_range[\"", graph_op_def.output_arg(i).name(), "\"]");
    if (is_list_output[i]) {
      strings::StrAppend(out, "  for (int32 i = ", arg_range, ".first; i < ",
                         arg_range, ".second; ++i)\n");
      strings::StrAppend(out, "    this->", output_names[i],
                         ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", output_names[i], " = Output(ret, ",
                         arg_range, ".first);\n");
    }
  }
}

string OpInfo::GetConstructorBody() const {
  const string scope_str = arg_names[0];

  string body;
  string return_on_error =
      strings::StrCat("if (!", scope_str, ".ok()) return;");

  strings::StrAppend(&body, "  ", return_on_error, "\n");

  for (int i = 0; i < graph_op_def.input_arg_size(); ++i) {
    const auto& arg(graph_op_def.input_arg(i));
    const auto& api_def_arg(api_def.in_arg(i));
    strings::StrAppend(
        &body, "  auto _", api_def_arg.rename_to(), " = ::tensorflow::ops::",
        ArgIsList(arg) ? "AsNodeOutList" : "AsNodeOut", "(", scope_str, ", ",
        AvoidCPPKeywords(api_def_arg.rename_to()), ");\n");
    strings::StrAppend(&body, "  ", return_on_error, "\n");
  }

  strings::StrAppend(&body, "  ::tensorflow::Node* ret;\n");
  strings::StrAppend(&body, "  const auto unique_name = ", scope_str,
                     ".GetUniqueNameForOp(\"", op_name, "\");\n");
  strings::StrAppend(
      &body, "  auto builder = ::tensorflow::NodeBuilder(unique_name, \"",
      graph_op_def.name(), "\")\n");
  const string spaces = "                     ";
  for (int i = 0; i < api_def.in_arg_size(); ++i) {
    const auto& arg(api_def.in_arg(i));
    strings::StrAppend(&body, spaces, ".Input(_", arg.rename_to(), ")\n");
  }
  for (int i = 0; i < api_def.attr_size(); ++i) {
    const auto& graph_attr(graph_op_def.attr(i));
    const auto& api_def_attr(api_def.attr(i));
    if (inferred_input_attrs.find(api_def_attr.name()) !=
        inferred_input_attrs.end()) {
      continue;
    }
    const string attr_name =
        api_def_attr.has_default_value()
            ? strings::StrCat("attrs.", api_def_attr.rename_to(), "_")
            : AvoidCPPKeywords(api_def_attr.rename_to());
    strings::StrAppend(&body, spaces, ".Attr(\"", graph_attr.name(), "\", ",
                       attr_name, ")\n");
  }
  strings::StrAppend(&body, "  ;\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateBuilder(&builder);\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(builder.Finalize(",
                     scope_str, ".graph(), &ret));\n");
  strings::StrAppend(&body, "  ", return_on_error, "\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(", scope_str,
                     ".DoShapeInference(ret));\n");

  GetOutput(&body);
  return body;
}

void OpInfo::WriteClassDef(WritableFile* cc) const {
  string class_def;
  strings::StrAppend(&class_def,
                     GetConstructorDecl(strings::StrCat(op_name, "::"),
                                        /* include_attr */ true),
                     " {\n");
  strings::StrAppend(&class_def, GetConstructorBody());
  strings::StrAppend(&class_def, "}\n\n");

  if (has_optional_attrs) {
    strings::StrAppend(&class_def,
                       GetConstructorDecl(strings::StrCat(op_name, "::"),
                                          /* include_attr */ false));
    strings::StrAppend(&class_def, "\n  : ", op_name, "(");
    int i = 0;
    for (; i < arg_names.size(); ++i) {
      if (i > 0) strings::StrAppend(&class_def, ", ");
      strings::StrAppend(&class_def, arg_names[i]);
    }
    if (i > 0) strings::StrAppend(&class_def, ", ");
    strings::StrAppend(&class_def, op_name, "::Attrs()");
    strings::StrAppend(&class_def, ") {}\n\n");
  }
  TF_CHECK_OK(cc->Append(class_def));
}

void WriteCCOp(const OpDef& graph_op_def, const ApiDef& api_def,
               const std::vector<string>& aliases, WritableFile* h,
               WritableFile* cc) {
  OpInfo op_info(graph_op_def, api_def, aliases);

  op_info.WriteClassDecl(h);
  op_info.WriteClassDef(cc);
}

void StartFiles(bool internal, const string& dot_h_fname, WritableFile* h,
                WritableFile* cc, string* op_header_guard) {
  const string header =
      R"header(// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
)header";

  // TODO(keveman): Make namespaces configurable.
  const string namespace_begin = internal ? R"namespace(
namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

)namespace"
                                          : R"namespace(
namespace tensorflow {
namespace ops {

)namespace";

  const string op_header = GetPath(dot_h_fname);
  *op_header_guard = ToGuard(op_header);
  const string cc_header = strings::StrCat(
      R"include(// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
)include",
      "#include \"", op_header, "\"\n", namespace_begin);

  const string filename = GetFilename(dot_h_fname);
  const string doxygen = strings::StrCat("/// @defgroup ", filename, " ",
                                         ToTitle(filename), "\n", "/// @{\n\n");

  TF_CHECK_OK(h->Append(
      strings::StrCat("// This file is MACHINE GENERATED! Do not edit.\n\n"
                      "#ifndef ",
                      *op_header_guard,
                      "\n"
                      "#define ",
                      *op_header_guard, "\n\n")));
  TF_CHECK_OK(h->Append(header));
  TF_CHECK_OK(h->Append(namespace_begin));
  TF_CHECK_OK(h->Append(doxygen));
  TF_CHECK_OK(cc->Append(cc_header));
}

void FinishFiles(bool internal, WritableFile* h, WritableFile* cc,
                 const string& op_header_guard) {
  const string footer = internal ? R"footer(}  // namespace internal
}  // namespace ops
}  // namespace tensorflow
)footer"
                                 :
                                 R"footer(/// @}

}  // namespace ops
}  // namespace tensorflow
)footer";

  TF_CHECK_OK(h->Append(footer));
  TF_CHECK_OK(
      h->Append(strings::StrCat("\n#endif  ", "// ", op_header_guard, "\n")));
  TF_CHECK_OK(cc->Append(footer));

  TF_CHECK_OK(cc->Close());
  TF_CHECK_OK(h->Close());
}

string MakeInternal(const string& fname) {
  auto dot_pos = fname.rfind('.');
  if (dot_pos == string::npos) {
    return strings::StrCat(fname, "_internal");
  } else {
    return strings::StrCat(fname.substr(0, dot_pos), "_internal",
                           fname.substr(dot_pos));
  }
}

}  // namespace

void WriteCCOps(const OpList& ops, const ApiDefMap& api_def_map,
                const string& dot_h_fname, const string& dot_cc_fname) {
  Env* env = Env::Default();

  // Write the initial boilerplate to the .h and .cc files.
  std::unique_ptr<WritableFile> h = nullptr;
  std::unique_ptr<WritableFile> cc = nullptr;
  TF_CHECK_OK(env->NewWritableFile(dot_h_fname, &h));
  TF_CHECK_OK(env->NewWritableFile(dot_cc_fname, &cc));
  string op_header_guard;
  StartFiles(false, dot_h_fname, h.get(), cc.get(), &op_header_guard);

  // Create the internal versions of these files for the hidden ops.
  std::unique_ptr<WritableFile> internal_h = nullptr;
  std::unique_ptr<WritableFile> internal_cc = nullptr;
  const string internal_dot_h_fname = MakeInternal(dot_h_fname);
  TF_CHECK_OK(env->NewWritableFile(internal_dot_h_fname, &internal_h));
  TF_CHECK_OK(env->NewWritableFile(MakeInternal(dot_cc_fname), &internal_cc));
  string internal_op_header_guard;
  StartFiles(true /* internal */, internal_dot_h_fname, internal_h.get(),
             internal_cc.get(), &internal_op_header_guard);

  for (const auto& graph_op_def : ops.op()) {
    // Skip deprecated ops.
    // TODO(josh11b): If needed, can put them into a "deprecated" namespace
    // instead of skipping.
    if (graph_op_def.has_deprecation() &&
        graph_op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
      continue;
    }

    // We use a hand-written wrapper for "Const", since the generated
    // code depends on it.
    if (graph_op_def.name() == "Const") continue;

    const auto* api_def = api_def_map.GetApiDef(graph_op_def.name());

    std::vector<string> aliases;
    if (api_def->visibility() == ApiDef::SKIP) continue;
    // First endpoint is canonical, the rest are aliases.
    for (int endpoint_i = 1; endpoint_i < api_def->endpoint_size();
         ++endpoint_i) {
      aliases.push_back(api_def->endpoint(endpoint_i).name());
    }
    if (api_def->visibility() == ApiDef::HIDDEN) {
      // Write hidden ops to _internal.h and _internal.cc.
      WriteCCOp(graph_op_def, *api_def, aliases, internal_h.get(),
                internal_cc.get());
      continue;
    }
    // This isn't a hidden op, write it to the main files.
    WriteCCOp(graph_op_def, *api_def, aliases, h.get(), cc.get());
  }

  FinishFiles(false, h.get(), cc.get(), op_header_guard);
  FinishFiles(true /* internal */, internal_h.get(), internal_cc.get(),
              internal_op_header_guard);
}

}  // namespace tensorflow
