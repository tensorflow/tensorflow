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

#include "tensorflow/cc/framework/cc_op_gen_util.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace cc_op {

tsl::StatusOr<ApiDefMap> LoadOpsAndApiDefs(
    OpList& ops, bool include_internal,
    const std::vector<string>& api_def_dirs) {
  OpRegistry::Global()->Export(include_internal, &ops);
  ApiDefMap api_def_map(ops);
  if (!api_def_dirs.empty()) {
    Env* env = Env::Default();
    // Only load files that correspond to "ops".
    for (const auto& op : ops.op()) {
      for (const auto& api_def_dir : api_def_dirs) {
        const std::string api_def_file_pattern =
            io::JoinPath(api_def_dir, "api_def_" + op.name() + ".pbtxt");
        if (env->FileExists(api_def_file_pattern).ok()) {
          auto status = api_def_map.LoadFile(env, api_def_file_pattern);
          if (!status.ok()) return status;
        }
      }
    }
  }
  return api_def_map;
}

string GetPath(StringPiece dot_h_fname) {
  auto pos = dot_h_fname.find("/bin/");
  string result(dot_h_fname);
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

string GetFilename(StringPiece path) {
  size_t slash_pos = path.rfind('/');
  if (slash_pos == path.npos) slash_pos = -1;
  size_t dot_pos = path.rfind('.');
  return string(path.substr(slash_pos + 1, dot_pos - (slash_pos + 1)));
}

string ToGuard(StringPiece path) {
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

string ToTitle(StringPiece name) {
  string title(name);
  for (int i = 0; i < title.size(); ++i) {
    if (title[i] == '_') title[i] = ' ';
  }
  str_util::TitlecaseString(&title, " ");
  return title;
}

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

string PrintString(StringPiece str) {
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

string PrintTensor(const TensorProto& tensor_proto) {
  Tensor t(tensor_proto.dtype());
  CHECK(t.FromProto(tensor_proto));
  const int64_t num_elts = t.NumElements();
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
      return PrintArray(num_elts, t.flat<int64_t>().data());
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

string ToCamelCase(StringPiece str) {
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

string SeparateNamespaces(StringPiece str) {
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

std::pair<StringPiece, bool> AttrTypeName(StringPiece attr_type) {
  static const auto* attr_type_map =
      new std::unordered_map<StringPiece, std::pair<StringPiece, bool>,
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

StringPiece ListElementTypeName(StringPiece attr_type) {
  static const auto* attr_list_type_map =
      new absl::flat_hash_map<StringPiece, StringPiece, StringPieceHasher>{
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
  static const absl::flat_hash_set<StringPiece, StringPieceHasher>*
      // Keywords obtained from http://en.cppreference.com/w/cpp/keyword
      kCPPReserved = new absl::flat_hash_set<StringPiece, StringPieceHasher>{
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
  return kCPPReserved->count(name) > 0;
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
      // TODO(joshl): Word wrap this.
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
        // TODO(joshl): Word wrap this.
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

}  // namespace cc_op
}  // namespace tensorflow
