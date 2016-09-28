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

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/cc/framework/cc_op_gen.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

const int kRightMargin = 79;

// Converts:
//   bazel-out/.../genfiles/(external/YYY/)?XX
// to: XX.
string GetPath(const std::string& dot_h_fname) {
  auto pos = dot_h_fname.find("/genfiles/");
  string result = dot_h_fname;
  if (pos != string::npos) {
    // - 1 account for the terminating null character (\0) in "/genfiles/".
    result = dot_h_fname.substr(pos + sizeof("/genfiles/") - 1);
  }
  if (result.size() > sizeof("external/") &&
      result.compare(0, sizeof("external/") - 1, "external/") == 0) {
    result = result.substr(sizeof("external/") - 1);
    pos = result.find("/");
    if (pos != string::npos) {
      result = result.substr(pos + 1);
    }
  }
  return result;
}

// Converts:
//   cc/ops/gen_foo_ops.h
// to:
//   CC_OPS_GEN_FOO_OPS_H_
string ToGuard(const std::string& path) {
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

// Change:     Into:
//   ABC         // ABC
//               //
//   DEF         // DEF
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
      strings::StrAppend(&ret, indent, "//\n");
    } else {
      strings::StrAppend(&ret, indent, "// ",
                         text.substr(0, last_non_space + 1), "\n");
    }
    text.remove_prefix(newline + 1);
  }
  return ret;
}

string PrintString(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}

string PrintTensorShape(const TensorShape& shape) {
  string ret = "{";
  for (int d = 0; d < shape.dims(); ++d) {
    if (d > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, shape.dim_size(d));
  }
  strings::StrAppend(&ret, "}");
  return ret;
}

template <typename T>
string PrintArray(int64 num_elts, const T* array) {
  string ret;
  for (int64 i = 0; i < num_elts; ++i) {
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
      for (int64 i = 0; i < num_elts; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        strings::StrAppend(&ret, str_util::CEscape(t.flat<string>()(i)));
      }
      return ret;
    }
    default: {
      LOG(FATAL) << "Not handling type " << EnumName_DataType(t.dtype());
      return string();
    }
  }
}

string PrintAttrValue(string op, const AttrValue& attr_value) {
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
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PrintTensorShape(TensorShape(attr_value.shape()));
    case AttrValue::kTensor:
      return strings::StrCat(
          "Input::Initializer(", "{", PrintTensor(attr_value.tensor()), "}, ",
          PrintTensorShape(TensorShape(attr_value.tensor().tensor_shape())),
          ").AsTensorProto()");
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
          strings::StrAppend(&ret,
                             EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(
              &ret, PrintTensorShape(TensorShape(attr_value.list().shape(i))));
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

string ToCamelCase(const string& str) {
  string result;
  const char joiner = '_';
  size_t i = 0;
  bool cap = true;
  while (i < str.size()) {
    const char c = str[i++];
    if (c == joiner) {
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

// Returns a <string, bool> pair. The string is the C++ type name to be used for
// attr_type when defining an object of that type. The bool is a flag to
// indicate whether to treat the type as const when accepting the C++ type as an
// argument to a function.
std::pair<const char*, bool> AttrTypeName(StringPiece attr_type) {
  static const std::unordered_map<StringPiece, std::pair<const char*, bool>,
                                  StringPiece::Hasher>
      attr_type_map{
          {"string", {"StringPiece", false}},
          {"list(string)", {"gtl::ArraySlice<string>", true}},
          {"int", {"int64", false}},
          {"list(int)", {"gtl::ArraySlice<int>", true}},
          {"float", {"float", false}},
          {"list(float)", {"gtl::ArraySlice<float>", true}},
          {"bool", {"bool", false}},
          {"list(bool)", {"gtl::ArraySlice<bool>", true}},
          {"type", {"DataType", false}},
          {"list(type)", {"DataTypeSlice", true}},
          {"shape", {"TensorShape", false}},
          {"list(shape)", {"gtl::ArraySlice<TensorShape>", true}},
          {"tensor", {"TensorProto", true}},
          {"list(tensor)", {"gtl::ArraySlice<TensorProto>", true}},
          {"func", {"NameAttrList", true}},
      };

  auto entry = attr_type_map.find(attr_type);
  if (entry == attr_type_map.end()) {
    LOG(FATAL) << "Unsupported Attr type: " << attr_type;
    return {"", false};
  }
  return entry->second;
}

bool IsCPPKeyword(StringPiece name) {
  static const std::unordered_set<StringPiece, StringPiece::Hasher>
      // Keywords obtained from http://en.cppreference.com/w/cpp/keyword
      kCPPReserved{
          "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel",
          "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool",
          "break", "case", "catch", "char", "char16_t", "char32_t", "class",
          "compl", "concept", "const", "const_cast", "constexpr", "continue",
          "decltype", "default", "delete", "do", "double", "dynamic_cast",
          "else", "enum", "explicit", "export", "extern", "false", "final",
          "float", "for", "friend", "goto", "if", "import", "inline", "int",
          "long", "module", "mutable", "namespace", "new", "noexcept", "not",
          "not_eq", "nullptr", "operator", "or", "or_eq", "override", "private",
          "protected", "public", "register", "reinterpret_cast", "requires",
          "return", "short", "signed", "sizeof", "static", "static_assert",
          "static_cast", "struct", "switch", "synchronized", "template", "this",
          "thread_local", "throw", "true", "try", "typedef", "typeid",
          "typename", "union", "unsigned", "using", "virtual", "void",
          "volatile", "wchar_t", "while", "xor", "xor_eq",

          // The following are not C++ keywords, but names of local variables
          // and parameters used in the op constructor. Treating them as
          // keywords, so that other parameter names don't conflict with these.
          "builder", "node", "ret", "scope", "unique_name",
      };
  return kCPPReserved.count(name) > 0;
}

string AvoidCPPKeywords(StringPiece name) {
  if (IsCPPKeyword(name)) {
    return strings::StrCat(name, "_");
  }
  return name.ToString();
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
    const OpDef& op_def,
    const std::unordered_map<string, string>& inferred_input_attrs) {
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    if ((inferred_input_attrs.find(attr.name()) ==
         inferred_input_attrs.end()) &&
        attr.has_default_value()) {
      return true;
    }
  }
  return false;
}

struct OpInfo {
  explicit OpInfo(const OpDef& op_def);
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

  const OpDef& op_def;
  std::unordered_map<string, string> inferred_input_attrs;
};

OpInfo::OpInfo(const OpDef& op_def) : op_def(op_def) {
  op_name = op_def.name();
  InferOpAttributes(op_def, &inferred_input_attrs);
  has_optional_attrs = HasOptionalAttrs(op_def, inferred_input_attrs);
  arg_types.push_back("const ::tensorflow::Scope&");
  arg_names.push_back("scope");

  if (op_def.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(op_def.summary(), "\n");
    if (op_def.has_deprecation()) {
      strings::StrAppend(&comment, "\nDEPRECATED at GraphDef version ",
                         op_def.deprecation().version(), ":\n",
                         op_def.deprecation().explanation(), ".\n");
    }
    if (!op_def.description().empty()) {
      strings::StrAppend(&comment, "\n", op_def.description(), "\n");
    }
  }
  strings::StrAppend(&comment, "\nArguments:\n* scope: A Scope object\n");

  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    arg_types.push_back(strings::StrCat(
        "::tensorflow::ops::", ArgIsList(arg) ? "InputList" : "Input"));
    arg_names.push_back(AvoidCPPKeywords(arg.name()));

    // TODO(keveman): Include input type information.
    StringPiece description = arg.description();
    if (!description.empty()) {
      ConsumeEquals(&description);
      strings::StrAppend(&comment, "* ", AvoidCPPKeywords(arg.name()), ": ",
                         arg.description(), "\n");
    }
  }
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // If the attr is going to be inferred or is optional, don't add it as a
    // required argument.
    if ((inferred_input_attrs.find(attr.name()) !=
         inferred_input_attrs.end()) ||
        attr.has_default_value()) {
      continue;
    }
    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;

    arg_types.push_back(strings::StrCat(use_const ? "const " : "",
                                        attr_type_name, use_const ? "&" : ""));
    arg_names.push_back(AvoidCPPKeywords(attr.name()));
    if (!attr.description().empty()) {
      strings::StrAppend(&comment, "* ", AvoidCPPKeywords(attr.name()), ":\n");
      // TODO(keveman): Word wrap and indent this, to handle multi-line
      // descriptions.
      strings::StrAppend(&comment, "    ", attr.description(), "\n");
    }
  }
  comment = MakeComment(comment, "");

  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    const auto& arg = op_def.output_arg(i);
    bool is_list = ArgIsList(arg);
    output_types.push_back(strings::StrCat("::tensorflow::ops::",
                                           is_list ? "OutputList" : "Output"));
    output_names.push_back(AvoidCPPKeywords(arg.name()));
    is_list_output.push_back(is_list);
  }
}

string OpInfo::GetOpAttrStruct() const {
  string struct_fields;
  string setters;
  string attrs_comment = strings::StrCat("Optional attribute setters for ",
                                         op_def.name(), " :\n\n");

  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // If attr will be inferred or it doesn't have a default value, don't
    // add it to the struct.
    if ((inferred_input_attrs.find(attr.name()) !=
         inferred_input_attrs.end()) ||
        !attr.has_default_value()) {
      continue;
    }
    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;
    const string camel_case_name = ToCamelCase(attr.name());
    const string suffix =
        (camel_case_name == op_name || camel_case_name == "Attrs") ? "_" : "";
    const string attr_func_def =
        strings::StrCat(camel_case_name, suffix, "(", use_const ? "const " : "",
                        attr_type_name, use_const ? "&" : "");

    strings::StrAppend(&attrs_comment, attr_func_def, "): Defaults to ",
                       SummarizeAttrValue(attr.default_value()), "\n");
    if (!attr.description().empty()) {
      // TODO(keveman): Word wrap and indent this to handle multi-line
      // description.
      strings::StrAppend(&attrs_comment, "    ", attr.description(), "\n");
    }
    strings::StrAppend(&setters, "    Attrs ", attr_func_def, " x) {\n");
    strings::StrAppend(&setters, "      Attrs ret = *this;\n");
    strings::StrAppend(&setters, "      ret.", attr.name(), "_ = x;\n");
    strings::StrAppend(&setters, "      return ret;\n    }\n\n");

    strings::StrAppend(
        &struct_fields, "    ", attr_type_name, " ", attr.name(), "_ = ",
        PrintAttrValue(op_def.name(), attr.default_value()), ";\n");
  }

  if (struct_fields.empty()) {
    return "";
  }

  string struct_decl = MakeComment(attrs_comment, "  ");
  strings::StrAppend(&struct_decl, "  struct Attrs {\n");
  strings::StrAppend(&struct_decl, setters, struct_fields);
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
                       "  operator ::tensorflow::ops::Operation() const { "
                       "return operation; }\n");
  } else if (output_types.size() == 1) {
    if (is_list_output[0]) {
      // Write the subscript operator, allowing out[i] for the list-typed
      // output.
      strings::StrAppend(&class_decl,
                         "  ::tensorflow::ops::Output operator[](size_t index) "
                         "const { return ",
                         output_names[0], "[index]; }\n\n");

    } else {
      // Write type cast functions, allowing casting this class to Input and
      // Output.
      strings::StrAppend(
          &class_decl, "  operator ::tensorflow::ops::Output() const { return ",
          output_names[0], "; }\n");
      strings::StrAppend(
          &class_decl, "  operator ::tensorflow::ops::Input() const { return ",
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
    for (int i = 0; i < op_def.attr_size(); ++i) {
      const auto& attr(op_def.attr(i));
      if ((inferred_input_attrs.find(attr.name()) !=
           inferred_input_attrs.end()) ||
          !attr.has_default_value()) {
        continue;
      }
      const auto entry = AttrTypeName(attr.type());
      const auto attr_type_name = entry.first;
      const bool use_const = entry.second;
      const string camel_case_name = ToCamelCase(attr.name());
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

  strings::StrAppend(&class_decl, "\n");

  if (output_types.empty()) {
    strings::StrAppend(&class_decl, "  Operation operation;\n");
  }
  for (int i = 0; i < output_types.size(); ++i) {
    strings::StrAppend(&class_decl, "  ", output_types[i], " ", output_names[i],
                       ";\n");
  }

  strings::StrAppend(&class_decl, "};\n\n");
  TF_CHECK_OK(h->Append(class_decl));
}

void OpInfo::GetOutput(string* out) const {
  const string scope_str = arg_names[0];
  string return_on_error =
      strings::StrCat("if (!", scope_str, ".ok()) return;");

  // No outputs.
  if (op_def.output_arg_size() == 0) {
    strings::StrAppend(out, "  this->operation = Operation(ret);\n  return;\n");
    return;
  }
  if (op_def.output_arg_size() == 1) {
    // One output, no need for NameRangeMap
    if (is_list_output[0]) {
      strings::StrAppend(out,
                         "  for (int64 i = 0; i < ret->num_outputs(); ++i)\n");
      strings::StrAppend(out, "    this->", output_names[0],
                         ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", output_names[0],
                         " = Output(ret, 0);\n");
    }
    return;
  }
  strings::StrAppend(out, "  ::tensorflow::NameRangeMap _outputs_range;\n");
  strings::StrAppend(
      out,
      "  ::tensorflow::Status _status_ = "
      "::tensorflow::NameRangesForNode(ret->def(), ret->op_def(), "
      "nullptr, &_outputs_range);\n");
  strings::StrAppend(out, "  if (!_status_.ok()) {\n", "    ", scope_str,
                     ".UpdateStatus(_status_);\n", "    return;\n");
  strings::StrAppend(out, "  }\n\n");

  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    const string arg_range = strings::StrCat(
        "_outputs_range[\"", op_def.output_arg(i).name(), "\"]");
    if (is_list_output[i]) {
      strings::StrAppend(out, "  for (int64 i = ", arg_range, ".first; i < ",
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

  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    strings::StrAppend(&body, "  auto _", arg.name(), " = ::tensorflow::ops::",
                       ArgIsList(arg) ? "AsNodeOutList" : "AsNodeOut", "(",
                       scope_str, ", ", AvoidCPPKeywords(arg.name()), ");\n");
    strings::StrAppend(&body, "  ", return_on_error, "\n");
  }

  strings::StrAppend(&body, "  ::tensorflow::Node* ret;\n");
  strings::StrAppend(&body, "  const auto  unique_name = ", scope_str,
                     ".GetUniqueNameForOp(\"", op_def.name(), "\");\n");
  strings::StrAppend(
      &body, "  auto builder = ::tensorflow::NodeBuilder(unique_name, \"",
      op_def.name(), "\")\n");
  const string spaces = "                     ";
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    strings::StrAppend(&body, spaces, ".Input(_", arg.name(), ")\n");
  }
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    if (inferred_input_attrs.find(attr.name()) != inferred_input_attrs.end()) {
      continue;
    }
    const string attr_name = attr.has_default_value()
                                 ? strings::StrCat("attrs.", attr.name(), "_")
                                 : AvoidCPPKeywords(attr.name());
    strings::StrAppend(&body, spaces, ".Attr(\"", attr.name(), "\", ",
                       attr_name, ")\n");
  }
  strings::StrAppend(&body, "  ;\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateBuilder(&builder);\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(builder.Finalize(",
                     scope_str, ".graph(), &ret));\n");

  // TODO(b/28152992): Enable this code-path once we have converted
  // all python shape functions to call their C++ versions.

  // strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(", scope_str,
  //                    ".refiner()->AddNode(ret));\n");

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

void WriteCCOp(const OpDef& op_def, WritableFile* h, WritableFile* cc) {
  OpInfo op_info(op_def);

  op_info.WriteClassDecl(h);
  op_info.WriteClassDef(cc);
}

}  // namespace

void WriteCCOps(const OpList& ops, const std::string& dot_h_fname,
                const std::string& dot_cc_fname) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> h = nullptr;
  std::unique_ptr<WritableFile> cc = nullptr;
  TF_CHECK_OK(env->NewWritableFile(dot_h_fname, &h));
  TF_CHECK_OK(env->NewWritableFile(dot_cc_fname, &cc));

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
  const string namespace_begin = R"namespace(
namespace tensorflow {
namespace ops {

)namespace";

  const string footer = R"footer(}  // namespace ops
}  // namespace tensorflow
)footer";

  const string op_header = GetPath(dot_h_fname);
  const string op_header_guard = ToGuard(op_header);
  const string cc_header = strings::StrCat(
      R"include(// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
)include",
      "#include \"", op_header, "\"\n", namespace_begin);

  TF_CHECK_OK(h->Append(
      strings::StrCat("// This file is MACHINE GENERATED! Do not edit.\n\n"
                      "#ifndef ",
                      op_header_guard,
                      "\n"
                      "#define ",
                      op_header_guard, "\n\n")));
  TF_CHECK_OK(h->Append(header));
  TF_CHECK_OK(h->Append(namespace_begin));
  TF_CHECK_OK(cc->Append(cc_header));

  for (const auto& op_def : ops.op()) {
    if (op_def.name() == "Const") {
      // We use a hand-written wrapper for "Const", since the
      // generated code depends on it.
      continue;
    }
    WriteCCOp(op_def, h.get(), cc.get());
  }

  TF_CHECK_OK(h->Append(footer));
  TF_CHECK_OK(
      h->Append(strings::StrCat("\n#endif  ", "// ", op_header_guard, "\n")));
  TF_CHECK_OK(cc->Append(footer));

  TF_CHECK_OK(cc->Close());
  TF_CHECK_OK(h->Close());
}

}  // namespace tensorflow
