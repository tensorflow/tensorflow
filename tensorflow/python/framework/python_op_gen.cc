/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/python/framework/python_op_gen.h"

#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

const int kRightMargin = 78;

bool IsPythonReserved(const string& s) {
  static const std::set<string>* const kPythonReserved = new std::set<string>(
      {// Keywords in Python, from:
       //   import keyword
       //   print keyword.kwlist
       "and", "as", "assert", "break", "class", "continue", "def", "del",
       "elif", "else", "except", "exec", "finally", "for", "from", "global",
       "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",
       "raise", "return", "try", "while", "with", "yield",
       // Built-in functions and types in Python, from:
       //   [x for x in dir(__builtins__) if not x[0].islower()]
       "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
       "BufferError", "BytesWarning", "DeprecationWarning", "EOFError",
       "Ellipsis", "EnvironmentError", "Exception", "False",
       "FloatingPointError", "FutureWarning", "GeneratorExit", "IOError",
       "ImportError", "ImportWarning", "IndentationError", "IndexError",
       "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError",
       "NameError", "None", "NotImplemented", "NotImplementedError", "OSError",
       "OverflowError", "PendingDeprecationWarning", "ReferenceError",
       "RuntimeError", "RuntimeWarning", "StandardError", "StopIteration",
       "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit", "TabError",
       "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError",
       "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
       "UnicodeWarning", "UserWarning", "ValueError", "Warning",
       "ZeroDivisionError", "__debug__", "__doc__", "__import__", "__name__",
       "__package__",
       // Imports and symbols used in the generated code:
       "_op_def_lib", "text_format", "op_def_pb2", "op_def_library", "ops"});

  return kPythonReserved->count(s) > 0;
}

// Add a _ to the end of s if necessary to avoid a Python keyword or built-in.
string AvoidPythonReserved(const string& s) {
  if (IsPythonReserved(s)) return strings::StrCat(s, "_");
  return s;
}

// Indent the first line by "initial" spaces and all following lines
// by "rest" spaces.
string Indent(int initial, int rest, StringPiece in) {
  // TODO(josh11b): Also word-wrapping?
  string copy(in.data(), in.size());
  str_util::StripTrailingWhitespace(&copy);
  std::vector<string> v = str_util::Split(copy, '\n');

  string result;
  bool first = true;
  for (const string& line : v) {
    if (first) {
      result = strings::StrCat(Spaces(initial), line, "\n");
      first = false;
    } else {
      if (line.empty()) {
        strings::StrAppend(&result, "\n");
      } else {
        strings::StrAppend(&result, Spaces(rest), line, "\n");
      }
    }
  }
  return result;
}

// Adds append to *dest, with a space if the first line will be <= width,
// or a newline otherwise.
void AppendWithinWidth(string* dest, StringPiece append, int width) {
  auto first_line = append.find('\n');
  if (first_line == string::npos) first_line = append.size();
  if (dest->size() + first_line + 1 /* space */ > static_cast<size_t>(width)) {
    strings::StrAppend(dest, "\n", append);
  } else {
    strings::StrAppend(dest, " ", append);
  }
}

// Like DataTypeString() but uses the Python names for the
// float types.
string PythonDataTypeString(DataType dtype) {
  switch (dtype) {
    case DT_FLOAT:
      return "float32";
    case DT_DOUBLE:
      return "float64";
    default:
      return DataTypeString(dtype);
  }
}

string TypeString(DataType dtype, bool ref) {
  if (ref) {
    return strings::StrCat("mutable `", PythonDataTypeString(dtype), "`");
  } else {
    return strings::StrCat("`", PythonDataTypeString(dtype), "`");
  }
}

string TypeListString(const AttrValue& value) {
  string ret;
  for (int t : value.list().type()) {
    if (!ret.empty()) strings::StrAppend(&ret, ", ");
    DataType dtype = static_cast<DataType>(t);
    if (IsRefType(dtype)) {
      strings::StrAppend(&ret, PythonDataTypeString(RemoveRefType(dtype)),
                         " mutable");
    } else {
      strings::StrAppend(&ret, "`", PythonDataTypeString(dtype), "`");
    }
  }
  return ret;
}

string SingleTensorName(DataType dtype, bool is_ref) {
  const string type_str = TypeString(dtype, is_ref);
  return strings::StrCat("A `Tensor` of type ", type_str, ".");
}

const char kUnknownTensorType[] = {"A `Tensor`."};

string ArgTypeName(const OpDef& op_def, const OpDef::ArgDef& arg,
                   const std::unordered_map<string, string>& inferred_attrs,
                   bool is_output) {
  if (!arg.number_attr().empty()) {
    // N Tensors with the same type
    const string* original_arg =
        gtl::FindOrNull(inferred_attrs, arg.number_attr());
    string prefix;
    if (original_arg == nullptr) {
      prefix = strings::StrCat("A list of `", arg.number_attr(), "`");
    } else if (*original_arg == arg.name()) {
      const OpDef::AttrDef* attr = FindAttr(arg.number_attr(), op_def);
      if (attr->has_minimum() && attr->minimum() > 0) {
        prefix = strings::StrCat("A list of at least ", attr->minimum());
      } else {
        prefix = "A list of";
      }
    } else {
      prefix = strings::StrCat(
          "A list with the same number of `Tensor` objects as `",
          AvoidPythonReserved(*original_arg), "` of");
    }

    if (arg.type() != DT_INVALID) {
      return strings::StrCat(prefix, " `Tensor` objects of type ",
                             TypeString(arg.type(), arg.is_ref()), ".");
    } else {
      original_arg = gtl::FindOrNull(inferred_attrs, arg.type_attr());
      if (arg.is_ref()) {
        strings::StrAppend(&prefix, " mutable");
      }
      if (original_arg == nullptr) {
        return strings::StrCat(prefix, " `Tensor` objects of type ",
                               arg.type_attr(), ".");
      } else if (*original_arg == arg.name()) {
        const OpDef::AttrDef* attr = FindAttr(arg.type_attr(), op_def);
        if (attr->has_allowed_values()) {
          return strings::StrCat(prefix,
                                 " `Tensor` objects of the same type in: ",
                                 TypeListString(attr->allowed_values()), ".");
        } else {
          return strings::StrCat(prefix, " `Tensor` objects of the same type.");
        }
      } else {
        return strings::StrCat(prefix, " `Tensor` objects of the same type as ",
                               AvoidPythonReserved(*original_arg), ".");
      }
    }
  } else if (!arg.type_attr().empty() || !arg.type_list_attr().empty()) {
    const bool is_list = !arg.type_list_attr().empty();
    const string attr_name = is_list ? arg.type_list_attr() : arg.type_attr();
    const OpDef::AttrDef* attr = FindAttr(attr_name, op_def);
    const string mutable_str = arg.is_ref() ? "mutable " : "";
    const string prefix =
        is_list ? strings::StrCat("A list of ", mutable_str, "`Tensor` objects")
                : strings::StrCat("A ", mutable_str, "`Tensor`");
    const string* original_arg = gtl::FindOrNull(inferred_attrs, attr_name);
    if (original_arg == nullptr) {
      return strings::StrCat(prefix, " of type `", attr_name, "`.");
    } else if (*original_arg == arg.name()) {
      if (attr->has_allowed_values()) {
        if (is_list) {
          return strings::StrCat(prefix, " with types from: ",
                                 TypeListString(attr->allowed_values()), ".");
        } else {
          return strings::StrCat(
              prefix, is_output ? ". Has one of the following types: "
                                : ". Must be one of the following types: ",
              TypeListString(attr->allowed_values()), ".");
        }
      } else {
        return strings::StrCat(prefix, ".");
      }
    } else {
      return strings::StrCat(prefix,
                             is_output ? ". Has the same type as `"
                                       : ". Must have the same type as `",
                             AvoidPythonReserved(*original_arg), "`.");
    }
  } else {
    return SingleTensorName(arg.type(), arg.is_ref());
  }
}

static string GetReturns(const OpDef& op_def,
                         const std::vector<string>& output_type_string) {
  string result;
  DCHECK_EQ(op_def.output_arg_size(), output_type_string.size());
  const int num_outs = op_def.output_arg_size();
  strings::Appendf(&result, "\n  Returns:\n");
  if (num_outs == 0) {
    strings::Appendf(&result, "    The created Operation.\n");
  } else {
    if (num_outs == 1) {
      StringPiece description = op_def.output_arg(0).description();
      if (ConsumeEquals(&description)) {  // Skip the generated type info.
        strings::Appendf(&result, "%s", Indent(4, 4, description).c_str());
      } else {
        // Special case of one output, don't use the name of the output unless
        // there is no description.
        string desc = output_type_string.empty() ? kUnknownTensorType
                                                 : output_type_string[0];
        if (desc == kUnknownTensorType) {
          // Special case where we don't understand how the output tensor type
          // depends on the input tensor types, just use the output arg
          // description if we can.
          if (!description.empty()) {
            desc = op_def.output_arg(0).description();
          } else if (!op_def.output_arg(0).name().empty()) {
            desc = strings::StrCat(" The ", op_def.output_arg(0).name(),
                                   " `Tensor`.");
          }
        } else if (!description.empty()) {
          AppendWithinWidth(&desc, description, kRightMargin - 4 /* indent */);
        }
        strings::Appendf(&result, "%s", Indent(4, 4, desc).c_str());
      }
    } else {
      std::vector<string> out_names(num_outs);
      for (int i = 0; i < num_outs; ++i) {
        if (!op_def.output_arg(i).name().empty()) {
          out_names[i] = op_def.output_arg(i).name();
        } else {
          out_names[i] = strings::StrCat("output", i);
        }
      }
      strings::Appendf(&result, "    A tuple of `Tensor` objects (%s).\n",
                       str_util::Join(out_names, ", ").c_str());
      for (int i = 0; i < num_outs; ++i) {
        string desc = strings::StrCat(out_names[i], ": ");
        StringPiece description = op_def.output_arg(i).description();
        if (ConsumeEquals(&description)) {  // Skip the generated type info.
          strings::StrAppend(&desc, description);
        } else {
          const string type = static_cast<size_t>(i) < output_type_string.size()
                                  ? output_type_string[i]
                                  : kUnknownTensorType;
          if (!description.empty()) {
            if (type == kUnknownTensorType) {
              // Special case where we don't understand how the output tensor
              // type depends on the input tensor types, so we just use the
              // output arg description.
              strings::StrAppend(&desc, description);
            } else {
              strings::StrAppend(&desc, type, " ", description);
            }
          } else {
            strings::StrAppend(&desc, type);
          }
        }
        strings::Appendf(&result, "%s", Indent(4, 6, desc).c_str());
      }
    }
  }
  return result;
}

string StringToPython(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}

string DataTypeToPython(DataType dtype) {
  return strings::StrCat("tf.", PythonDataTypeString(dtype));
}

string ShapeToPython(const TensorShapeProto& shape) {
  string python = "[";
  for (const auto& dim : shape.dim()) {
    if (python.size() > 1) strings::StrAppend(&python, ", ");
    if (!dim.name().empty()) {
      strings::StrAppend(&python, "(", StringToPython(dim.name()), ", ",
                         dim.size(), ")");
    } else {
      strings::StrAppend(&python, dim.size());
    }
  }
  strings::StrAppend(&python, "]");
  return python;
}

string AttrListToPython(const AttrValue& value) {
  string ret;
  if (value.list().s_size() > 0) {
    for (int i = 0; i < value.list().s_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, StringToPython(value.list().s(i)));
    }
  } else if (value.list().i_size() > 0) {
    for (int i = 0; i < value.list().i_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().i(i));
    }
  } else if (value.list().f_size() > 0) {
    for (int i = 0; i < value.list().f_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().f(i));
    }
  } else if (value.list().b_size() > 0) {
    for (int i = 0; i < value.list().b_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().b(i) ? "True" : "False");
    }
  } else if (value.list().type_size() > 0) {
    for (int i = 0; i < value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, DataTypeToPython(value.list().type(i)));
    }
  } else if (value.list().shape_size() > 0) {
    for (int i = 0; i < value.list().shape_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, ShapeToPython(value.list().shape(i)));
    }
  }
  return ret;
}

string AttrValueToPython(const string& type, const AttrValue& value) {
  if (type == "string") {
    return StringToPython(value.s());
  } else if (type == "int") {
    return strings::StrCat(value.i());
  } else if (type == "float") {
    return strings::StrCat(value.f());
  } else if (type == "bool") {
    return value.b() ? "True" : "False";
  } else if (type == "type") {
    return DataTypeToPython(value.type());
  } else if (type == "shape") {
    return ShapeToPython(value.shape());
  } else {
    return strings::StrCat("[", AttrListToPython(value), "]");
  }
}

static string GetPythonOp(const OpDef& op_def, bool is_hidden, string op_name) {
  string result;
  // Map from attr name to the first input arg it is inferred from.
  std::unordered_map<string, string> inferred_attrs;
  // This has all the input args followed by those attrs that don't have
  // defaults.
  std::vector<string> args_no_default;
  // The parameters with defaults (these have to be listed after those without).
  // No input args are included, just attrs and the graph ("g") parameter.
  std::vector<string> args_with_defaults;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    args_no_default.push_back(arg.name());
    if (!arg.type_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.type_attr(), arg.name());
    } else if (!arg.type_list_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.type_list_attr(),
                              arg.name());
    }
    if (!arg.number_attr().empty()) {
      gtl::InsertIfNotPresent(&inferred_attrs, arg.number_attr(), arg.name());
    }
  }
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // Do not add inferred attrs to the Python function signature.
    if (inferred_attrs.find(attr.name()) == inferred_attrs.end()) {
      if (attr.has_default_value()) {
        args_with_defaults.push_back(attr.name());
      } else {
        args_no_default.push_back(attr.name());
      }
    }
  }

  // Save the list of attr parameters (attrs that won't be inferred),
  // those with defaults go at the end.
  std::vector<string> attrs;
  // Get the attrs in the order we want by taking the attrs without defaults
  // from the end of args_no_default, and adding args_no_default (before
  // "g" gets added to args_no_default, so it only has attrs).
  attrs.reserve(args_no_default.size() - op_def.input_arg_size() +
                args_with_defaults.size());
  attrs.insert(attrs.end(), args_no_default.begin() + op_def.input_arg_size(),
               args_no_default.end());
  attrs.insert(attrs.end(), args_with_defaults.begin(),
               args_with_defaults.end());

  std::vector<string> param_names;
  param_names.reserve(args_no_default.size() + args_with_defaults.size());
  string parameters;
  for (const string& name : args_no_default) {
    if (!parameters.empty()) strings::StrAppend(&parameters, ", ");
    const string param = AvoidPythonReserved(name);
    strings::StrAppend(&parameters, param);
    param_names.push_back(param);
  }
  for (const string& name : args_with_defaults) {
    if (!parameters.empty()) strings::StrAppend(&parameters, ", ");
    const string param = AvoidPythonReserved(name);
    strings::StrAppend(&parameters, param, "=None");
    param_names.push_back(param);
  }
  const bool has_args = args_no_default.size() + args_with_defaults.size() > 0;

  const string lower_op_name = strings::StrCat(is_hidden ? "_" : "", op_name);

  // Prepare the list of output names
  const int num_outs = op_def.output_arg_size();
  std::vector<string> out_names(num_outs);
  for (int i = 0; i < num_outs; ++i) {
    if (!op_def.output_arg(i).name().empty()) {
      out_names[i] = op_def.output_arg(i).name();
    } else {
      out_names[i] = strings::StrCat("output", i);
    }
  }
  string out_names_list =
      strings::StrCat("[\"", str_util::Join(out_names, "\", \""), "\"]");

  // Provide the output names as a Python list
  string lower_op_name_outputs =
      strings::StrCat("_", lower_op_name, "_outputs");
  const string outputs_prefix = strings::StrCat(lower_op_name_outputs, " = ");
  strings::Appendf(
      &result, "%s\n",
      WordWrap(outputs_prefix, out_names_list, kRightMargin).c_str());
  strings::Appendf(&result, "\n\n");

  // Prepare a NamedTuple type to hold the outputs, if there are multiple
  if (num_outs > 1) {
    const string tuple_type_prefix =
        strings::StrCat("_", op_def.name(), "Output = collections.namedtuple(");
    const string tuple_type_suffix = strings::StrCat(
        "\"", op_def.name(), "\", ", lower_op_name_outputs, ")");
    strings::Appendf(
        &result, "%s\n",
        WordWrap(tuple_type_prefix, tuple_type_suffix, kRightMargin).c_str());
    strings::Appendf(&result, "\n\n");
  }

  // Print: def Function(parameters):
  const string def_prefix = strings::StrCat("def ", lower_op_name, "(");
  const string def_suffix =
      strings::StrCat(parameters, has_args ? ", " : "", "name=None):");

  strings::Appendf(&result, "%s\n",
                   WordWrap(def_prefix, def_suffix, kRightMargin).c_str());

  // Format the Op's descriptions so that it can be a Python docstring.
  string comment;
  if (op_def.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(op_def.summary(), "\n");
    if (!op_def.description().empty()) {
      strings::StrAppend(&comment, "\n", Indent(2, 2, op_def.description()));
    }
  }

  strings::Appendf(&result, "  r\"\"\"%s\n  Args:\n", comment.c_str());

  // Inputs
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    StringPiece description = op_def.input_arg(i).description();
    string desc;
    if (ConsumeEquals(&description)) {  // Skip the generated type info.
      desc = strings::StrCat(param_names[i], ": ");
    } else {
      desc = strings::StrCat(param_names[i], ": ",
                             ArgTypeName(op_def, arg, inferred_attrs, false));
    }
    if (!description.empty()) {
      AppendWithinWidth(&desc, description, kRightMargin - 4 /* indent */);
    }
    strings::Appendf(&result, "%s", Indent(4, 6, desc).c_str());
  }

  // Attrs
  for (const string& name : attrs) {
    const auto& attr = *FindAttr(name, op_def);
    string desc = strings::StrCat(AvoidPythonReserved(name), ": ");

    static const char* const kAttrTypeName[][2] = {
        {"string", "`string`"},
        {"list(string)", "list of `strings`"},
        {"int", "`int`"},
        {"list(int)", "list of `ints`"},
        {"float", "`float`"},
        {"list(float)", "list of `floats`"},
        {"bool", "`bool`"},
        {"list(bool)", "list of `bools`"},
        {"type", "`tf.DType`"},
        {"list(type)", "list of `tf.DTypes`"},
        {"shape", "`tf.TensorShape` or list of `ints`"},
        {"list(shape)",
         "list of shapes (each a `tf.TensorShape` or list of `ints`)"},
    };
    for (size_t i = 0; i < TF_ARRAYSIZE(kAttrTypeName); ++i) {
      if (attr.type() == kAttrTypeName[i][0]) {
        string s;
        if (attr.has_default_value()) {
          s = strings::StrCat("optional ", kAttrTypeName[i][1]);
        } else {
          s = kAttrTypeName[i][1];
        }
        if (s[0] == 'o' || (s[0] == '`' && (s[1] == 'i' || s[1] == 'o'))) {
          strings::StrAppend(&desc, "An ", s);
        } else {
          strings::StrAppend(&desc, "A ", s);
        }
        break;
      }
    }

    if (attr.has_allowed_values()) {
      strings::StrAppend(&desc, " from: `",
                         AttrListToPython(attr.allowed_values()), "`");
    }

    if (attr.has_minimum()) {
      if (attr.type() == "int") {
        strings::StrAppend(&desc, " that is `>= ", attr.minimum(), "`");
      } else if (attr.minimum() > 0) {
        strings::StrAppend(&desc, " that has length `>= ", attr.minimum(), "`");
      }
    }

    strings::StrAppend(&desc, ".");

    if (attr.has_default_value()) {
      strings::StrAppend(&desc, " Defaults to `",
                         AttrValueToPython(attr.type(), attr.default_value()),
                         "`.");
    }

    if (!attr.description().empty()) {
      AppendWithinWidth(&desc, attr.description(),
                        kRightMargin - 4 /* indent */);
    }
    strings::Appendf(&result, "%s", Indent(4, 6, desc).c_str());
  }

  strings::Appendf(&result, "    name: A name for the operation (optional).\n");

  std::vector<string> output_type_string;
  output_type_string.reserve(op_def.output_arg_size());
  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    output_type_string.push_back(
        ArgTypeName(op_def, op_def.output_arg(i), inferred_attrs, true));
  }
  strings::StrAppend(&result, GetReturns(op_def, output_type_string));

  string return_prefix = strings::StrCat("  result = _op_def_lib.apply_op(");
  string return_args = strings::StrCat("\"", op_def.name(), "\", ");
  for (size_t i = 0; i < param_names.size(); ++i) {
    strings::StrAppend(&return_args, param_names[i], "=", param_names[i], ", ");
  }
  strings::StrAppend(&return_args, "name=name)");

  strings::Appendf(&result, "  \"\"\"\n%s\n",
                   // Wrap the arguments, and indent to the (.
                   WordWrap(return_prefix, return_args, kRightMargin).c_str());

  if (num_outs <= 1) {
    strings::Appendf(&result, "  return result\n");
  } else {
    string return_tuple =
        strings::StrCat("  return _", op_def.name(), "Output._make(result)\n");
    strings::Appendf(&result, "%s", return_tuple.c_str());
  }

  strings::Appendf(&result, "\n\n");
  return result;
}

void GenerateLowerCaseOpName(const string& str, string* result) {
  char joiner = '_';
  int last_index = str.size() - 1;
  for (int i = 0; i <= last_index; ++i) {
    char c = str[i];
    // Emit a joiner only if a previous-lower-to-now-upper or a
    // now-upper-to-next-lower transition happens.
    if (isupper(c) && (i > 0)) {
      if (islower(str[i - 1]) || ((i < last_index) && islower(str[i + 1]))) {
        result->push_back(joiner);
      }
    }
    result->push_back(tolower(c));
  }
}

}  // namespace

string GetPythonOps(const OpList& ops, const string& hidden_ops,
                    bool require_shapes) {
  string result;
  // Header
  // TODO(josh11b): Mention the library for which wrappers are being generated.
  strings::Appendf(&result, R"("""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


)");

  std::vector<string> hidden_vec = str_util::Split(hidden_ops, ',');

  // We'll make a copy of ops that filters out descriptions.
  OpList cleaned_ops;
  auto out = cleaned_ops.mutable_op();
  out->Reserve(ops.op_size());
  for (const auto& op_def : ops.op()) {
    bool is_hidden = false;
    for (const string& hidden : hidden_vec) {
      if (op_def.name() == hidden) {
        is_hidden = true;
        break;
      }
    }

    // PrintPythonOp(op_def, is_hidden, op_def.name());
    string lower_case_name;
    GenerateLowerCaseOpName(op_def.name(), &lower_case_name);

    // When users create custom python wrappers, they may link in the
    // default op registry by accident, and because they can't
    // enumerate all 'hidden' symbols, this guard is to prevent
    // instantiating a python reserved word in their wrapper.
    if (!is_hidden && IsPythonReserved(lower_case_name)) {
      continue;
    }

    strings::StrAppend(&result,
                       GetPythonOp(op_def, is_hidden, lower_case_name));

    if (!require_shapes) {
      strings::Appendf(&result, "ops.RegisterShape(\"%s\")(None)\n",
                       op_def.name().c_str());
    }

    auto added = out->Add();
    *added = op_def;
    RemoveNonDeprecationDescriptionsFromOpDef(added);
  }

  strings::Appendf(&result, R"(def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """%s"""


_op_def_lib = _InitOpDefLibrary()
)",
                   cleaned_ops.DebugString().c_str());
  return result;
}

void PrintPythonOps(const OpList& ops, const string& hidden_ops,
                    bool require_shapes) {
  printf("%s", GetPythonOps(ops, hidden_ops, require_shapes).c_str());
}

string GetAllPythonOps(const char* hidden, bool require_shapes) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);
  return GetPythonOps(ops, hidden, require_shapes);
}

string GetPythonWrappers(const char* op_wrapper_buf, size_t op_wrapper_len) {
  string op_list_str(op_wrapper_buf, op_wrapper_len);
  OpList ops;
  ops.ParseFromString(op_list_str);
  return GetPythonOps(ops, "", false);
}

}  // namespace tensorflow
