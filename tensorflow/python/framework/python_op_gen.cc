/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/framework/python_op_gen_annotator.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace {

// Names specified in tf_export decorators are exported to
// TensorFlow 2.0 by default.
const int kLatestAPIExportVersion = 2;

const int kRightMargin = 78;

constexpr char kEagerFallbackSuffix[] = "_eager_fallback";

// Maps C++ dtype enum values to Python annotation types
const std::unordered_map<string, string> dtype_type{
    {"_dtypes.float16", "_atypes.Float16"},
    {"_dtypes.half", "_atypes.Half"},
    {"_dtypes.float32", "_atypes.Float32"},
    {"_dtypes.float64", "_atypes.Float64"},
    {"_dtypes.bfloat16", "_atypes.BFloat16"},
    {"_dtypes.complex64", "_atypes.Complex64"},
    {"_dtypes.complex128", "_atypes.Complex128"},
    {"_dtypes.int8", "_atypes.Int8"},
    {"_dtypes.uint8", "_atypes.UInt8"},
    {"_dtypes.uint16", "_atypes.UInt16"},
    {"_dtypes.uint32", "_atypes.UInt32"},
    {"_dtypes.uint64", "_atypes.UInt64"},
    {"_dtypes.int16", "_atypes.Int16"},
    {"_dtypes.int32", "_atypes.Int32"},
    {"_dtypes.int64", "_atypes.Int64"},
    {"_dtypes.bool", "_atypes.Bool"},
    {"_dtypes.string", "_atypes.String"},
    {"_dtypes.qint8", "_atypes.QInt8"},
    {"_dtypes.quint8", "_atypes.QUInt8"},
    {"_dtypes.qint16", "_atypes.QInt16"},
    {"_dtypes.quint16", "_atypes.QUInt16"},
    {"_dtypes.qint32", "_atypes.QInt32"},
    {"_dtypes.resource", "_atypes.Resource"},
    {"_dtypes.variant", "_atypes.Variant"},
    {"_dtypes.float8_e4m3fn", "_atypes.Float8e4m3fn"},
    {"_dtypes.float8_e5m2", "_atypes.Float8e5m2"},
    {"_dtypes.float8_e4m3fnuz", "_atypes.Float8e4m3fnuz"},
    {"_dtypes.float8_e4m3b11fnuz", "_atypes.Float8e4m3b11fnuz"},
    {"_dtypes.float8_e5m2fnuz", "_atypes.Float8e5m2fnuz"},
    {"_dtypes.int4", "_atypes.Int4"},
    {"_dtypes.uint4", "_atypes.UInt4"},
    {"_dtypes.int2", "_atypes.Int2"},
    {"_dtypes.uint2", "_atypes.UInt2"},
};

string AttrVarName(const string& attr_name,
                   std::unordered_map<string, string>* attr_expressions) {
  const string var = strings::StrCat("_attr_", attr_name);
  if (attr_expressions != nullptr) (*attr_expressions)[attr_name] = var;
  return var;
}

void AddInferredAttr(const string& indentation, const string& attr_name,
                     const string& value_expression, string* result,
                     std::unordered_map<string, string>* attr_expressions) {
  strings::StrAppend(result, indentation,
                     AttrVarName(attr_name, attr_expressions), " = ",
                     value_expression, "\n");
}

string VectorToTuple(const std::vector<string>& l) {
  if (l.size() == 1) return strings::StrCat("(", l.front(), ",)");
  string ret = "(";
  for (int i = 0, end = l.size(); i < end; ++i) {
    if (i > 0) {
      strings::StrAppend(&ret, ", ");
    }
    strings::StrAppend(&ret, l[i]);
  }
  strings::StrAppend(&ret, ")");
  return ret;
}

void Unflatten(const string& prefix, const std::vector<string>& output_sizes,
               const string& var, string* result) {
  for (int i = 0, end = output_sizes.size(); i < end; ++i) {
    if (!output_sizes[i].empty()) {
      strings::StrAppend(result, prefix, var, " = ");
      if (i > 0) strings::StrAppend(result, var, "[:", i, "] + ");
      if (i + 1 < end) {
        // Special case i == 0 to avoid "0 +" in the generated code.
        if (i == 0) {
          strings::StrAppend(result, "[", var, "[:", output_sizes[i], "]] + ",
                             var, "[", output_sizes[i], ":]");
        } else {
          strings::StrAppend(result, "[", var, "[", i, ":", i, " + ",
                             output_sizes[i], "]] + ", var, "[", i, " + ",
                             output_sizes[i], ":]");
        }
      } else {
        strings::StrAppend(result, "[", var, "[", i, ":]]");
      }
      strings::StrAppend(result, "\n");
    }
  }
}

string TensorPBString(const TensorProto& pb) {
  // Explicitly not using ShortDebugString, because ShortDebugString should
  // not be used as a format for transporting information (it's e.g. subject
  // to redaction of sensitive information). There is a PrintShortTextProto
  // helper, but it's not feasible to depend on that library).

  std::string message_short_text;

  ::tensorflow::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.SetExpandAny(true);

  printer.PrintToString(pb, &message_short_text);

  // Note: This gets used in the argument list, and so must survive naive
  // word wrapping.
  return strings::StrCat("\"\"\"", message_short_text, "\"\"\"");
}

// Returns true if s is a Python keyword or built-in.
bool IsPythonReserved(const string& s);

// Whether the op should be prefixed with underscore.
bool IsOpWithUnderscorePrefix(const string& s);

// Add a _ to the end of s if necessary to avoid a Python keyword or built-in.
// Also convert namespace characters ('>') to '_' because python does not
// support '>' in names
string AvoidPythonReserved(const string& s);

// Convert an AttrValue with type `type` to the Python representation for
// that value.
string AttrValueToPython(const string& type, const AttrValue& value,
                         const string& dtype_module = "tf.");

void GenerateLowerCaseOpName(const string& str, string* result);

string DataTypeToPython(DataType dtype, const string& dtype_module);

// Names that corresponds to a single input parameter.
class ParamNames {
 public:
  // Create param based on Arg.
  ParamNames(const string& name, const string& rename_to) : name_(name) {
    rename_to_ = AvoidPythonReserved(rename_to);
  }

  // Get original parameter name.
  string GetName() const { return name_; }

  // Get the name to rename the parameter to. Note that AvoidPythonReserved
  // has already been applied.
  string GetRenameTo() const { return rename_to_; }

 private:
  // Original parameter name.
  string name_;
  // API name for this parameter.
  string rename_to_;
};

class GenPythonOp {
 public:
  GenPythonOp(
      const OpDef& op_def, const ApiDef& api_def, const string& function_name,
      python_op_gen_internal::GeneratedCodeAnnotator* annotator = nullptr)
      : op_def_(op_def),
        api_def_(api_def),
        function_name_(function_name),
        num_outs_(op_def.output_arg_size()),
        annotator_(annotator) {
    op_name_ = function_name_;
    absl::ConsumePrefix(&op_name_, "_");
  }
  ~GenPythonOp() = default;

  string Code();

 protected:
  void AddDefLine(const string& function_name, const string& parameters);
  void AddDefLine(const string& parameters);

  // Format the Op's descriptions so that it can be a Python docstring.
  void AddDocStringDescription();

  void AddDocStringArgs();
  void AddDocStringInputs();
  void AddDocStringAttrs();
  void AddDocStringNameArg();
  void AddOutputGlobals();
  void AddDocStringOutputs();
  void AddBody(const string& prefix);
  void AddBodyNoReturn(const string& apply_prefix);
  void AddExport();

  void HandleGraphMode(const string& function_setup,
                       const std::vector<string>& output_sizes);

  string GetEagerNotAllowedError();
  void ExpectListArg(const string& indentation, const string& arg_name,
                     string* output);
  bool GetEagerFunctionSetup(const string& indentation, string* function_setup);
  void GetOutputSizesAndNumOutputsExpr(std::vector<string>* output_sizes,
                                       string* num_outputs_expr);

  void AddEagerFunctionTeardown(const string& indentation,
                                const std::vector<string>& output_sizes,
                                bool execute_record_gradient);

  bool AddEagerFastPathAndGraphCode(
      const string& parameters, const std::vector<string>& output_sizes,
      const string& eager_not_allowed_error,
      const std::unordered_map<string, string>& type_annotations);
  bool AddEagerFallbackCode(
      const string& parameters, const std::vector<string>& output_sizes,
      const string& num_outputs_expr, const string& eager_not_allowed_error,
      const std::unordered_map<string, string>& type_annotations);
  void AddEagerFastPathExecute();

  void AddEagerInferredAttrs(const string& indentation);
  void AddEagerInputCasts(const string& indentation);
  void AddEagerAttrs(const string& indentation);
  void AddEagerExecute(const string& indentation,
                       const string& num_outputs_expr);
  void AddFallbackDispatch(const string& prefix);
  void AddTypeBasedDispatch(const string& prefix);
  void AddTypeBasedDispatcherAlias();

  void AddRawOpExport(const string& parameters);

  std::unordered_map<string, string> GetTypeAnnotations();

  void GenerateTypeVars(
      const std::unordered_map<string, string>& type_annotations);

  void AddReturnTypeAnnotation(
      const std::unordered_map<string, string>& type_annotations);

  void AddAttrForArg(const string& attr, int arg_index) {
    gtl::InsertIfNotPresent(&inferred_attrs_, attr,
                            op_def_.input_arg(arg_index).name());
    auto iter = attr_to_args_.find(attr);
    if (iter == attr_to_args_.end()) {
      attr_to_args_.insert(AttrToArgMap::value_type(attr, {arg_index}));
    } else {
      iter->second.push_back(arg_index);
    }
  }

  // Returns a string expression representing a flattened list of all
  // the inputs given by `*input_indices` (or all inputs if
  // `input_indices` is nullptr).  `*output_sizes` can be used to unflatten.
  string FlattenInputs(const std::vector<int>* input_indices,
                       std::vector<string>* output_sizes) const;

  // From constructor arguments
  const OpDef& op_def_;
  const ApiDef& api_def_;
  const string function_name_;
  const int num_outs_;
  python_op_gen_internal::GeneratedCodeAnnotator* annotator_ = nullptr;
  uint32_t def_offset_start_ = 0;

  // Return value from Code() is prelude_ + result_.
  string prelude_;  // Code before function definition
  string result_;   // Function definition

  // Map from attr name to the first input arg it is inferred from
  std::unordered_map<string, string> inferred_attrs_;

  // The names of the non-inferred attrs, in parameter order
  std::vector<string> attrs_;

  // All parameters, including inputs & non-inferred attrs, required and those
  // with defaults, except "name"
  std::vector<ParamNames> param_names_;

  absl::string_view op_name_;
  typedef std::unordered_map<string, std::vector<int>> AttrToArgMap;
  AttrToArgMap attr_to_args_;
  std::unordered_map<string, string> attr_expressions_;
  // This has all the input args followed by those attrs that don't have
  // defaults.
  std::vector<ParamNames> params_no_default_;
  // The parameters with defaults (these have to be listed after those without).
  // No input args are included, just attrs.
  std::vector<std::pair<ParamNames, string>> params_with_default_;
};

string GetEagerPythonOp(
    const OpDef& op_def, const ApiDef& api_def, const string& function_name,
    python_op_gen_internal::GeneratedCodeAnnotator* annotator = nullptr) {
  return GenPythonOp(op_def, api_def, function_name, annotator).Code();
}

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
       "__package__"});

  return kPythonReserved->count(s) > 0;
}

bool IsOpWithUnderscorePrefix(const string& s) {
  static const std::set<string>* const kUnderscoreOps = new std::set<string>(
      {// Lowercase built-in functions and types in Python, from:
       // [x for x in dir(__builtins__) if x[0].islower()] except "round".
       // These need to be excluded so they don't conflict with actual built-in
       // functions since we use '*' imports.
       "abs", "all", "any", "apply", "bin", "bool", "buffer", "bytearray",
       "bytes", "callable", "chr", "classmethod", "cmp", "coerce", "compile",
       "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod",
       "enumerate", "eval", "execfile", "exit", "file", "filter", "float",
       "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
       "hex", "id", "input", "int", "intern", "isinstance", "issubclass",
       "iter", "len", "license", "list", "locals", "long", "map", "max",
       "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
       "print", "property", "quit", "range", "raw_input", "reduce", "reload",
       "repr", "reversed", "set", "setattr", "slice", "sorted", "staticmethod",
       "str", "sum", "super", "tuple", "type", "unichr", "unicode", "vars",
       "xrange", "zip",
       // These have the same name as ops defined in Python and might be used
       // incorrectly depending on order of '*' imports.
       // TODO(annarev): reduce usage of '*' imports and remove these from the
       // list.
       "fused_batch_norm", "histogram_fixed_width", "stack",
       "batch_norm_with_global_normalization", "clip_by_value"});
  return kUnderscoreOps->count(s) > 0;
}

string AvoidPythonReserved(const string& s) {
  // Convert namespace separators ('>' characters) to joiners
  string result = absl::StrReplaceAll(s, {{">", "_"}});

  if (IsPythonReserved(result)) return strings::StrCat(result, "_");
  return result;
}

// Indent the first line by "initial" spaces and all following lines
// by "rest" spaces.
string Indent(int initial, int rest, absl::string_view in) {
  // TODO(josh11b): Also word-wrapping?
  string copy(in.data(), in.size());
  absl::StripTrailingAsciiWhitespace(&copy);
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
void AppendWithinWidth(string* dest, absl::string_view append, int width) {
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
      prefix = strings::StrCat("A list with the same length as `",
                               AvoidPythonReserved(*original_arg), "` of");
    }

    if (arg.type() != DT_INVALID) {
      return strings::StrCat(prefix, " `Tensor` objects with type ",
                             TypeString(arg.type(), arg.is_ref()), ".");
    } else {
      original_arg = gtl::FindOrNull(inferred_attrs, arg.type_attr());
      if (arg.is_ref()) {
        strings::StrAppend(&prefix, " mutable");
      }
      if (original_arg == nullptr) {
        return strings::StrCat(prefix, " `Tensor` objects with type `",
                               arg.type_attr(), "`.");
      } else if (*original_arg == arg.name()) {
        const OpDef::AttrDef* attr = FindAttr(arg.type_attr(), op_def);
        if (attr->has_allowed_values()) {
          return strings::StrCat(prefix,
                                 " `Tensor` objects with the same type in: ",
                                 TypeListString(attr->allowed_values()), ".");
        } else {
          return strings::StrCat(prefix,
                                 " `Tensor` objects with the same type.");
        }
      } else {
        return strings::StrCat(prefix,
                               " `Tensor` objects with the same type as `",
                               AvoidPythonReserved(*original_arg), "`.");
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
          return strings::StrCat(prefix,
                                 is_output
                                     ? ". Has one of the following types: "
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

string GetReturns(const OpDef& op_def,
                  const std::vector<string>& output_type_string) {
  string result;
  DCHECK_EQ(op_def.output_arg_size(), output_type_string.size());
  const int num_outs = op_def.output_arg_size();
  strings::StrAppend(&result, "\n  Returns:\n");
  if (num_outs == 0) {
    strings::StrAppend(&result, "    The created Operation.\n");
  } else {
    if (num_outs == 1) {
      absl::string_view description = op_def.output_arg(0).description();
      if (ConsumeEquals(&description)) {  // Skip the generated type info.
        strings::StrAppend(&result, Indent(4, 4, description));
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
        strings::StrAppend(&result, Indent(4, 4, desc));
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
      strings::StrAppend(&result, "    A tuple of `Tensor` objects (",
                         absl::StrJoin(out_names, ", "), ").\n\n");
      for (int i = 0; i < num_outs; ++i) {
        string desc = strings::StrCat(out_names[i], ": ");
        absl::string_view description = op_def.output_arg(i).description();
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
        strings::StrAppend(&result, Indent(4, 6, desc));
      }
    }
  }
  return result;
}

string StringToPython(const string& str) {
  return strings::StrCat("\"", absl::CEscape(str), "\"");
}

string DataTypeToPython(DataType dtype, const string& dtype_module) {
  return strings::StrCat(dtype_module, PythonDataTypeString(dtype));
}

string ShapeToPython(const TensorShapeProto& shape) {
  if (shape.unknown_rank()) {
    return "None";
  }
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

string TensorToPython(const TensorProto& proto) {
  return tsl::LegacyUnredactedShortDebugString(proto);
}

string AttrListToPython(const AttrValue& value,
                        const string& dtype_module = "tf.") {
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
      strings::StrAppend(&ret,
                         DataTypeToPython(value.list().type(i), dtype_module));
    }
  } else if (value.list().shape_size() > 0) {
    for (int i = 0; i < value.list().shape_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, ShapeToPython(value.list().shape(i)));
    }
  } else if (value.list().tensor_size() > 0) {
    for (int i = 0; i < value.list().tensor_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, TensorToPython(value.list().tensor(i)));
    }
  } else if (value.list().func_size() > 0) {
    for (int i = 0; i < value.list().func_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, StringToPython(value.list().func(i).name()));
    }
  }
  return ret;
}

// NOTE: The return value may contain spaces (for example, it could be
// a string "foo bar" with an embedded space) and is not safe to pass
// to WordWrap().
string AttrValueToPython(const string& type, const AttrValue& value,
                         const string& dtype_module) {
  if (type == "string") {
    return StringToPython(value.s());
  } else if (type == "int") {
    return strings::StrCat(value.i());
  } else if (type == "float") {
    if (std::isnan(value.f()) || std::isinf(value.f())) {
      return strings::StrCat("float('", value.f(), "')");
    } else {
      // Use locale-independent conversion.
      static_assert(FLT_DIG < 10, "FLT_DIG is too big");
      std::ostringstream s;
      s.imbue(std::locale::classic());
      s << std::setprecision(FLT_DIG) << value.f();
      // If there is no I/O error for `std::ostringstream s` return s.str(),
      // otherwise fallback to strings::StrCat(value.f()).
      if (s.good()) {
        return s.str();
      }
      return strings::StrCat(value.f());
    }
  } else if (type == "bool") {
    return value.b() ? "True" : "False";
  } else if (type == "type") {
    return DataTypeToPython(value.type(), dtype_module);
  } else if (type == "shape") {
    return ShapeToPython(value.shape());
  } else if (type == "tensor") {
    return TensorToPython(value.tensor());
  } else if (type == "func") {
    return StringToPython(value.func().name());
  } else if (absl::StartsWith(type, "list(")) {
    return strings::StrCat("[", AttrListToPython(value, dtype_module), "]");
  } else {
    return "?";
  }
}

void GenerateLowerCaseOpName(const string& str, string* result) {
  const char joiner = '_';
  const char namespace_separator = '>';
  const int last_index = str.size() - 1;
  for (int i = 0; i <= last_index; ++i) {
    const char c = str[i];
    // Convert namespace separators ('>' characters) to joiners
    if (c == namespace_separator) {
      result->push_back(joiner);
      continue;
    }

    // Emit a joiner only if a previous-lower-to-now-upper or a
    // now-upper-to-next-lower transition happens.
    // (But don't emit an extra joiner if we just saw a namespace separator
    if (isupper(c) && (i > 0)) {
      if (islower(str[i - 1]) || ((i < last_index) && islower(str[i + 1]))) {
        if (!(str[i - 1] == namespace_separator)) {
          result->push_back(joiner);
        }
      }
    }
    result->push_back(tolower(c));
  }
}

static void AddDelimiter(string* append_to, const string& delim) {
  if (!append_to->empty()) strings::StrAppend(append_to, delim);
}

const ApiDef::Attr* FindAttr(absl::string_view name, const ApiDef& api_def) {
  for (int i = 0; i < api_def.attr_size(); ++i) {
    if (api_def.attr(i).name() == name) {
      return &api_def.attr(i);
    }
  }
  return nullptr;
}

void GenPythonOp::AddExport() {
  if (api_def_.visibility() != ApiDef::VISIBLE) {
    return;
  }
  // Whether op should be available in latest export version.
  bool op_available_in_latest =
      !api_def_.deprecation_version() ||
      api_def_.deprecation_version() > kLatestAPIExportVersion;

  string names;
  string names_v1;
  string deprecated_endpoints;

  for (const auto& endpoint : api_def_.endpoint()) {
    string endpoint_name;
    GenerateLowerCaseOpName(endpoint.name(), &endpoint_name);
    if (endpoint.deprecated() || endpoint.deprecation_version() > 0) {
      AddDelimiter(&deprecated_endpoints, ", ");
      strings::StrAppend(&deprecated_endpoints, "'", endpoint_name, "'");
    }
    // Add all endpoints to TensorFlow 1.* API.
    AddDelimiter(&names_v1, ", ");
    strings::StrAppend(&names_v1, "'", endpoint_name, "'");
    // Add non-deprecated endpoints to TensorFlow 2.* API.
    if (op_available_in_latest &&
        (!endpoint.deprecation_version() ||
         endpoint.deprecation_version() > kLatestAPIExportVersion)) {
      AddDelimiter(&names, ", ");
      strings::StrAppend(&names, "'", endpoint_name, "'");
    }
  }

  // tf_export decorator has the following format:
  // @tf_export(v2_name, v2_name, v1=[v1_name, v1_name])
  if (names != names_v1) {
    AddDelimiter(&names, ", ");
    strings::StrAppend(&names, "v1=[", names_v1, "]");
  }
  strings::StrAppend(&result_, "@tf_export(", names, ")\n");

  // If all endpoints are deprecated, add @deprecated decorator.
  if (!api_def_.deprecation_message().empty()) {
    const string instructions = api_def_.deprecation_message();
    strings::StrAppend(&result_, "@deprecated(None, '", instructions, "')\n");
  }
  // Add @deprecated_endpoints decorator.
  if (!deprecated_endpoints.empty()) {
    strings::StrAppend(&result_, "@deprecated_endpoints(", deprecated_endpoints,
                       ")\n");
  }
}

void GenPythonOp::AddDefLine(const string& function_name,
                             const string& parameters) {
  strings::StrAppend(&result_, "def ", function_name, "(", parameters, "):\n");
}

void GenPythonOp::AddDefLine(const string& parameters) {
  AddDefLine(function_name_, parameters);
}

void GenPythonOp::AddDocStringDescription() {
  string comment;
  if (api_def_.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(api_def_.summary(), "\n");
    if (!api_def_.description().empty()) {
      strings::StrAppend(&comment, "\n", Indent(2, 2, api_def_.description()));
    }
  }
  strings::StrAppend(&result_, "  r\"\"\"", comment, "\n");
}

void GenPythonOp::AddDocStringArgs() {
  strings::StrAppend(&result_, "  Args:\n");
}

void GenPythonOp::AddDocStringInputs() {
  for (int i = 0; i < api_def_.arg_order_size(); ++i) {
    const auto& arg = *FindInputArg(api_def_.arg_order(i), op_def_);
    const auto& api_def_arg = *FindInputArg(api_def_.arg_order(i), api_def_);
    absl::string_view description = api_def_arg.description();
    string desc;
    if (ConsumeEquals(&description)) {  // Skip the generated type info.
      desc = strings::StrCat(param_names_[i].GetRenameTo(), ": ");
    } else {
      desc = strings::StrCat(param_names_[i].GetRenameTo(), ": ",
                             ArgTypeName(op_def_, arg, inferred_attrs_, false));
    }
    if (!description.empty()) {
      AppendWithinWidth(&desc, description, kRightMargin - 4 /* indent */);
    }
    strings::StrAppend(&result_, Indent(4, 6, desc));
  }
}

void GenPythonOp::AddDocStringAttrs() {
  for (const string& name : attrs_) {
    const auto& attr = *FindAttr(name, op_def_);
    const auto& api_def_attr = *FindAttr(name, api_def_);
    string desc =
        strings::StrCat(AvoidPythonReserved(api_def_attr.rename_to()), ": ");

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
        {"tensor", "`tf.TensorProto`"},
        {"list(tensor)", "list of `tf.TensorProto` objects"},
        {"func", "function decorated with @Defun"},
        {"list(func)", "list of functions decorated with @Defun"},
    };
    for (size_t i = 0; i < TF_ARRAYSIZE(kAttrTypeName); ++i) {
      if (attr.type() == kAttrTypeName[i][0]) {
        string s;
        if (api_def_attr.has_default_value()) {
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

    if (api_def_attr.has_default_value()) {
      strings::StrAppend(
          &desc, " Defaults to `",
          AttrValueToPython(attr.type(), api_def_attr.default_value()), "`.");
    }
    if (!api_def_attr.description().empty()) {
      AppendWithinWidth(&desc, api_def_attr.description(),
                        kRightMargin - 4 /* indent */);
    }
    strings::StrAppend(&result_, Indent(4, 6, desc));
  }
}

void GenPythonOp::AddDocStringNameArg() {
  strings::StrAppend(&result_,
                     "    name: A name for the operation (optional).\n");
}

void GenPythonOp::AddOutputGlobals() {
  // Generate a namedtuple class to hold the outputs, if there are multiple.
  // Example:
  //
  // _OpOutputs = collections.namedtuple(
  //     "_OpOutputs",
  //     "out1 out2 out3")
  if (num_outs_ > 1) {
    std::vector<string> out_names;
    out_names.reserve(num_outs_);
    for (int i = 0; i < num_outs_; ++i) {
      const string out_name = !api_def_.out_arg(i).rename_to().empty()
                                  ? api_def_.out_arg(i).rename_to()
                                  : strings::StrCat("output", i);
      out_names.push_back(strings::StrCat("\"", out_name, "\""));
    }

    strings::StrAppend(&prelude_, "_", AvoidPythonReserved(op_def_.name()),
                       "Output = collections.namedtuple(\n");
    strings::StrAppend(&prelude_, "    \"", AvoidPythonReserved(op_def_.name()),
                       "\",\n");
    strings::StrAppend(&prelude_, "    [", absl::StrJoin(out_names, ", "),
                       "])");
    strings::StrAppend(&prelude_, "\n\n");
  }
  strings::StrAppend(&prelude_, "\n");
}

void GenPythonOp::AddDocStringOutputs() {
  std::vector<string> output_type_string;
  output_type_string.reserve(num_outs_);
  for (int i = 0; i < num_outs_; ++i) {
    output_type_string.push_back(
        ArgTypeName(op_def_, op_def_.output_arg(i), inferred_attrs_, true));
  }
  strings::StrAppend(&result_, GetReturns(op_def_, output_type_string));
}

void GenPythonOp::AddBody(const string& prefix) {
  const string apply_prefix = strings::StrCat(
      prefix, "_result = _op_def_lib.apply_op(\"", op_def_.name(), "\", ");
  AddBodyNoReturn(apply_prefix);
  if (num_outs_ > 1) {
    strings::StrAppend(&result_, prefix, "_result = _",
                       AvoidPythonReserved(op_def_.name()),
                       "Output._make(_result)\n");
  }
  strings::StrAppend(&result_, prefix, "return _result\n");
}

void GenPythonOp::AddBodyNoReturn(const string& apply_prefix) {
  string args;
  for (size_t i = 0; i < param_names_.size(); ++i) {
    strings::StrAppend(&args, AvoidPythonReserved(param_names_[i].GetName()),
                       "=", param_names_[i].GetRenameTo(), ", ");
  }
  strings::StrAppend(&args, "name=name)");

  strings::StrAppend(&result_,
                     // Wrap the arguments, and indent to the (.
                     WordWrap(apply_prefix, args, kRightMargin), "\n");
}

string GenPythonOp::FlattenInputs(const std::vector<int>* input_indices,
                                  std::vector<string>* output_sizes) const {
  string inputs;
  enum { STARTING, WAS_LIST_INPUT, WAS_SOLO_INPUT } inputs_state = STARTING;
  const int n = input_indices != nullptr ? input_indices->size()
                                         : op_def_.input_arg_size();
  for (int j = 0; j < n; ++j) {
    const int i = input_indices ? (*input_indices)[j] : j;
    const auto& arg(op_def_.input_arg(i));
    const bool is_list =
        !arg.type_list_attr().empty() || !arg.number_attr().empty();
    if (is_list) {
      if (inputs_state == WAS_SOLO_INPUT) {
        strings::StrAppend(&inputs, "] + ");
      } else if (inputs_state == WAS_LIST_INPUT) {
        strings::StrAppend(&inputs, " + ");
      }
      strings::StrAppend(&inputs, "list(", param_names_[i].GetRenameTo(), ")");
      inputs_state = WAS_LIST_INPUT;
      if (output_sizes != nullptr) {
        if (!arg.number_attr().empty()) {
          output_sizes->emplace_back(AttrVarName(arg.number_attr(), nullptr));
        } else {
          output_sizes->emplace_back(
              strings::StrCat("len(", param_names_[i].GetRenameTo(), ")"));
        }
      }
    } else {
      if (inputs_state == WAS_SOLO_INPUT) {
        strings::StrAppend(&inputs, ", ");
      } else if (inputs_state == WAS_LIST_INPUT) {
        strings::StrAppend(&inputs, " + [");
      } else {
        strings::StrAppend(&inputs, "[");
      }
      strings::StrAppend(&inputs, param_names_[i].GetRenameTo());
      inputs_state = WAS_SOLO_INPUT;
      if (output_sizes != nullptr) output_sizes->emplace_back();
    }
  }
  if (inputs_state == STARTING) return "[]";
  if (inputs_state == WAS_SOLO_INPUT) {
    strings::StrAppend(&inputs, "]");
  }
  return inputs;
}

string GenPythonOp::Code() {
  if (api_def_.visibility() == ApiDef::SKIP) {
    return "";
  }

  for (int i = 0; i < api_def_.arg_order_size(); ++i) {
    const auto& arg = *FindInputArg(api_def_.arg_order(i), op_def_);
    const auto& api_def_arg = *FindInputArg(api_def_.arg_order(i), api_def_);
    params_no_default_.emplace_back(api_def_arg.name(),
                                    api_def_arg.rename_to());
    if (!arg.type_attr().empty()) {
      AddAttrForArg(arg.type_attr(), i);
    } else if (!arg.type_list_attr().empty()) {
      AddAttrForArg(arg.type_list_attr(), i);
    }
    if (!arg.number_attr().empty()) {
      AddAttrForArg(arg.number_attr(), i);
    }
  }
  for (int i = 0; i < op_def_.attr_size(); ++i) {
    const auto& attr(op_def_.attr(i));
    const auto& api_def_attr(api_def_.attr(i));
    // Do not add inferred attrs to the Python function signature.
    if (inferred_attrs_.find(attr.name()) == inferred_attrs_.end()) {
      if (api_def_attr.has_default_value()) {
        if (attr.type() == "tensor") {
          params_with_default_.emplace_back(
              ParamNames(api_def_attr.name(), api_def_attr.rename_to()),
              strings::StrCat(
                  "_execute.make_tensor(",
                  TensorPBString(api_def_attr.default_value().tensor()), ", \"",
                  api_def_attr.rename_to(), "\")"));
        } else if (attr.type() == "list(tensor)") {
          std::vector<string> pbtxt;
          for (const auto& pb : api_def_attr.default_value().list().tensor()) {
            pbtxt.emplace_back(TensorPBString(pb));
          }
          params_with_default_.emplace_back(
              ParamNames(api_def_attr.name(), api_def_attr.rename_to()),
              strings::StrCat("[_execute.make_tensor(_pb, \"",
                              api_def_attr.rename_to(), "\") for _pb in ",
                              VectorToTuple(pbtxt), "]"));
        } else {
          params_with_default_.emplace_back(
              ParamNames(api_def_attr.name(), api_def_attr.rename_to()),
              AttrValueToPython(attr.type(), api_def_attr.default_value(),
                                "_dtypes."));
        }
      } else {
        params_no_default_.emplace_back(api_def_attr.name(),
                                        api_def_attr.rename_to());
      }
    }
  }

  // Save the list of attr parameters (attrs that won't be inferred),
  // those with defaults go at the end.
  // Get the attrs in the order we want by taking the attrs without defaults
  // from the end of params_no_default_, and adding params_no_default_.
  attrs_.reserve(params_no_default_.size() - op_def_.input_arg_size() +
                 params_with_default_.size());
  for (int i = op_def_.input_arg_size(), end = params_no_default_.size();
       i < end; ++i) {
    attrs_.push_back(params_no_default_[i].GetName());
  }
  for (const auto& p : params_with_default_) {
    attrs_.push_back(p.first.GetName());
  }

  // TODO(slebedev): call AvoidPythonReserved on each param?
  param_names_.reserve(params_no_default_.size() + params_with_default_.size());
  param_names_.insert(param_names_.begin(), params_no_default_.begin(),
                      params_no_default_.end());
  for (const auto& param_and_default : params_with_default_) {
    param_names_.push_back(param_and_default.first);
  }

  std::unordered_map<string, string> type_annotations = GetTypeAnnotations();

  string parameters;
  // Param can be an input or an attr
  for (const auto& param : params_no_default_) {
    if (!parameters.empty()) strings::StrAppend(&parameters, ", ");
    strings::StrAppend(&parameters, param.GetRenameTo());

    if (type_annotations.find(param.GetName()) != type_annotations.end()) {
      strings::StrAppend(&parameters, ": ",
                         type_annotations.at(param.GetName()));
    }
  }

  string parameters_with_defaults = parameters;
  for (const auto& param_and_default : params_with_default_) {
    if (!parameters.empty()) strings::StrAppend(&parameters, ", ");
    if (!parameters_with_defaults.empty())
      strings::StrAppend(&parameters_with_defaults, ", ");

    strings::StrAppend(&parameters, param_and_default.first.GetRenameTo());
    strings::StrAppend(&parameters_with_defaults,
                       param_and_default.first.GetRenameTo());
    if (type_annotations.find(param_and_default.first.GetName()) !=
        type_annotations.end()) {
      const string param_type =
          type_annotations.at(param_and_default.first.GetName());
      // Append to parameters and parameters_with_defaults because multiple
      // functions are generated by AddEagerFastPathAndGraphCode() and
      // AddEagerFallbackCode()
      strings::StrAppend(&parameters, ": ", param_type);
      strings::StrAppend(&parameters_with_defaults, ":", param_type);
    }

    strings::StrAppend(&parameters_with_defaults, "=",
                       param_and_default.second);
  }

  strings::StrAppend(&parameters, parameters.empty() ? "" : ", ", "name");
  strings::StrAppend(&parameters_with_defaults,
                     parameters_with_defaults.empty() ? "" : ", ", "name=None");

  // Add attr_expressions_ for attrs that are params.
  for (int i = 0, end = attrs_.size(); i < end; ++i) {
    const string& attr_name = attrs_[i];
    const string& attr_api_name =
        param_names_[i + op_def_.input_arg_size()].GetRenameTo();
    attr_expressions_[attr_name] = attr_api_name;
  }
  // Add attr_expressions_ for attrs that are inferred.
  for (int i = 0; i < op_def_.attr_size(); ++i) {
    const auto& attr(op_def_.attr(i));
    if (attr.type() == "int") {
      auto arg_list = attr_to_args_.find(attr.name());
      if (arg_list != attr_to_args_.end()) {
        AttrVarName(attr.name(), &attr_expressions_);
      }
    }
  }

  string num_outputs_expr;
  std::vector<string> output_sizes(num_outs_);
  GetOutputSizesAndNumOutputsExpr(&output_sizes, &num_outputs_expr);

  string eager_not_allowed_error = GetEagerNotAllowedError();

  if (!AddEagerFastPathAndGraphCode(parameters_with_defaults, output_sizes,
                                    eager_not_allowed_error,
                                    type_annotations)) {
    return result_;
  }

  if (!AddEagerFallbackCode(parameters, output_sizes, num_outputs_expr,
                            eager_not_allowed_error, type_annotations)) {
    return result_;
  }

  if (annotator_ != nullptr) {
    // prelude_ will be prepended.
    def_offset_start_ += prelude_.length();
    annotator_->AddAnnotation(op_def_, function_name_, def_offset_start_);
  }

  return prelude_ + result_;
}

std::unordered_map<string, string> GenPythonOp::GetTypeAnnotations() {
  std::unordered_map<string, string> type_annotations;
  // Map attrs to TypeVars
  for (const auto& attr : op_def_.attr()) {
    if (attr.type() == "type") {
      const string type_var_name =
          AvoidPythonReserved("TV_" + op_def_.name() + "_" + attr.name());
      type_annotations[attr.name()] = type_var_name;
    } else if (attr.type() == "bool" || attr.type() == "float" ||
               attr.type() == "int" || attr.type() == "bytes") {
      type_annotations[attr.name()] = attr.type();
    } else if (attr.type() == "string") {
      type_annotations[attr.name()] = "str";
    }
  }

  // Map input Tensors to their types
  for (const auto& arg : op_def_.input_arg()) {
    // TODO(rahulkamat): Add type annotations to args that accept a sequence of
    // Tensors
    if (!arg.type_list_attr().empty()) continue;
    type_annotations[arg.name()] = GetArgAnnotation(arg, type_annotations);
  }

  // TODO(rahulkamat): Add type annotations to handle return types of a sequence
  // of Tensors. Map output Tensor to its type
  if (op_def_.output_arg_size() == 1) {
    const auto& arg = op_def_.output_arg(0);
    if (arg.number_attr().empty() && arg.type_list_attr().empty())
      type_annotations[arg.name()] = GetArgAnnotation(arg, type_annotations);
  }

  return type_annotations;
}

// Generate TypeVars using attrs
void GenPythonOp::GenerateTypeVars(
    const std::unordered_map<string, string>& type_annotations) {
  bool added_typevar = false;
  for (const auto& attr : op_def_.attr()) {
    if (attr.type() == "type") {
      std::vector<string> allowed_types;
      for (int t : attr.allowed_values().list().type()) {
        DataType dtype = static_cast<DataType>(t);
        const string py_dtype = DataTypeToPython(dtype, "_dtypes.");
        allowed_types.emplace_back(dtype_type.at(py_dtype));
      }

      // When a Tensor does not have any dtypes specified, all dtypes are
      // allowed
      if (allowed_types.empty()) {
        for (std::pair<string, string> map_dtype : dtype_type) {
          allowed_types.emplace_back(map_dtype.second);
        }
      }

      std::sort(allowed_types.begin(), allowed_types.end());

      // When there is only one type allowed make it a bound
      // TypeVars dont allow a single constraint
      string typevar_dtypes;
      if (allowed_types.size() == 1) {
        strings::StrAppend(&typevar_dtypes, "bound=", allowed_types[0]);
      } else {
        for (std::vector<string>::iterator it = allowed_types.begin();
             it != allowed_types.end(); ++it) {
          if (!typevar_dtypes.empty())
            strings::StrAppend(&typevar_dtypes, ", ");
          strings::StrAppend(&typevar_dtypes, "\"");
          strings::StrAppend(&typevar_dtypes, *it);
          strings::StrAppend(&typevar_dtypes, "\"");
        }
      }

      const string type_var_name = type_annotations.at(attr.name());
      strings::StrAppend(&result_, type_var_name, " = TypeVar(\"",
                         type_var_name, "\", ", typevar_dtypes, ")\n");
      added_typevar = true;
    }
  }

  if (added_typevar) strings::StrAppend(&result_, "\n");
}

void GenPythonOp::AddReturnTypeAnnotation(
    const std::unordered_map<string, string>& type_annotations) {
  if (op_def_.output_arg_size() == 1) {
    const auto& arg = op_def_.output_arg(0);
    if (arg.number_attr().empty() && arg.type_list_attr().empty()) {
      const string return_type = type_annotations.at(arg.name());
      // TODO(rahulkamat): Modify AddDefLine() to add return type annotation to
      // avoid erasing ":\n" from the end of the def line
      result_.erase(result_.length() - 2);
      strings::StrAppend(&result_, " -> ", return_type, ":\n");
    }
  }
}

void GenPythonOp::HandleGraphMode(const string& function_setup,
                                  const std::vector<string>& output_sizes) {
  if (api_def_.visibility() == ApiDef::VISIBLE) {
    strings::StrAppend(&result_, "  else:\n");
    AddTypeBasedDispatch("    ");
  }
  strings::StrAppend(&result_, "  # Add nodes to the TensorFlow graph.\n");
  strings::StrAppend(&result_, function_setup);
  if (api_def_.visibility() == ApiDef::VISIBLE) {
    strings::StrAppend(&result_, "  try:\n  ");
  }
  strings::StrAppend(
      &result_, "  _, _, _op, _outputs = _op_def_library._apply_op_helper(\n");
  AddBodyNoReturn(strings::StrCat("        \"", op_def_.name(), "\", "));
  AddFallbackDispatch("  ");

  if (num_outs_ > 0) {
    strings::StrAppend(&result_, "  _result = _outputs[:]\n");
    // Special case handling for stateful op with single list output
    // that might be empty.
    if (num_outs_ == 1 && op_def_.is_stateful() &&
        (!op_def_.output_arg(0).number_attr().empty() ||
         !op_def_.output_arg(0).type_list_attr().empty())) {
      // TODO(josh11b): Can skip this if the number_attr/type_list_attr has
      // a constraint indicating that this can never be empty.
      strings::StrAppend(&result_,
                         "  if not _result:\n"
                         "    return _op\n");
    }

    // Compute graph-mode attrs when we need to record a gradient.
    strings::StrAppend(&result_, "  if _execute.must_record_gradient():\n");
    if (op_def_.attr_size() > 0) {
      string attr_values;
      for (int i = 0; i < op_def_.attr_size(); ++i) {
        if (i > 0) strings::StrAppend(&attr_values, ", ");
        const auto& attr_name(op_def_.attr(i).name());
        if (op_def_.attr(i).type() == "type") {
          strings::StrAppend(&attr_values, "\"", attr_name,
                             "\", _op._get_attr_type(\"", attr_name, "\")");
        } else if (op_def_.attr(i).type() == "bool") {
          strings::StrAppend(&attr_values, "\"", attr_name,
                             "\", _op._get_attr_bool(\"", attr_name, "\")");
        } else if (op_def_.attr(i).type() == "int") {
          strings::StrAppend(&attr_values, "\"", attr_name,
                             "\", _op._get_attr_int(\"", attr_name, "\")");
        } else {
          strings::StrAppend(&attr_values, "\"", attr_name,
                             "\", _op.get_attr(\"", attr_name, "\")");
        }
      }
      strings::StrAppend(&attr_values, ")");
      strings::StrAppend(&result_,
                         WordWrap("    _attrs = (", attr_values, kRightMargin),
                         "\n");

    } else {
      strings::StrAppend(&result_, "    _attrs = ()\n");
    }

    strings::StrAppend(&result_, "    _inputs_flat = _op.inputs\n");
    strings::StrAppend(&result_, "    _execute.record_gradient(\n",
                       "        \"", op_def_.name(),
                       "\", _inputs_flat, _attrs, _result)\n");

    if (num_outs_ == 1 && !output_sizes[0].empty()) {
      // Single list result.
    } else if (num_outs_ == 1) {
      // Execute returns a single-element list which we need to destructure.
      strings::StrAppend(&result_, "  ", "_result, = _result\n");
    } else {
      // Have multiple outputs, so we will need to reformat the return
      // value of execute() to be a list with one entry per op output
      // (that entry will be a list of tensors if that output is of list
      // type).
      // For list outputs, convert the right subrange of _result into a list.
      Unflatten("  ", output_sizes, "_result", &result_);
      // Convert to a named tuple.
      strings::StrAppend(&result_, "  _result = _",
                         AvoidPythonReserved(op_def_.name()),
                         "Output._make(_result)\n");
    }
    strings::StrAppend(&result_, "  return _result\n\n");
  } else {
    strings::StrAppend(&result_, "  return _op\n");
  }
}

string GenPythonOp::GetEagerNotAllowedError() {
  bool eager_allowed = true;
  string ref_arg;
  for (int i = 0; i < op_def_.input_arg_size(); ++i) {
    const auto& arg = op_def_.input_arg(i);
    if (arg.is_ref()) {
      eager_allowed = false;
      DCHECK_EQ(op_def_.input_arg(i).name(), api_def_.in_arg(i).name());
      ref_arg = api_def_.in_arg(i).rename_to();
    }
  }
  for (int i = 0; i < op_def_.output_arg_size(); ++i) {
    const auto& arg = op_def_.output_arg(i);
    if (arg.is_ref()) {
      eager_allowed = false;
      DCHECK_EQ(op_def_.output_arg(i).name(), api_def_.out_arg(i).name());
      ref_arg = api_def_.out_arg(i).rename_to();
    }
  }

  if (eager_allowed) return "";

  return strings::StrCat("raise RuntimeError(\"", op_name_,
                         " op does not support eager execution. ", "Arg '",
                         ref_arg, "' is a ref.\")\n");
}

void GenPythonOp::ExpectListArg(const string& indentation,
                                const string& arg_name, string* output) {
  strings::StrAppend(output, indentation, "if not isinstance(", arg_name,
                     ", (list, tuple)):\n", indentation, "  raise TypeError(\n",
                     indentation, "      \"Expected list for '", arg_name,
                     "' argument to \"\n", indentation, "      \"'", op_name_,
                     "' Op, not %r.\" % ", arg_name, ")\n");
}

bool GenPythonOp::GetEagerFunctionSetup(const string& indentation,
                                        string* function_setup) {
  // Validate list inputs, infer length attrs.
  for (int i = 0; i < op_def_.attr_size(); ++i) {
    const auto& attr(op_def_.attr(i));
    if (attr.type() == "int") {
      auto arg_list = attr_to_args_.find(attr.name());
      if (arg_list != attr_to_args_.end()) {
        // Inferred int attrs are the lengths of inputs. Validate those
        // inputs are lists and have the same length.
        for (auto iter = arg_list->second.begin();
             iter != arg_list->second.end(); ++iter) {
          const string& arg_api_name = param_names_[*iter].GetRenameTo();
          ExpectListArg(indentation, arg_api_name, function_setup);
          if (iter == arg_list->second.begin()) {
            AddInferredAttr(indentation, attr.name(),
                            strings::StrCat("len(", arg_api_name, ")"),
                            function_setup, &attr_expressions_);
          } else {
            const auto& attr_var = attr_expressions_[attr.name()];
            strings::StrAppend(
                function_setup, indentation, "if len(", arg_api_name,
                ") != ", attr_var, ":\n", indentation, "  raise ValueError(\n",
                indentation, "      \"List argument '", arg_api_name, "' to '",
                op_name_, "' Op with length %d \"\n", indentation,
                "      \"must match length %d of argument '",
                inferred_attrs_[attr.name()], "'.\" %\n", indentation,
                "      (len(", arg_api_name, "), ", attr_var, "))\n");
          }
        }
      }
    }
  }

  for (int i = 0, end = attrs_.size(); i < end; ++i) {
    const string& attr_name = attrs_[i];
    const auto& param = param_names_[i + op_def_.input_arg_size()];
    const auto& attr = *FindAttr(attr_name, op_def_);
    const string& attr_api_name = param.GetRenameTo();
    absl::string_view attr_type = attr.type();
    attr_expressions_[attr_name] = attr_api_name;
    const int default_index = i - (attrs_.size() - params_with_default_.size());
    if (default_index >= 0) {
      const string& default_value = params_with_default_[default_index].second;
      strings::StrAppend(function_setup, indentation, "if ", attr_api_name,
                         " is None:\n");
      strings::StrAppend(function_setup, indentation, "  ", attr_api_name,
                         " = ", default_value, "\n");
    }
    if (absl::StartsWith(attr_type, "list(")) {
      ExpectListArg(indentation, attr_api_name, function_setup);
    }

    if (attr_type == "string") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_str(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(string)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_str(_s, \"", attr_api_name,
                         "\") for _s in ", attr_api_name, "]\n");
    } else if (attr_type == "int") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_int(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(int)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_int(_i, \"", attr_api_name,
                         "\") for _i in ", attr_api_name, "]\n");
    } else if (attr_type == "float") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_float(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(float)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_float(_f, \"", attr_api_name,
                         "\") for _f in ", attr_api_name, "]\n");
    } else if (attr_type == "bool") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_bool(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(bool)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_bool(_b, \"", attr_api_name,
                         "\") for _b in ", attr_api_name, "]\n");
    } else if (attr_type == "type") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_type(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(type)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_type(_t, \"", attr_api_name,
                         "\") for _t in ", attr_api_name, "]\n");
    } else if (attr_type == "shape") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_shape(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(shape)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_shape(_s, \"", attr_api_name,
                         "\") for _s in ", attr_api_name, "]\n");
    } else if (attr_type == "tensor") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = _execute.make_tensor(", attr_api_name, ", \"",
                         attr_api_name, "\")\n");
    } else if (attr_type == "list(tensor)") {
      strings::StrAppend(function_setup, indentation, attr_api_name,
                         " = [_execute.make_tensor(_t, \"", attr_api_name,
                         "\") for _t in ", attr_api_name, "]\n");
    } else if (attr_type != "func" && attr_type != "list(func)") {
      *function_setup =
          strings::StrCat("# No definition for ", function_name_,
                          " since we don't support attrs with type\n"
                          "# '",
                          attr_type, "' right now.\n\n");
      return false;
    }
  }
  return true;
}

// If output i is list output, output_sizes[i] will be set to a
// string with the python expression that will evaluate to its
// length. output_sizes[i] is empty for non-list outputs.
void GenPythonOp::GetOutputSizesAndNumOutputsExpr(
    std::vector<string>* output_sizes, string* num_outputs_expr) {
  // Expression representing the number of outputs.
  int num_fixed_outputs = 0;
  for (int i = 0; i < num_outs_; ++i) {
    const auto& arg(op_def_.output_arg(i));
    if (!arg.number_attr().empty()) {
      if (!num_outputs_expr->empty()) {
        strings::StrAppend(num_outputs_expr, " + ");
      }
      (*output_sizes)[i] = attr_expressions_[arg.number_attr()];
      strings::StrAppend(num_outputs_expr, (*output_sizes)[i]);
    } else if (!arg.type_list_attr().empty()) {
      if (!num_outputs_expr->empty()) {
        strings::StrAppend(num_outputs_expr, " + ");
      }
      // Have to be careful to use an expression that works in both
      // graph and eager paths here.
      const auto iter = inferred_attrs_.find(arg.type_list_attr());
      if (iter == inferred_attrs_.end()) {
        (*output_sizes)[i] = strings::StrCat(
            "len(", attr_expressions_[arg.type_list_attr()], ")");
      } else {
        (*output_sizes)[i] = strings::StrCat("len(", iter->second, ")");
      }
      strings::StrAppend(num_outputs_expr, (*output_sizes)[i]);
    } else {
      ++num_fixed_outputs;
    }
  }
  if (num_fixed_outputs > 0) {
    if (!num_outputs_expr->empty()) {
      strings::StrAppend(num_outputs_expr, " + ");
    }
    strings::StrAppend(num_outputs_expr, num_fixed_outputs);
  } else if (num_outputs_expr->empty()) {
    *num_outputs_expr = "0";
  }
}

void GenPythonOp::AddEagerFunctionTeardown(
    const string& indentation, const std::vector<string>& output_sizes,
    bool execute_record_gradient) {
  if (num_outs_ > 0) {
    if (execute_record_gradient) {
      strings::StrAppend(&result_, indentation,
                         "if _execute.must_record_gradient():\n");
      strings::StrAppend(&result_, indentation, "  _execute.record_gradient(\n",
                         "        \"", op_def_.name(),
                         "\", _inputs_flat, _attrs, _result)\n");
    }
    if (num_outs_ == 1 && !output_sizes[0].empty()) {
      // Single list result.
    } else if (num_outs_ == 1) {
      // Execute returns a single-element list which we need to destructure.
      strings::StrAppend(&result_, indentation, "_result, = _result\n");
    } else {
      // Have multiple outputs, so we will need to reformat the return
      // value of execute() to be a list with one entry per op output
      // (that entry will be a list of tensors if that output is of list
      // type).
      // For list outputs, convert the right subrange of _result into a list.
      Unflatten(indentation, output_sizes, "_result", &result_);
      // Convert to a named tuple.
      strings::StrAppend(&result_, indentation, "_result = _",
                         AvoidPythonReserved(op_def_.name()),
                         "Output._make(_result)\n");
    }
  } else {
    strings::StrAppend(&result_, indentation, "_result = None\n");
  }
  strings::StrAppend(&result_, indentation, "return _result\n\n");
}

bool GenPythonOp::AddEagerFastPathAndGraphCode(
    const string& parameters, const std::vector<string>& output_sizes,
    const string& eager_not_allowed_error,
    const std::unordered_map<string, string>& type_annotations) {
  GenerateTypeVars(type_annotations);
  if (api_def_.visibility() == ApiDef::VISIBLE) {
    strings::StrAppend(&result_, "@_dispatch.add_fallback_dispatch_list\n");
    strings::StrAppend(&result_, "@_dispatch.add_type_based_api_dispatcher\n");
  }

  AddExport();
  if (annotator_ != nullptr) {
    // The generated function name will start at the character after
    // the current cursor + len("def ")
    def_offset_start_ = result_.length() + 4;
  }
  AddDefLine(function_name_, parameters);
  AddReturnTypeAnnotation(type_annotations);
  AddDocStringDescription();
  AddDocStringArgs();
  AddDocStringInputs();
  AddDocStringAttrs();
  AddDocStringNameArg();
  AddOutputGlobals();  // Added to prelude_
  AddDocStringOutputs();
  strings::StrAppend(&result_, "  \"\"\"\n");

  strings::StrAppend(&result_,
                     "  _ctx = _context._context or _context.context()\n"
                     "  tld = _ctx._thread_local_data\n",
                     "  if tld.is_eager:", "\n");
  if (eager_not_allowed_error.empty()) {
    AddEagerFastPathExecute();
  } else {
    strings::StrAppend(&result_, "    ", eager_not_allowed_error);
  }

  // Handle graph-mode case
  string function_setup;
  if (!GetEagerFunctionSetup("  ", &function_setup)) {
    result_ = function_setup;
    return false;
  }
  HandleGraphMode(function_setup, output_sizes);

  AddRawOpExport(parameters);
  AddTypeBasedDispatcherAlias();
  strings::StrAppend(&result_, "\n\n");
  return true;
}

bool GenPythonOp::AddEagerFallbackCode(
    const string& parameters, const std::vector<string>& output_sizes,
    const string& num_outputs_expr, const string& eager_not_allowed_error,
    const std::unordered_map<string, string>& type_annotations) {
  AddDefLine(
      strings::StrCat(function_name_, kEagerFallbackSuffix),
      strings::StrCat(parameters, parameters.empty() ? "" : ", ", "ctx"));
  AddReturnTypeAnnotation(type_annotations);
  if (!eager_not_allowed_error.empty()) {
    strings::StrAppend(&result_, "  ", eager_not_allowed_error);
    return true;
  }

  string function_setup;
  if (!GetEagerFunctionSetup("  ", &function_setup)) {
    result_ = function_setup;
    return false;
  }
  strings::StrAppend(&result_, function_setup);

  AddEagerInferredAttrs("  ");
  AddEagerInputCasts("  ");
  strings::StrAppend(
      &result_, "  _inputs_flat = ", FlattenInputs(nullptr, nullptr), "\n");
  AddEagerAttrs("  ");
  AddEagerExecute("  ", num_outputs_expr);

  AddEagerFunctionTeardown("  ", output_sizes,
                           true /* execute_record_gradient */);

  return true;
}

void GenPythonOp::AddEagerFastPathExecute() {
  string fastpath_execute_params =
      strings::StrCat("_ctx, \"", op_def_.name(), "\", ", "name");
  string fallback_params;

  for (int i = 0; i < api_def_.in_arg_size(); i++) {
    const string param_name = param_names_[i].GetRenameTo();
    strings::StrAppend(&fastpath_execute_params, ", ", param_name);
    if (!fallback_params.empty()) strings::StrAppend(&fallback_params, ", ");
    strings::StrAppend(&fallback_params, param_name);
  }

  for (const auto& attr : api_def_.attr()) {
    if (inferred_attrs_.find(attr.name()) == inferred_attrs_.end()) {
      strings::StrAppend(&fastpath_execute_params, ", \"", attr.name(), "\", ",
                         attr.rename_to());

      if (!fallback_params.empty()) strings::StrAppend(&fallback_params, ", ");
      strings::StrAppend(&fallback_params, attr.rename_to(), "=",
                         attr.rename_to());
    }
  }

  if (!fallback_params.empty()) strings::StrAppend(&fallback_params, ", ");
  strings::StrAppend(&fallback_params, "name=name");

  strings::StrAppend(&result_, "    try:\n");
  strings::StrAppend(
      &result_, "      ", "_result = pywrap_tfe.TFE_Py_FastPathExecute(\n",
      WordWrap(strings::StrCat("        "),
               strings::StrCat(fastpath_execute_params, ")"), kRightMargin),
      "\n");

  if (op_def_.output_arg_size() > 1) {
    const string output_tuple_name =
        strings::StrCat("_", AvoidPythonReserved(op_def_.name()), "Output");
    strings::StrAppend(&result_, "      ", "_result = ", output_tuple_name,
                       "._make(_result)\n");
  }
  strings::StrAppend(&result_, "      ", "return _result\n");

  // Handle fallback.
  if (!fallback_params.empty()) strings::StrAppend(&fallback_params, ", ");
  strings::StrAppend(&fallback_params, "ctx=_ctx");

  // Any errors thrown from execute need to be unwrapped from
  // _NotOkStatusException.
  strings::StrAppend(&result_, "    ",
                     "except _core._NotOkStatusException as e:\n");
  strings::StrAppend(&result_, "      ",
                     "_ops.raise_from_not_ok_status(e, name)\n");

  strings::StrAppend(&result_, "    ", "except _core._FallbackException:\n");
  strings::StrAppend(&result_, "      pass\n");
  strings::StrAppend(&result_, "    try:\n");
  AddTypeBasedDispatch("      ");
  strings::StrAppend(
      &result_, "      ", "return ", function_name_, kEagerFallbackSuffix,
      "(\n",
      WordWrap(strings::StrCat("          "),
               strings::StrCat(fallback_params, ")"), kRightMargin),
      "\n");
  strings::StrAppend(&result_, "    except _core._SymbolicException:\n");
  strings::StrAppend(&result_,
                     "      pass  # Add nodes to the TensorFlow graph.\n");
  AddFallbackDispatch("    ");
}

void GenPythonOp::AddEagerInferredAttrs(const string& indentation) {
  // Figure out values for inferred attrs, and cast to eager tensors.
  for (int i = 0; i < op_def_.attr_size(); ++i) {
    const auto& attr(op_def_.attr(i));
    const auto& api_def_attr(api_def_.attr(i));
    auto arg_list = attr_to_args_.find(attr.name());
    if (arg_list != attr_to_args_.end()) {
      if (attr.type() == "type") {
        std::vector<string> output_sizes;
        const string flattened =
            FlattenInputs(&arg_list->second, &output_sizes);
        string conversion = strings::StrCat("_execute.args_to_matching_eager(",
                                            flattened, ", ctx");

        strings::StrAppend(&conversion, ", [");
        for (int t : attr.allowed_values().list().type()) {
          DataType dtype = static_cast<DataType>(t);
          const string py_dtype = DataTypeToPython(dtype, "_dtypes.");
          strings::StrAppend(&conversion, py_dtype, ", ");
        }
        strings::StrAppend(&conversion, "]");

        if (attr.has_default_value()) {
          strings::StrAppend(
              &conversion, ", ",
              AttrValueToPython(attr.type(), api_def_attr.default_value(),
                                "_dtypes."));
        }
        strings::StrAppend(&conversion, ")");
        const string var_name = AttrVarName(attr.name(), &attr_expressions_);
        if (output_sizes.size() == 1) {
          // Avoid creating a temporary variable in the case where
          // we can easily assign to the right value directly.
          const string inputs_var =
              param_names_[arg_list->second.front()].GetRenameTo();
          if (output_sizes.front().empty()) {
            strings::StrAppend(&result_, indentation, var_name, ", (",
                               inputs_var, ",) = ", conversion, "\n");
          } else {
            strings::StrAppend(&result_, indentation, var_name, ", ",
                               inputs_var, " = ", conversion, "\n");
          }
        } else {
          const string inputs_var = strings::StrCat("_inputs_", attr.name());
          strings::StrAppend(&result_, indentation, var_name, ", ", inputs_var,
                             " = ", conversion, "\n");
          // Convert from a flat list of eager tensors back to the
          // parameter variables.
          Unflatten(indentation, output_sizes, inputs_var, &result_);
          std::vector<string> p;
          for (int j : arg_list->second) {
            p.emplace_back(param_names_[j].GetRenameTo());
          }
          strings::StrAppend(&result_, indentation, VectorToTuple(p), " = ",
                             inputs_var, "\n");
        }
      } else if (attr.type() == "list(type)") {
        // NOTE: We ignore default values for these attrs, since it is
        // unclear how you would use it, and the one use case is
        // parse_single_sequence_example which only needs it for
        // backwards compatibility.
        const string var_name = AttrVarName(attr.name(), &attr_expressions_);
        string inputs_var;
        string conversion;
        if (arg_list->second.size() > 1) {
          // If you have more than one list(tensor) argument, their types
          // have to match.
          std::vector<string> lists;
          for (auto iter = arg_list->second.begin();
               iter != arg_list->second.end(); ++iter) {
            lists.push_back(param_names_[*iter].GetRenameTo());
          }
          inputs_var = VectorToTuple(lists);
          conversion = "_execute.args_to_mixed_eager_tensors";
        } else {
          // For one list(tensor) argument, we just convert every
          // element of the list to an eager tensor.
          inputs_var = param_names_[arg_list->second.front()].GetRenameTo();
          conversion = "_execute.convert_to_mixed_eager_tensors";
        }
        strings::StrAppend(&result_, indentation, var_name, ", ", inputs_var,
                           " = ", conversion, "(", inputs_var, ", ctx)\n");
      }
    }
  }
}

void GenPythonOp::AddEagerInputCasts(const string& indentation) {
  // Cast remaining args to eager tensors
  for (int i = 0; i < op_def_.input_arg_size(); ++i) {
    const auto& arg(op_def_.input_arg(i));
    if (!arg.type_attr().empty() || !arg.type_list_attr().empty()) continue;
    const string& param = param_names_[i].GetRenameTo();
    const string fn = arg.number_attr().empty() ? "" : "n_";
    const string dtype = DataTypeToPython(arg.type(), "_dtypes.");
    strings::StrAppend(&result_, indentation, param, " = _ops.convert_", fn,
                       "to_tensor(", param, ", ", dtype, ")\n");
  }
}

void GenPythonOp::AddEagerAttrs(const string& indentation) {
  // Compute eager attrs
  if (op_def_.attr_size() > 0) {
    string attr_values;
    for (int i = 0; i < op_def_.attr_size(); ++i) {
      if (i > 0) strings::StrAppend(&attr_values, ", ");
      const auto& attr_name(op_def_.attr(i).name());
      strings::StrAppend(&attr_values, "\"", attr_name, "\", ",
                         attr_expressions_[attr_name]);
    }
    strings::StrAppend(&attr_values, ")");
    strings::StrAppend(
        &result_,
        WordWrap(indentation, strings::StrCat("_attrs = (", attr_values),
                 kRightMargin),
        "\n");
  } else {
    strings::StrAppend(&result_, indentation, "_attrs = None\n");
  }
}

void GenPythonOp::AddEagerExecute(const string& indentation,
                                  const string& num_outputs_expr) {
  const string return_prefix =
      strings::StrCat(indentation, "_result = _execute.execute(");
  const string return_args = strings::StrCat(
      "b\"", op_def_.name(), "\", ", num_outputs_expr,
      ", inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)");
  strings::StrAppend(&result_,
                     // Wrap the arguments, and indent to the (.
                     WordWrap(return_prefix, return_args, kRightMargin), "\n");
}

void GenPythonOp::AddFallbackDispatch(const string& prefix) {
  if (api_def_.visibility() != ApiDef::VISIBLE) return;

  strings::StrAppend(&result_, prefix, "except (TypeError, ValueError):\n");
  strings::StrAppend(&result_, prefix, "  _result = _dispatch.dispatch(\n");
  AddBodyNoReturn(strings::StrCat(prefix, "        ", function_name_,
                                  ", "
                                  "(), dict("));
  strings::StrAppend(&result_, prefix, "      )\n");
  strings::StrAppend(&result_, prefix,
                     "  if _result is not "
                     "_dispatch.OpDispatcher.NOT_SUPPORTED:\n");
  strings::StrAppend(&result_, prefix, "    return _result\n");
  strings::StrAppend(&result_, prefix, "  raise\n");
}

void GenPythonOp::AddTypeBasedDispatcherAlias() {
  // It's possible for the name of a parameter to be the same as the name of
  // an op, in which case the parameter shadows the op's function.  To avoid
  // this, we add a private variable with the dispatcher, and access that
  // directly.
  if (api_def_.visibility() == ApiDef::VISIBLE) {
    strings::StrAppend(&result_, "_dispatcher_for_", function_name_, " = ",
                       function_name_, "._tf_type_based_dispatcher.Dispatch\n");
  }
}
void GenPythonOp::AddTypeBasedDispatch(const string& prefix) {
  if (api_def_.visibility() != ApiDef::VISIBLE) return;
  std::string args("(");
  for (const auto& name : param_names_) {
    strings::StrAppend(&args, name.GetRenameTo(), ", ");
  }
  strings::StrAppend(&args, "name,), None");

  strings::StrAppend(
      &result_, prefix, "_result = ", "_dispatcher_for_", function_name_, "(\n",
      WordWrap(strings::StrCat(prefix, "    "), args, kRightMargin), ")\n");
  strings::StrAppend(&result_, prefix, "if _result is not NotImplemented:\n",
                     prefix, "  return _result\n");
}

void GenPythonOp::AddRawOpExport(const string& parameters) {
  // Example:
  //
  // Identity = tf_export("raw_ops.Identity")(_ops._to_raw_op(identity))
  const string raw_function_name = AvoidPythonReserved(op_def_.name());
  strings::StrAppend(&result_, raw_function_name, " = tf_export(\"raw_ops.",
                     raw_function_name, "\")", "(_ops.to_raw_op(",
                     function_name_, "))\n");
}

string GetPythonOpsImpl(const OpList& ops, const ApiDefMap& api_defs,
                        const OpRegOffsets& op_reg_offsets,
                        absl::Span<const string> hidden_ops,
                        absl::Span<const string> source_file_list) {
  python_op_gen_internal::GeneratedCodeAnnotator annotator;
  bool annotate = !op_reg_offsets.offsets().empty();

  string result;
  // Header
  // TODO(josh11b): Mention the library for which wrappers are being generated.
  strings::StrAppend(&result, R"("""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
)");

  // Mention the original source file so someone tracing back through
  // generated Python code will know where to look next.
  if (!source_file_list.empty()) {
    strings::StrAppend(&result, "Original C++ source file: ");
    strings::StrAppend(&result, absl::StrJoin(source_file_list, ", "));
    strings::StrAppend(&result, "\n");
  }

  strings::StrAppend(&result, R"("""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated
)");
  for (const auto& op_def : ops.op()) {
    const auto* api_def = api_defs.GetApiDef(op_def.name());

    if (api_def->visibility() == ApiDef::SKIP) {
      continue;
    }
    // An op is hidden if either its ApiDef visibility is HIDDEN
    // or it is in the hidden_ops list.
    bool is_hidden = api_def->visibility() == ApiDef::HIDDEN;
    bool hidden_by_api_def = is_hidden;
    if (!is_hidden) {
      for (const string& hidden : hidden_ops) {
        if (op_def.name() == hidden) {
          is_hidden = true;
          break;
        }
      }
    }

    string function_name;
    GenerateLowerCaseOpName(op_def.name(), &function_name);
    bool is_reserved = IsPythonReserved(function_name);

    // Prefix an op with underscore if the op is listed in hidden_ops or
    // name is reserved or it is of the exceptions in IsOpWithUnderscorePrefix.
    // Do not add underscores to ops set to HIDDEN in ApiDef otherwise.
    // TODO(annarev): don't prefix with underscores even if op is in hidden_ops.
    if (is_hidden) {
      if (!hidden_by_api_def || is_reserved ||
          IsOpWithUnderscorePrefix(function_name)) {
        function_name = strings::StrCat("_", function_name);
      }
    } else if (is_reserved) {
      // When users create custom python wrappers, they may link in the
      // default op registry by accident, and because they can't
      // enumerate all 'hidden' symbols, this guard is to prevent
      // instantiating a python reserved word in their wrapper.
      continue;
    }

    if (annotate) {
      annotator.SetBase(result.length());
    }
    strings::StrAppend(&result,
                       GetEagerPythonOp(op_def, *api_def, function_name,
                                        annotate ? &annotator : nullptr));
  }

  if (annotate) {
    annotator.FillSourceOffsets(op_reg_offsets);
    strings::StrAppend(&result, annotator.BuildKytheMetadata());
  }

  return result;
}

}  // namespace

string GetPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const OpRegOffsets& op_reg_offsets,
                    absl::Span<const string> hidden_ops,
                    absl::Span<const string> source_file_list) {
  return GetPythonOpsImpl(ops, api_defs, op_reg_offsets, hidden_ops,
                          source_file_list);
}

void PrintPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const OpRegOffsets& op_reg_offsets,
                    absl::Span<const string> hidden_ops,
                    absl::Span<const string> source_file_list) {
  printf("%s", GetPythonOpsImpl(ops, api_defs, op_reg_offsets, hidden_ops,
                                source_file_list)
                   .c_str());
}

string GetPythonWrappers(const char* op_list_buf, size_t op_list_len) {
  OpList ops;
  ops.ParseFromArray(op_list_buf, op_list_len);

  ApiDefMap api_def_map(ops);
  return GetPythonOpsImpl(ops, api_def_map, OpRegOffsets(), {}, {});
}

string GetSingleTensorArgAnnotation(
    const OpDef::ArgDef& arg,
    const std::unordered_map<string, string>& type_annotations) {
  if (!arg.type_attr().empty()) {
    // Get the correct TypeVar if arg maps to an attr
    return type_annotations.at(arg.type_attr());
  } else {
    // Get the dtype of the Tensor
    const string py_dtype = DataTypeToPython(arg.type(), "_dtypes.");
    return dtype_type.at(py_dtype);
  }
}

string GetArgAnnotation(
    const OpDef::ArgDef& arg,
    const std::unordered_map<string, string>& type_annotations) {
  if (!arg.number_attr().empty()) {
    return strings::StrCat("Annotated[List[Any], ",
                           GetSingleTensorArgAnnotation(arg, type_annotations),
                           "]");
  }
  return strings::StrCat("Annotated[Any, ",
                         GetSingleTensorArgAnnotation(arg, type_annotations),
                         "]");
}

}  // namespace tensorflow
