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

#include "tensorflow/core/framework/node_def_util.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

AttrSlice::AttrSlice(const NodeDef& node_def)
    : ndef_(&node_def), attrs_(&ndef_->attr()) {}

AttrSlice::AttrSlice(const AttrValueMap* a) : ndef_(nullptr), attrs_(a) {}

string SummarizeNodeDef(const NodeDef& node_def) {
  string ret = strings::StrCat(node_def.name(), " = ", node_def.op(), "[");

  // We sort the attrs so the output is deterministic.
  std::vector<string> attr_names;
  attr_names.reserve(node_def.attr().size());
  for (const auto& attr : node_def.attr()) {
    attr_names.push_back(attr.first);
  }
  std::sort(attr_names.begin(), attr_names.end());
  bool first = true;
  for (const string& attr_name : attr_names) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    auto iter = node_def.attr().find(attr_name);
    strings::StrAppend(&ret, attr_name, "=", SummarizeAttrValue(iter->second));
  }

  // Consider the device to be a final attr with name "_device".
  if (!node_def.device().empty()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, "_device=\"", node_def.device(), "\"");
  }
  strings::StrAppend(&ret, "](");

  // Output inputs, including control inputs, verbatim.
  first = true;
  for (const string& input : node_def.input()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, input);
  }
  strings::StrAppend(&ret, ")");
  return ret;
}

const AttrValue* AttrSlice::Find(StringPiece attr_name) const {
  auto iter = attrs_->find(attr_name.ToString());
  if (iter == attrs_->end()) return nullptr;
  return &iter->second;
}

Status AttrSlice::Find(StringPiece attr_name,
                       const AttrValue** attr_value) const {
  *attr_value = Find(attr_name);
  if (*attr_value != nullptr) {
    return Status::OK();
  }
  Status s = errors::NotFound("No attr named '", attr_name, "' in NodeDef:");
  // Skip AttachDef for internal attrs since it is a little bit
  // expensive and it is common for them to correctly not be included
  // in a NodeDef.
  if (!StringPiece(attr_name).starts_with("_") && ndef_) {
    s = AttachDef(s, *ndef_);
  }
  return s;
}

// The ... is to allow the caller to inject some value validation code.  Use
// just ; if no additional validation code is needed.
#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...)         \
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,           \
                     TYPE* value) {                                           \
    const AttrValue* attr_value;                                              \
    TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));                   \
    TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, ATTR_TYPE));             \
    const auto& v = attr_value->FIELD();                                      \
    __VA_ARGS__;                                                              \
    *value = CAST;                                                            \
    return Status::OK();                                                      \
  }                                                                           \
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,           \
                     std::vector<TYPE>* value) {                              \
    const AttrValue* attr_value;                                              \
    TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));                   \
    TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "list(" ATTR_TYPE ")")); \
    for (const auto& v : attr_value->list().FIELD()) {                        \
      __VA_ARGS__;                                                            \
      value->APPEND_OP(CAST);                                                 \
    }                                                                         \
    return Status::OK();                                                      \
  }

DEFINE_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_GET_ATTR(int32, i, "int", emplace_back, static_cast<int32>(v),
                if (static_cast<int64>(static_cast<int32>(v)) != v) {
                  return errors::InvalidArgument("Attr ", attr_name,
                                                 " has value ", v,
                                                 " out of range for an int32");
                })
DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
// std::vector<bool> specialization does not have emplace_back until
// c++14, so we have to use push_back (see
// http://en.cppreference.com/w/cpp/container/vector/emplace_back)
DEFINE_GET_ATTR(bool, b, "bool", push_back, v, ;)
DEFINE_GET_ATTR(DataType, type, "type", emplace_back, static_cast<DataType>(v),
                ;)
DEFINE_GET_ATTR(TensorShapeProto, shape, "shape", emplace_back, v, ;)
DEFINE_GET_ATTR(TensorShape, shape, "shape", emplace_back, TensorShape(v),
                TF_RETURN_IF_ERROR(TensorShape::IsValidShape(v));)
DEFINE_GET_ATTR(PartialTensorShape, shape, "shape", emplace_back,
                PartialTensorShape(v),
                TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(v));)
DEFINE_GET_ATTR(Tensor, tensor, "tensor", emplace_back, t, Tensor t;
                if (!t.FromProto(v)) {
                  return errors::InvalidArgument(
                      "Attr ", attr_name, " has value ", v.ShortDebugString(),
                      " that can't be converted to a Tensor");
                })

#undef DEFINE_GET_ATTR

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   DataTypeVector* value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "list(type)"));
  for (const auto& v : attr_value->list().type()) {
    value->push_back(static_cast<DataType>(v));
  }
  return Status::OK();
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const TensorProto** value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "tensor"));
  *value = &attr_value->tensor();
  return Status::OK();
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const NameAttrList** value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "func"));
  *value = &attr_value->func();
  return Status::OK();
}

namespace {  // Helper for InOutTypesForNode().

Status AddArgToSig(const NodeDef& node_def, const OpDef::ArgDef& arg_def,
                   DataTypeVector* sig) {
  const int original_size = sig->size();
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "repeats" times.
    int32 repeats = -1;
    TF_RETURN_IF_ERROR(GetNodeAttr(node_def, arg_def.number_attr(), &repeats));
    if (repeats < 0) {
      return errors::InvalidArgument("Value for number_attr() ", repeats,
                                     " < 0");
    }

    if (!arg_def.type_attr().empty()) {
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node_def, arg_def.type_attr(), &dtype));
      for (int i = 0; i < repeats; ++i) {
        sig->push_back(dtype);
      }
    } else if (arg_def.type() != DT_INVALID) {
      for (int i = 0; i < repeats; ++i) {
        sig->push_back(arg_def.type());
      }
    } else {
      return errors::InvalidArgument("Missing type or type_attr field in ",
                                     arg_def.ShortDebugString());
    }
  } else if (!arg_def.type_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        AttrSlice(node_def).Find(arg_def.type_attr(), &attr_value));
    sig->push_back(attr_value->type());
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        AttrSlice(node_def).Find(arg_def.type_list_attr(), &attr_value));
    for (int dtype : attr_value->list().type()) {
      sig->push_back(static_cast<DataType>(dtype));
    }
  } else if (arg_def.type() != DT_INVALID) {
    sig->push_back(arg_def.type());
  } else {
    return errors::InvalidArgument("No type fields in ",
                                   arg_def.ShortDebugString());
  }
  if (arg_def.is_ref()) {
    // For all types that were added by this function call, make them refs.
    for (size_t i = original_size; i < sig->size(); ++i) {
      (*sig)[i] = MakeRefType((*sig)[i]);
    }
  }
  return Status::OK();
}

}  // namespace

Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs) {
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, inputs));
  }
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, outputs));
  }
  return Status::OK();
}

Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def) {
  if (node_def.op() != op_def.name()) {
    return errors::InvalidArgument("NodeDef op '", node_def.op(),
                                   "' does not match ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  bool seen_control = false;
  size_t num_inputs = 0;
  // TODO(josh11b): Unify the input field validation.
  for (const string& input : node_def.input()) {
    if (StringPiece(input).starts_with("^")) {
      seen_control = true;
      if (input.find(':') != string::npos) {
        return errors::InvalidArgument("Control input '", input,
                                       "' must not have ':' in NodeDef: ",
                                       SummarizeNodeDef(node_def));
      }
    } else if (seen_control) {
      return errors::InvalidArgument("Non-control input '", input,
                                     "' after control input in NodeDef: ",
                                     SummarizeNodeDef(node_def));
    } else {
      ++num_inputs;
    }
  }

  std::unordered_map<string, const OpDef::AttrDef*> op_attrs;
  for (const auto& attr : op_def.attr()) {
    if (!gtl::InsertIfNotPresent(&op_attrs, attr.name(), &attr)) {
      return errors::InvalidArgument("OpDef has duplicate attr name '",
                                     attr.name(), "': ",
                                     SummarizeOpDef(op_def));
    }
  }
  for (const auto& attr : node_def.attr()) {
    // Allow internal optional attributes with names starting with "_".
    if (StringPiece(attr.first).starts_with("_")) {
      continue;
    }
    auto iter = op_attrs.find(attr.first);
    if (iter == op_attrs.end()) {
      return errors::InvalidArgument("NodeDef mentions attr '", attr.first,
                                     "' not in ", SummarizeOpDef(op_def),
                                     "; NodeDef: ", SummarizeNodeDef(node_def));
    }
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ValidateAttrValue(attr.second, *iter->second), "; NodeDef: ",
        SummarizeNodeDef(node_def), "; ", SummarizeOpDef(op_def));
    // Keep track of which attr names have (not) been found in the NodeDef.
    op_attrs.erase(iter);
  }

  // Were all attrs in the OpDef found in the NodeDef?
  if (!op_attrs.empty()) {
    string attrs;
    for (const auto& attr_pair : op_attrs) {
      if (!attrs.empty()) strings::StrAppend(&attrs, "', '");
      strings::StrAppend(&attrs, attr_pair.first);
    }
    return errors::InvalidArgument("NodeDef missing attr",
                                   op_attrs.size() == 1 ? " '" : "s '", attrs,
                                   "' from ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  // Validate the number of inputs.
  DataTypeVector inputs, outputs;
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "NodeDef expected inputs '", DataTypeVectorString(inputs),
        "' do not match ", num_inputs, " inputs specified; ",
        SummarizeOpDef(op_def), "; NodeDef: ", SummarizeNodeDef(node_def));
  }

  return Status::OK();
}

namespace {  // Helpers for NameRangesForNode()

Status ComputeArgRange(const NodeDef& node_def, const OpDef::ArgDef& arg_def,
                       const OpDef& op_def, int* num) {
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    return GetNodeAttr(node_def, arg_def.number_attr(), num);
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        AttrSlice(node_def).Find(arg_def.type_list_attr(), &attr_value));
    *num = attr_value->list().type_size();
  } else if (!arg_def.type_attr().empty() || arg_def.type() != DT_INVALID) {
    *num = 1;
  } else {
    return errors::InvalidArgument("Argument '", arg_def.name(),
                                   "' incorrectly specified in op definition: ",
                                   SummarizeOpDef(op_def));
  }
  return Status::OK();
}

Status NameRangesHelper(const NodeDef& node_def,
                        const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                        const OpDef& op_def, NameRangeMap* result) {
  int start = 0;
  int num;
  for (const auto& arg : args) {
    TF_RETURN_IF_ERROR(ComputeArgRange(node_def, arg, op_def, &num));
    (*result)[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return Status::OK();
}

}  // namespace

Status NameRangesForNode(const NodeDef& node_def, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  TF_RETURN_IF_ERROR(
      NameRangesHelper(node_def, op_def.input_arg(), op_def, inputs));
  return NameRangesHelper(node_def, op_def.output_arg(), op_def, outputs);
}

void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def) {
  for (const auto& attr_def : op_def.attr()) {
    AttrSlice attrs(*node_def);
    if (attr_def.has_default_value() && !attrs.Find(attr_def.name())) {
      AddNodeAttr(attr_def.name(), attr_def.default_value(), node_def);
    }
  }
}

namespace {

using ::tensorflow::strings::Scanner;

bool IsValidOpName(StringPiece sp) {
  return Scanner(sp)
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
      .Eos()
      .GetResult();
}

bool IsValidDataInputName(StringPiece sp) {
  // Data inputs are op_name, op_name:0, or op_name:12345.
  Scanner scan(sp);
  scan.One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  if (scan.Peek() == ':') {
    scan.OneLiteral(":");
    if (scan.Peek() == '0') {
      scan.OneLiteral("0");  // :0
    } else {
      scan.Many(Scanner::DIGIT);  // :[1-9][0-9]*
    }
  }
  scan.Eos();

  return scan.GetResult();
}

bool IsValidControlInputName(StringPiece sp) {
  return Scanner(sp)
      .OneLiteral("^")
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
      .Eos()
      .GetResult();
}

}  // namespace

Status ValidateOpInput(const string& input_name, bool* is_control_input) {
  *is_control_input = false;
  if (IsValidDataInputName(input_name)) {
    return Status::OK();
  } else if (IsValidControlInputName(input_name)) {
    *is_control_input = true;
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op input name '", input_name, "'");
  }
}

Status ValidateOpName(const string& op_name) {
  if (IsValidOpName(op_name)) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op name '", op_name, "'");
  }
}

Status ValidateExternalNodeDefSyntax(const NodeDef& node_def) {
  Status s = ValidateOpName(node_def.name());
  if (!s.ok()) {
    return AttachDef(s, node_def);
  }
  bool in_control_inputs = false;
  for (const string& input_name : node_def.input()) {
    bool is_control_input;
    s = ValidateOpInput(input_name, &is_control_input);
    if (!s.ok()) {
      return AttachDef(s, node_def);
    }

    if (in_control_inputs && !is_control_input) {
      return AttachDef(errors::InvalidArgument(
                           "All control inputs must follow all data inputs"),
                       node_def);
    }
    in_control_inputs = is_control_input;
  }
  return Status::OK();
}

Status AttachDef(const Status& status, const NodeDef& node_def) {
  Status ret = status;
  errors::AppendToMessage(
      &ret, strings::StrCat(" [[Node: ", SummarizeNodeDef(node_def), "]]"));
  return ret;
}

}  // namespace tensorflow
