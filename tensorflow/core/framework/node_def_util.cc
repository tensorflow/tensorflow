/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

const char* const kColocationAttrName = "_class";
const char* const kColocationGroupPrefix = "loc:@";
// For TPU distributed rewrite, TPU args are collected and "staged" on the local
// host using an IdentityN TF op. Some args may result from a remote source.
// When all arg tensors are available, the TPUExecute op can be inovoked. See
// DistributedTPURewritePass for more details.
const char* const kTpuExecuteStagingOp = "IdentityN";
const char* const kTpuExecuteStagingNodeName = "_variable_copy";

AttrSlice::AttrSlice() : ndef_(nullptr) {
  static const AttrValueMap* const kEmptyAttrValueMap = new AttrValueMap;
  attrs_ = kEmptyAttrValueMap;
}

AttrSlice::AttrSlice(const NodeDef& node_def)
    : ndef_(&node_def), attrs_(&ndef_->attr()) {}

AttrSlice::AttrSlice(const AttrValueMap* a) : ndef_(nullptr), attrs_(a) {}

string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device) {
  string ret;

  // We sort the attrs so the output is deterministic.
  std::vector<string> attr_names;
  attr_names.reserve(attrs.size());
  for (const auto& attr : attrs) {
    attr_names.push_back(attr.first);
  }
  std::sort(attr_names.begin(), attr_names.end());
  bool first = true;
  for (const string& attr_name : attr_names) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, attr_name, "=",
                       SummarizeAttrValue(*attrs.Find(attr_name)));
  }

  // Consider the device to be a final attr with name "_device".
  if (!device.empty()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, "_device=\"", device, "\"");
  }
  return ret;
}

string AttrSlice::SummarizeNode() const {
  return ndef_ ? SummarizeNodeDef(*ndef_)
               : strings::StrCat(
                     "[", SummarizeAttrsHelper(*this, StringPiece()), "]");
}

string AttrSlice::DebugString() const {
  std::vector<string> attr_key_vals;
  attr_key_vals.reserve(attrs_->size());
  for (const auto& it : *this) {
    const string& name = it.first;
    const AttrValue& attr_value = it.second;
    attr_key_vals.push_back(
        absl::StrCat(name, "=", SummarizeAttrValue(attr_value)));
  }
  return absl::StrJoin(attr_key_vals, ", ");
}

string SummarizeNodeDef(const NodeDef& node_def, int max_inputs_in_summary) {
  string ret = strings::StrCat(errors::FormatNodeNameForError(node_def.name()),
                               " = ", node_def.op(), "[");
  strings::StrAppend(&ret, SummarizeAttrsHelper(node_def, node_def.device()));
  strings::StrAppend(&ret, "](");

  // Output inputs, including control inputs, verbatim.
  bool first = true;
  for (const string& input : node_def.input()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    if (max_inputs_in_summary-- == 0) {
      strings::StrAppend(&ret, "...");
      break;
    }
    strings::StrAppend(&ret, input);
  }
  strings::StrAppend(&ret, ")");
  return ret;
}

string SummarizeAttrs(const NodeDef& node_def) {
  return SummarizeAttrsHelper(node_def, node_def.device());
}

string FormatNodeDefForError(
    StringPiece node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info) {
  return !has_experimental_debug_info ||
                 experimental_debug_info.original_node_names().empty()
             ? errors::FormatNodeNameForError(string(node_name))
             : errors::FormatNodeNamesForError(
                   experimental_debug_info.original_node_names());
}

string FormatNodeDefForError(const NodeDef& node_def) {
  return FormatNodeDefForError(node_def.name(),
                               node_def.has_experimental_debug_info(),
                               node_def.experimental_debug_info());
}

const AttrValue* AttrSlice::Find(StringPiece attr_name) const {
  // Currently, the collection used for NodeDef::attr() (google::protobuf::Map)
  // requires that the keys used for lookups have type 'const string&'. Because
  // this method takes a StringPiece, it is necessary to allocate a temporary
  // string, copy attr_name to it, and then use that temporary string for the
  // lookup. This causes an excessive number of short-lived allocations, and for
  // large graphs, this can be a significant cost.
  //
  // Because most nodes have a small number of attributes, a simple linear scan
  // is generally more efficient than a hashed lookup.  If google::protobuf::Map
  // changes so that it supports efficient lookups using StringPiece instead of
  // const string&, then this code could be changed to use attrs_->find() again.

  for (const auto& attr : *attrs_) {
    if (attr.first == attr_name) {
      return &attr.second;
    }
  }
  return nullptr;
}

const AttrValue* AttrSlice::FindByString(const string& attr_name) const {
  auto iter = attrs_->find(attr_name);
  if (iter != attrs_->end()) {
    return &iter->second;
  } else {
    return nullptr;
  }
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
  if (!absl::StartsWith(attr_name, "_") && ndef_ != nullptr) {
    s = AttachDef(s, *ndef_);
  }
  return s;
}

bool AttrSlice::EqualAttrs(AttrSlice other, Scratch* scratch) const {
  if (size() != other.size()) return false;

  for (const auto& attr : *other.attrs_) {
    auto iter = attrs_->find(attr.first);
    if (iter == attrs_->end()) return false;
    // TODO(irving): Comparing AttrValues by proto is slightly buggy, since
    // TensorProto is a nonunique representation of Tensor.  This bug will go
    // away once AttrSlice switches over to NodeInfo.
    iter->second.SerializeToString(&scratch->a);
    attr.second.SerializeToString(&scratch->b);
    if (scratch->a != scratch->b) return false;
  }
  return true;
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
    value->reserve(attr_value->list().FIELD().size());                        \
    for (const auto& v : attr_value->list().FIELD()) {                        \
      __VA_ARGS__;                                                            \
      value->APPEND_OP(CAST);                                                 \
    }                                                                         \
    return Status::OK();                                                      \
  }

#define DEFINE_TRY_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...) \
  bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                      TYPE* value) {                                      \
    const AttrValue* attr_value = attrs.Find(attr_name);                  \
    if (attr_value == nullptr) {                                          \
      return false;                                                       \
    }                                                                     \
    Status s = AttrValueHasType(*attr_value, ATTR_TYPE);                  \
    if (!s.ok()) {                                                        \
      return false;                                                       \
    }                                                                     \
    const auto& v = attr_value->FIELD();                                  \
    __VA_ARGS__;                                                          \
    *value = CAST;                                                        \
    return true;                                                          \
  }                                                                       \
  bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                      std::vector<TYPE>* value) {                         \
    const AttrValue* attr_value = attrs.Find(attr_name);                  \
    if (attr_value == nullptr) {                                          \
      return false;                                                       \
    }                                                                     \
    Status s = AttrValueHasType(*attr_value, "list(" ATTR_TYPE ")");      \
    if (!s.ok()) {                                                        \
      return false;                                                       \
    }                                                                     \
    value->reserve(attr_value->list().FIELD().size());                    \
    for (const auto& v : attr_value->list().FIELD()) {                    \
      __VA_ARGS__;                                                        \
      value->APPEND_OP(CAST);                                             \
    }                                                                     \
    return true;                                                          \
  }
DEFINE_GET_ATTR(tstring, s, "string", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(tstring, s, "string", emplace_back, v, ;)
DEFINE_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_GET_ATTR(
    int32, i, "int", emplace_back, static_cast<int32>(v),
    if (static_cast<int64>(static_cast<int32>(v)) != v) {
      return errors::InvalidArgument("Attr ", attr_name, " has value ", v,
                                     " out of range for an int32");
    })
DEFINE_TRY_GET_ATTR(
    int32, i, "int", emplace_back, static_cast<int32>(v),
    if (static_cast<int64>(static_cast<int32>(v)) != v) {
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING) << "Attr " << attr_name << " has value " << v
                     << " out of range for an int32";
      }
      return false;
    })
DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(float, f, "float", emplace_back, v, ;)
// std::vector<bool> specialization does not have emplace_back until
// c++14, so we have to use push_back (see
// http://en.cppreference.com/w/cpp/container/vector/emplace_back)
DEFINE_GET_ATTR(bool, b, "bool", push_back, v, ;)
DEFINE_TRY_GET_ATTR(bool, b, "bool", push_back, v, ;)
DEFINE_GET_ATTR(DataType, type, "type", emplace_back, static_cast<DataType>(v),
                ;)
DEFINE_TRY_GET_ATTR(DataType, type, "type", emplace_back,
                    static_cast<DataType>(v),
                    ;)
DEFINE_GET_ATTR(TensorShapeProto, shape, "shape", emplace_back, v, ;)
DEFINE_GET_ATTR(TensorShape, shape, "shape", emplace_back, TensorShape(v),
                TF_RETURN_IF_ERROR(TensorShape::IsValidShape(v));)
DEFINE_TRY_GET_ATTR(
    TensorShape, shape, "shape", emplace_back, TensorShape(v),
    if (!TensorShape::IsValidShape(v).ok()) {
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING) << "Attr " << attr_name << " has invalid shape value "
                     << v.DebugString();
      }
      return false;
    })
DEFINE_GET_ATTR(PartialTensorShape, shape, "shape", emplace_back,
                PartialTensorShape(v),
                TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(v));)
DEFINE_GET_ATTR(
    Tensor, tensor, "tensor", emplace_back, t, Tensor t; if (!t.FromProto(v)) {
      return errors::InvalidArgument("Attr ", attr_name, " has value ",
                                     v.ShortDebugString(),
                                     " that can't be converted to a Tensor");
    })
DEFINE_GET_ATTR(NameAttrList, func, "func", emplace_back, v, ;);
#undef DEFINE_GET_ATTR

bool HasNodeAttr(const NodeDef& node_def, StringPiece attr_name) {
  return node_def.attr().find(string(attr_name)) != node_def.attr().end();
}

static const string& kEmptyString = *new string();

const string& GetNodeAttrString(const AttrSlice& attrs, StringPiece attr_name) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return kEmptyString;
  }
  Status s = AttrValueHasType(*attr_value, "string");
  if (!s.ok()) {
    return kEmptyString;
  }
  return attr_value->s();
}

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<const string*>* value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "list(string)");
  if (!s.ok()) {
    return false;
  }
  value->reserve(attr_value->list().s().size());
  for (const auto& v : attr_value->list().s()) {
    value->push_back(&v);
  }
  return true;
}

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<const TensorShapeProto*>* value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "list(shape)");
  if (!s.ok()) {
    return false;
  }
  value->reserve(attr_value->list().shape().size());
  for (const auto& v : attr_value->list().shape()) {
    value->push_back(&v);
  }
  return true;
}

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

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const TensorProto** value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "tensor");
  if (!s.ok()) {
    return false;
  }
  *value = &attr_value->tensor();
  return true;
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const NameAttrList** value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "func"));
  *value = &attr_value->func();
  return Status::OK();
}

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const NameAttrList** value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "func");
  if (!s.ok()) {
    return false;
  }
  *value = &attr_value->func();
  return true;
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   Padding* value) {
  string str_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_name, &str_value));
  return GetPaddingFromString(str_value, value);
}

namespace {  // Helper for InOutTypesForNode().

template <class NodeDefOrAttrSlice>
Status AddArgToSig(const NodeDefOrAttrSlice& node_or_attrs,
                   const OpDef::ArgDef& arg_def, DataTypeVector* sig) {
  const int original_size = sig->size();
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "repeats" times.
    int64 repeats = -1;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(node_or_attrs, arg_def.number_attr(), &repeats));
    // We can't handle outputs that are larger than int32 sizes.
    if (static_cast<int64>(static_cast<int32>(repeats)) != repeats) {
      return errors::InvalidArgument("Number of outputs is too big: ", repeats);
    }
    if (repeats < 0) {
      return errors::InvalidArgument("Value for number_attr() ", repeats,
                                     " < 0");
    }

    if (!arg_def.type_attr().empty()) {
      DataType dtype;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(node_or_attrs, arg_def.type_attr(), &dtype));
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
        AttrSlice(node_or_attrs).Find(arg_def.type_attr(), &attr_value));
    sig->push_back(attr_value->type());
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        AttrSlice(node_or_attrs).Find(arg_def.type_list_attr(), &attr_value));
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
      if (IsRefType((*sig)[i])) {
        return errors::InvalidArgument(
            "Requested reference to a reference type: ",
            arg_def.ShortDebugString());
      }
      (*sig)[i] = MakeRefType((*sig)[i]);
    }
  }
  return Status::OK();
}

}  // namespace

Status InputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                        int input_port, DataType* input_type) {
  DataTypeVector input_types;
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, &input_types));
    int input_types_size = input_types.size();
    if (input_types_size > input_port) {
      const DataType dtype = input_types[input_port];
      *input_type = dtype;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Input ", input_port, " not found for node ",
                                 node_def.name());
}

Status InputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs) {
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, inputs));
  }
  return Status::OK();
}

Status OutputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                         int output_port, DataType* output_type) {
  DataTypeVector output_types;
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, &output_types));
    int output_types_size = output_types.size();
    if (output_types_size > output_port) {
      const DataType dtype = output_types[output_port];
      *output_type = dtype;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Output ", output_port, " not found for node ",
                                 node_def.name());
}

Status OutputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                          DataTypeVector* outputs) {
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, outputs));
  }
  return Status::OK();
}

Status OutputTypesForNode(const AttrSlice& attrs, const OpDef& op_def,
                          DataTypeVector* outputs) {
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(attrs, arg, outputs));
  }
  return Status::OK();
}

Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs) {
  TF_RETURN_IF_ERROR(InputTypesForNode(node_def, op_def, inputs));
  return OutputTypesForNode(node_def, op_def, outputs);
}

Status NumOutputsForNode(const NodeDef& node_def, const OpDef& op_def,
                         int* num_outputs) {
  DataTypeVector outputs;
  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, op_def, &outputs));
  *num_outputs = outputs.size();
  return Status::OK();
}

Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def) {
  if (node_def.op() != op_def.name()) {
    return errors::InvalidArgument(
        "NodeDef op '", node_def.op(), "' does not match ",
        SummarizeOpDef(op_def), "; NodeDef: ", FormatNodeDefForError(node_def));
  }

  bool seen_control = false;
  size_t num_inputs = 0;
  // TODO(josh11b): Unify the input field validation.
  for (const string& input : node_def.input()) {
    if (absl::StartsWith(input, "^")) {
      seen_control = true;
      if (input.find(':') != string::npos) {
        return errors::InvalidArgument("Control input '", input,
                                       "' must not have ':' in NodeDef: ",
                                       FormatNodeDefForError(node_def));
      }
    } else if (seen_control) {
      return errors::InvalidArgument("Non-control input '", input,
                                     "' after control input in NodeDef: ",
                                     FormatNodeDefForError(node_def));
    } else {
      ++num_inputs;
    }
  }

  std::unordered_map<string, const OpDef::AttrDef*> op_attrs;
  for (const auto& attr : op_def.attr()) {
    if (!gtl::InsertIfNotPresent(&op_attrs, attr.name(), &attr)) {
      return errors::InvalidArgument("OpDef has duplicate attr name '",
                                     attr.name(),
                                     "': ", SummarizeOpDef(op_def));
    }
  }
  for (const auto& attr : node_def.attr()) {
    // Allow internal optional attributes with names starting with "_".
    if (absl::StartsWith(attr.first, "_")) {
      continue;
    }
    auto iter = op_attrs.find(attr.first);
    if (iter == op_attrs.end()) {
      LOG_EVERY_N_SEC(ERROR, 5)
          << "NodeDef mentions attribute " << attr.first
          << " which is not in the op definition: " << SummarizeOpDef(op_def)
          << " This may be expected if your graph generating binary is newer "
          << " than this binary. Unknown attributes will be ignored."
          << " NodeDef: " << FormatNodeDefForError(node_def);
      continue;
    }

    // If attr value is placeholder, do not check it.
    if (attr.second.placeholder().empty()) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ValidateAttrValue(attr.second, *iter->second),
          "; NodeDef: ", FormatNodeDefForError(node_def), "; ",
          SummarizeOpDef(op_def));
    }

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
    return errors::InvalidArgument(
        "NodeDef missing attr", op_attrs.size() == 1 ? " '" : "s '", attrs,
        "' from ", SummarizeOpDef(op_def),
        "; NodeDef: ", FormatNodeDefForError(node_def));
  }

  // Validate the number of inputs.
  DataTypeVector inputs, outputs;
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "NodeDef expected inputs '", DataTypeVectorString(inputs),
        "' do not match ", num_inputs, " inputs specified; ",
        SummarizeOpDef(op_def), "; NodeDef: ", FormatNodeDefForError(node_def));
  }

  return Status::OK();
}

namespace {  // Helpers for NameRangesForNode()

Status ComputeArgRange(const AttrSlice& attrs, const OpDef::ArgDef& arg_def,
                       const OpDef& op_def, int* num) {
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    return GetNodeAttr(attrs, arg_def.number_attr(), num);
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(attrs.Find(arg_def.type_list_attr(), &attr_value));
    *num = attr_value->list().type_size();
  } else if (!arg_def.type_attr().empty() || arg_def.type() != DT_INVALID) {
    *num = 1;
  } else {
    return errors::InvalidArgument(
        "Argument '", arg_def.name(),
        "' incorrectly specified in op definition: ", SummarizeOpDef(op_def));
  }
  return Status::OK();
}

Status NameRangesHelper(const AttrSlice& attrs,
                        const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                        const OpDef& op_def, NameRangeMap* result) {
  int start = 0;
  int num;
  for (const auto& arg : args) {
    TF_RETURN_IF_ERROR(ComputeArgRange(attrs, arg, op_def, &num));
    (*result)[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return Status::OK();
}

}  // namespace

Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  if (inputs != nullptr) {
    TF_RETURN_IF_ERROR(
        NameRangesHelper(attrs, op_def.input_arg(), op_def, inputs));
  }
  if (outputs != nullptr) {
    return NameRangesHelper(attrs, op_def.output_arg(), op_def, outputs);
  }
  return Status::OK();
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

using ::tensorflow::tstring;
using ::tensorflow::strings::Scanner;

bool IsValidNodeName(StringPiece sp) {
  Scanner scanner(sp);
  scanner.One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scanner.GetResult())  // Some error in previous iteration.
      return false;
    if (scanner.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another name/namespace, starting with a '>'
    scanner.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}

bool IsValidDataInputName(StringPiece sp) {
  // Data inputs are op_name, op_name:0, or op_name:12345.
  Scanner scan(sp);
  scan.One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scan.GetResult())  // Some error in previous iteration.
      return false;
    if (scan.empty())  // No error, but nothing left, good.
      return true;

    if (scan.Peek() == ':') {  // Absorb identifier after the colon
      scan.OneLiteral(":");
      if (scan.Peek() == '0') {
        scan.OneLiteral("0");  // :0
      } else {
        scan.Many(Scanner::DIGIT);  // :[1-9][0-9]*
      }
    } else {
      // Absorb another name/namespace, starting with a '>'
      scan.One(Scanner::RANGLE)
          .One(Scanner::LETTER_DIGIT_DOT)
          .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
    }
  }
}

bool IsValidControlInputName(StringPiece sp) {
  Scanner scan(sp);
  scan.OneLiteral("^")
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scan.GetResult())  // Some error in previous iteration.
      return false;
    if (scan.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another name/namespace, starting with a '>'
    scan.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}

const StringPiece kColocationGroupPrefixStringPiece(kColocationGroupPrefix);

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

Status ValidateNodeName(const string& node_name) {
  if (IsValidNodeName(node_name)) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op name '", node_name, "'");
  }
}

Status ValidateExternalNodeDefSyntax(const NodeDef& node_def) {
  Status s = ValidateNodeName(node_def.name());
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

Status AttachDef(const Status& status, const NodeDef& node_def,
                 bool allow_multiple_formatted_node) {
  Status ret = status;
  string node_error;
  if (!allow_multiple_formatted_node &&
      status.error_message().find("{{node ") != string::npos) {
    node_error = node_def.name();
  } else {
    node_error = FormatNodeDefForError(node_def);
  }
  errors::AppendToMessage(&ret, strings::StrCat(" [[", node_error, "]]"));
  return ret;
}

void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def) {
  node_def->mutable_attr()->insert(
      AttrValueMap::value_type(string(name), value));
}

void AddNodeAttr(StringPiece name, AttrValue&& value, NodeDef* node_def) {
  (*node_def->mutable_attr())[string(name)] = std::move(value);
}

#define ADD_NODE_ATTR(T)                                           \
  void AddNodeAttr(StringPiece name, T value, NodeDef* node_def) { \
    AttrValue attr_value;                                          \
    SetAttrValue(value, &attr_value);                              \
    AddNodeAttr(name, attr_value, node_def);                       \
  }
ADD_NODE_ATTR(StringPiece)
ADD_NODE_ATTR(const char*)
ADD_NODE_ATTR(int32)
ADD_NODE_ATTR(int64)
ADD_NODE_ATTR(float)
ADD_NODE_ATTR(double)
ADD_NODE_ATTR(bool)
ADD_NODE_ATTR(DataType)
ADD_NODE_ATTR(const PartialTensorShape&)
ADD_NODE_ATTR(const Tensor&)
ADD_NODE_ATTR(const TensorProto&)
ADD_NODE_ATTR(const NameAttrList&)
ADD_NODE_ATTR(gtl::ArraySlice<StringPiece>)
ADD_NODE_ATTR(gtl::ArraySlice<const char*>)
ADD_NODE_ATTR(gtl::ArraySlice<string>)
ADD_NODE_ATTR(gtl::ArraySlice<int32>)
ADD_NODE_ATTR(gtl::ArraySlice<int64>)
ADD_NODE_ATTR(gtl::ArraySlice<float>)
ADD_NODE_ATTR(gtl::ArraySlice<bool>)
ADD_NODE_ATTR(const std::vector<bool>&)
ADD_NODE_ATTR(gtl::ArraySlice<DataType>)
ADD_NODE_ATTR(gtl::ArraySlice<TensorShape>)
ADD_NODE_ATTR(gtl::ArraySlice<PartialTensorShape>)
ADD_NODE_ATTR(gtl::ArraySlice<TensorShapeProto>)
ADD_NODE_ATTR(gtl::ArraySlice<Tensor>)
ADD_NODE_ATTR(gtl::ArraySlice<NameAttrList>)
#undef ADD_NODE_ATTR

void AddAttr(StringPiece name, const AttrValue& value, AttrValueMap* map) {
  map->insert(AttrValueMap::value_type(string(name), value));
}

#define ADD_ATTR(T)                                            \
  void AddAttr(StringPiece name, T value, AttrValueMap* map) { \
    AttrValue attr_value;                                      \
    SetAttrValue(value, &attr_value);                          \
    AddAttr(name, attr_value, map);                            \
  }
ADD_ATTR(bool)
#undef ADD_ATTR

Status AddPrefixAndSuffixToNode(StringPiece prefix, StringPiece suffix,
                                NodeDef* node_def, bool uniquify_frame_name) {
  node_def->set_name(strings::StrCat(prefix, node_def->name(), suffix));

  // Update frame name to avoid multiple LoopCond nodes in one frame.
  if (uniquify_frame_name &&
      (node_def->op() == "Enter" || node_def->op() == "RefEnter")) {
    string frame_name;
    TF_RETURN_IF_ERROR(GetNodeAttr(*node_def, "frame_name", &frame_name));
    AttrValue& attr = (*node_def->mutable_attr())["frame_name"];
    frame_name = strings::StrCat(prefix, frame_name, suffix);
    attr.set_s(frame_name);
  }

  return Status::OK();
}

Status MaybeAddPrefixToColocationConstraints(
    const std::unordered_set<string>& match, StringPiece prefix,
    NodeDef* node_def) {
  auto attr = node_def->mutable_attr()->find(kColocationAttrName);
  if (attr == node_def->mutable_attr()->end()) {
    return Status::OK();
  }
  auto constraints_list = attr->second.mutable_list();
  auto constraints_size = constraints_list->s_size();
  for (size_t i = 0; i < constraints_size; ++i) {
    StringPiece original(constraints_list->s(i));
    if (absl::ConsumePrefix(&original, kColocationGroupPrefixStringPiece)) {
      if (match.find(string(original)) != match.end()) {
        (*constraints_list->mutable_s(i)) =
            strings::StrCat(kColocationGroupPrefix, prefix, original);
      }
    }
  }
  return Status::OK();
}

Status MaybeUpdateColocationConstraintsWithMap(
    const std::map<absl::string_view, absl::string_view>& node_name_map,
    NodeDef* node_def) {
  auto attr = node_def->mutable_attr()->find(kColocationAttrName);
  if (attr == node_def->mutable_attr()->end()) {
    return Status::OK();
  }
  auto constraints_list = attr->second.mutable_list();
  auto constraints_size = constraints_list->s_size();
  for (size_t i = 0; i < constraints_size; ++i) {
    StringPiece original(constraints_list->s(i));
    if (absl::ConsumePrefix(&original, kColocationGroupPrefixStringPiece)) {
      if (node_name_map.find(original) != node_name_map.end()) {
        (*constraints_list->mutable_s(i)) =
            strings::StrCat(kColocationGroupPrefix, node_name_map.at(original));
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
