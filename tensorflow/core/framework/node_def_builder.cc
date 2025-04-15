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

#include "tensorflow/core/framework/node_def_builder.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

NodeDefBuilder::NodeOut::NodeOut(absl::string_view n, int i, DataType dt)
    : node(n), index(i), data_type(dt) {}

NodeDefBuilder::NodeOut::NodeOut() {
  // uninitialized, call Reset() before use.
}

void NodeDefBuilder::NodeOut::Reset(absl::string_view n, int i, DataType dt) {
  node = string(n);
  index = i;
  data_type = dt;
}

NodeDefBuilder::NodeDefBuilder(absl::string_view name,
                               absl::string_view op_name,
                               const OpRegistryInterface* op_registry,
                               const NodeDebugInfo* debug) {
  node_def_.set_name(string(name));
  const absl::Status status =
      op_registry->LookUpOpDef(string(op_name), &op_def_);
  if (status.ok()) {
    Initialize();
  } else {
    errors_.push_back(std::string(status.message()));
    inputs_specified_ = 0;
  }
  if (debug != nullptr) MergeDebugInfo(*debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(absl::string_view name,
                               absl::string_view op_name,
                               const NodeDebugInfo& debug)
    : NodeDefBuilder(name, op_name) {
  MergeDebugInfo(debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(absl::string_view name, const OpDef* op_def)
    : op_def_(op_def) {
  node_def_.set_name(string(name));
  Initialize();
}

void NodeDefBuilder::Initialize() {
  inputs_specified_ = 0;
  node_def_.set_op(op_def_->name());
}

const OpDef::ArgDef* NodeDefBuilder::NextArgDef() {
  if (!NextArgAvailable()) return nullptr;
  return &op_def_->input_arg(inputs_specified_++);
}

bool NodeDefBuilder::NextArgAvailable() {
  if (op_def_ == nullptr) {
    return false;
  } else if (inputs_specified_ >= op_def_->input_arg_size()) {
    errors_.push_back(strings::StrCat("More Input() calls than the ",
                                      op_def_->input_arg_size(),
                                      " input_args"));
    return false;
  }
  return true;
}

NodeDefBuilder& NodeDefBuilder::Input(FakeInputFunctor fake_input) {
  if (NextArgAvailable()) {
    absl::Status status =
        fake_input(*op_def_, inputs_specified_, node_def_, this);
    if (!status.ok()) errors_.push_back(std::string(status.message()));
  }
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Input(absl::string_view src_node, int src_index,
                                      DataType dt) {
  const OpDef::ArgDef* arg = NextArgDef();
  if (arg != nullptr) SingleInput(arg, src_node, src_index, dt);
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Input(const NodeOut& src) {
  Input(src.node, src.index, src.data_type);
  return *this;
}

// For inputs that take a list of tensors.
NodeDefBuilder& NodeDefBuilder::Input(absl::Span<const NodeOut> src_list) {
  const OpDef::ArgDef* arg = NextArgDef();
  if (arg != nullptr) ListInput(arg, src_list);
  return *this;
}

void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg,
                                 absl::string_view src_node, int src_index,
                                 DataType dt) {
  AddInput(src_node, src_index);

  if (!input_arg->number_attr().empty() ||
      !input_arg->type_list_attr().empty()) {
    errors_.push_back(strings::StrCat("Single tensor passed to '",
                                      input_arg->name(), "', expected list"));
    return;
  }

  if (input_arg->type() != DT_INVALID) {
    const DataType expected = MaybeAddRef(input_arg, input_arg->type());
    VerifyInputType(input_arg, expected, dt);
  } else {
    VerifyInputRef(input_arg, dt);
    Attr(input_arg->type_attr(), BaseType(dt));
  }
}

void NodeDefBuilder::ListInput(const OpDef::ArgDef* input_arg,
                               absl::Span<const NodeOut> src_list) {
  for (const auto& node_out : src_list) {
    AddInput(node_out.node, node_out.index);
  }

  if (!input_arg->number_attr().empty()) {
    Attr(input_arg->number_attr(), static_cast<int64_t>(src_list.size()));
    if (input_arg->type() != DT_INVALID) {
      const DataType expected = MaybeAddRef(input_arg, input_arg->type());
      for (const auto& node_out : src_list) {
        VerifyInputType(input_arg, expected, node_out.data_type);
      }
    } else if (!src_list.empty()) {
      const DataType base = BaseType(src_list[0].data_type);
      Attr(input_arg->type_attr(), base);
      const DataType expected = MaybeAddRef(input_arg, base);
      for (const auto& node_out : src_list) {
        VerifyInputType(input_arg, expected, node_out.data_type);
      }
    }
  } else if (!input_arg->type_list_attr().empty()) {
    DataTypeVector type_vec;
    type_vec.reserve(src_list.size());
    for (const auto& node_out : src_list) {
      const DataType dt = node_out.data_type;
      VerifyInputRef(input_arg, dt);
      type_vec.push_back(BaseType(dt));
    }
    Attr(input_arg->type_list_attr(), type_vec);
  } else {
    errors_.push_back(strings::StrCat("List provided to input '",
                                      input_arg->name(),
                                      "' when single Tensor expected"));
  }
}

void NodeDefBuilder::AddInput(absl::string_view src_node, int src_index) {
  if (src_node.empty()) {
    errors_.push_back("Empty input node name");
  } else if (src_node[0] == '^') {
    errors_.push_back(
        strings::StrCat("Non-control input starting with ^: ", src_node));
  } else if (src_index > 0) {
    node_def_.add_input(strings::StrCat(src_node, ":", src_index));
  } else {
    node_def_.add_input(string(src_node));
  }
}

void NodeDefBuilder::VerifyInputType(const OpDef::ArgDef* input_arg,
                                     DataType expected, DataType dt) {
  if (!TypesCompatible(expected, dt)) {
    errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                                      DataTypeString(dt), " expected ",
                                      DataTypeString(expected)));
  }
}

void NodeDefBuilder::VerifyInputRef(const OpDef::ArgDef* input_arg,
                                    DataType dt) {
  if (input_arg->is_ref() && !IsRefType(dt)) {
    errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                                      DataTypeString(dt),
                                      " expected ref type"));
  }
}

NodeDefBuilder& NodeDefBuilder::ControlInput(absl::string_view src_node) {
  control_inputs_.emplace_back(src_node);
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Device(absl::string_view device_spec) {
  node_def_.set_device(string(device_spec));
  return *this;
}

absl::Status NodeDefBuilder::Finalize(NodeDef* node_def, bool consume) {
  const std::vector<string>* errors_ptr = &errors_;
  std::vector<string> errors_storage;
  if (op_def_ != nullptr && inputs_specified_ < op_def_->input_arg_size()) {
    // Since this is a const method, to add an error, we have to make
    // a copy of the existing errors.
    errors_storage = errors_;
    errors_storage.push_back(
        strings::StrCat(inputs_specified_, " inputs specified of ",
                        op_def_->input_arg_size(), " inputs in Op"));
    errors_ptr = &errors_storage;
  }

  if (!errors_ptr->empty()) {
    if (errors_ptr->size() == 1) {
      if (op_def_ == nullptr) {
        return errors::InvalidArgument((*errors_ptr)[0],
                                       " while building NodeDef '",
                                       node_def_.name(), "'");
      }
      return errors::InvalidArgument(
          (*errors_ptr)[0], " while building NodeDef '", node_def_.name(),
          "' using ", SummarizeOpDef(*op_def_));
    } else {
      if (op_def_ == nullptr) {
        return errors::InvalidArgument(
            errors_ptr->size(), " errors while building NodeDef '",
            node_def_.name(), "':\n", absl::StrJoin(*errors_ptr, "\n"));
      }
      return errors::InvalidArgument(
          errors_ptr->size(), " errors while building NodeDef '",
          node_def_.name(), "' using ", SummarizeOpDef(*op_def_), ":\n",
          absl::StrJoin(*errors_ptr, "\n"));
    }
  } else {
    NodeDef node_def_backup;
    if (node_def == nullptr) node_def = &node_def_backup;
    if (consume) {
      *node_def = std::move(node_def_);
    } else {
      *node_def = node_def_;
    }

    // Add control inputs after the regular inputs.
    for (const auto& control_input : control_inputs_) {
      node_def->add_input(strings::StrCat("^", control_input));
    }

    // Add default values for unspecified attrs.
    AddDefaultsToNodeDef(*op_def_, node_def);

    return absl::OkStatus();
  }
}

bool NodeDefBuilder::AttrValueAlreadyPresent(absl::string_view name,
                                             const AttrValue& value) {
  if (const AttrValue* found = AttrSlice(node_def_).Find(name)) {
    if (!AreAttrValuesEqual(*found, value)) {
      errors_.push_back(strings::StrCat("Inconsistent values for attr '", name,
                                        "' ", SummarizeAttrValue(*found),
                                        " vs. ", SummarizeAttrValue(value)));
    }
    return true;
  }
  return false;
}

NodeDefBuilder& NodeDefBuilder::Attr(absl::string_view name,
                                     const AttrValue& value) {
  if (!AttrValueAlreadyPresent(name, value)) {
    AddNodeAttr(name, value, &node_def_);
  }
  return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(absl::string_view name,
                                     AttrValue&& value) {
  if (!AttrValueAlreadyPresent(name, value)) {
    AddNodeAttr(name, std::move(value), &node_def_);
  }
  return *this;
}

#define ATTR(T)                                                     \
  NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, T value) { \
    AttrValue attr_value;                                           \
    SetAttrValue(value, &attr_value);                               \
    return Attr(name, attr_value);                                  \
  }
ATTR(absl::string_view)
ATTR(const char*)
ATTR(int32_t)
ATTR(int64_t)
ATTR(float)
ATTR(double)
ATTR(bool)
ATTR(DataType)
ATTR(const PartialTensorShape&)
ATTR(const Tensor&)
ATTR(const TensorProto&)
ATTR(const NameAttrList&)
ATTR(absl::Span<const absl::string_view>)
ATTR(absl::Span<const char* const>)
ATTR(absl::Span<const string>)
ATTR(absl::Span<const tstring>)
ATTR(absl::Span<const int32>)
ATTR(absl::Span<const int64_t>)
ATTR(absl::Span<const float>)
ATTR(absl::Span<const bool>)
ATTR(const std::vector<bool>&)
ATTR(absl::Span<const DataType>)
ATTR(absl::Span<const TensorShape>)
ATTR(absl::Span<const PartialTensorShape>)
ATTR(absl::Span<const TensorShapeProto>)
ATTR(absl::Span<const Tensor>)
ATTR(absl::Span<const NameAttrList>)
#undef ATTR

}  // namespace tensorflow
