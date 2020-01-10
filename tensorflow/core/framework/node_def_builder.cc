#include "tensorflow/core/framework/node_def_builder.h"

#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

NodeDefBuilder::NodeDefBuilder(const string& name, const string& op_name,
                               const OpRegistryInterface* op_registry) {
  node_def_.set_name(name);
  Status status;
  op_def_ = op_registry->LookUp(op_name, &status);
  if (op_def_ == nullptr) {
    errors_.push_back(status.error_message());
    inputs_specified_ = 0;
  } else {
    Initialize();
  }
}

NodeDefBuilder::NodeDefBuilder(const string& name, const OpDef* op_def)
    : op_def_(op_def) {
  node_def_.set_name(name);
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
    Status status =
        fake_input(*op_def_, inputs_specified_, node_def_, this);
    if (!status.ok()) errors_.push_back(status.error_message());
  }
  return *this;
}

void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg,
                                 const string& src_node, int src_index,
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
                               gtl::ArraySlice<NodeOut> src_list) {
  for (const auto& node_out : src_list) {
    AddInput(node_out.node, node_out.index);
  }

  if (!input_arg->number_attr().empty()) {
    Attr(input_arg->number_attr(), static_cast<int64>(src_list.size()));
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

void NodeDefBuilder::AddInput(const string& src_node, int src_index) {
  if (src_node.empty()) {
    errors_.push_back("Empty input node name");
  } else if (src_node[0] == '^') {
    errors_.push_back(
        strings::StrCat("Non-control input starting with ^: ", src_node));
  } else if (src_index > 0) {
    node_def_.add_input(strings::StrCat(src_node, ":", src_index));
  } else {
    node_def_.add_input(src_node);
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

Status NodeDefBuilder::Finalize(NodeDef* node_def) const {
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
      return errors::InvalidArgument(
          errors_ptr->size(), " errors while building NodeDef '",
          node_def_.name(), "' using ", SummarizeOpDef(*op_def_), ":\n",
          str_util::Join(*errors_ptr, "\n"));
    }
  } else {
    NodeDef node_def_backup;
    if (node_def == nullptr) node_def = &node_def_backup;
    *node_def = node_def_;

    // Add control inputs after the regular inputs.
    for (const auto& control_input : control_inputs_) {
      node_def->add_input(strings::StrCat("^", control_input));
    }

    // Add default values for unspecified attrs.
    AddDefaultsToNodeDef(*op_def_, node_def);

    return Status::OK();
  }
}

}  // namespace tensorflow
