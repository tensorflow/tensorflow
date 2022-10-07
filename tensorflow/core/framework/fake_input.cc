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

#include "tensorflow/core/framework/fake_input.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

class FakeInputImpl {
 public:
  FakeInputImpl(const OpDef* op_def, int in_index, const NodeDef* node_def,
                NodeDefBuilder* builder);
  void SetN(int n);
  void SetDataType(DataType dt);
  void SetTypeList(DataTypeSlice dts);
  Status AddInputToBuilder();

 private:
  static string FakeNodeName(int in_index);
  Status GetN(int* n) const;
  Status GetDataType(DataType* dt) const;
  void NSources(int n, DataType dt) const;
  void SourceList(DataTypeSlice dts) const;

  const OpDef* const op_def_;
  const OpDef::ArgDef* const arg_;
  const string in_node_;
  const NodeDef* const node_def_;
  NodeDefBuilder* const builder_;

  bool n_specified_;
  int n_;
  bool dt_specified_;
  DataType dt_;
  bool dts_specified_;
  DataTypeSlice dts_;
};

FakeInputImpl::FakeInputImpl(const OpDef* op_def, int in_index,
                             const NodeDef* node_def, NodeDefBuilder* builder)
    : op_def_(op_def),
      arg_(&op_def->input_arg(in_index)),
      in_node_(FakeNodeName(in_index)),
      node_def_(node_def),
      builder_(builder),
      n_specified_(false),
      dt_specified_(false),
      dts_specified_(false) {}

void FakeInputImpl::SetN(int n) {
  n_specified_ = true;
  n_ = n;
}

void FakeInputImpl::SetDataType(DataType dt) {
  dt_specified_ = true;
  dt_ = dt;
}

void FakeInputImpl::SetTypeList(DataTypeSlice dts) {
  dts_specified_ = true;
  dts_ = dts;
}

Status FakeInputImpl::AddInputToBuilder() {
  if (dts_specified_) {
    SourceList(dts_);

  } else if (n_specified_ || !arg_->number_attr().empty()) {
    int n;
    TF_RETURN_IF_ERROR(GetN(&n));

    DataType dt;
    if (n > 0) {
      TF_RETURN_IF_ERROR(GetDataType(&dt));
    } else {
      dt = DT_FLOAT;
    }

    NSources(n, dt);
  } else {
    if (!dt_specified_ && !arg_->type_list_attr().empty()) {
      DataTypeVector dts;
      Status status = GetNodeAttr(*node_def_, arg_->type_list_attr(), &dts);
      if (!status.ok()) {
        return errors::InvalidArgument(
            "Could not infer list of types for input '", arg_->name(),
            "': ", status.error_message());
      }
      SourceList(dts);
      return OkStatus();
    }

    DataType dt;
    TF_RETURN_IF_ERROR(GetDataType(&dt));
    builder_->Input(in_node_, 0, dt);
  }
  return OkStatus();
}

// static
string FakeInputImpl::FakeNodeName(int in_index) {
  char c = 'a' + (in_index % 26);
  return string(&c, 1);
}

Status FakeInputImpl::GetN(int* n) const {
  if (n_specified_) {
    *n = n_;
  } else {
    Status status = GetNodeAttr(*node_def_, arg_->number_attr(), n);
    if (!status.ok()) {
      return errors::InvalidArgument("Could not infer length of input '",
                                     arg_->name(),
                                     "': ", status.error_message());
    }
  }
  return OkStatus();
}

Status FakeInputImpl::GetDataType(DataType* dt) const {
  if (dt_specified_) {
    *dt = dt_;
    return OkStatus();  // Ignore is_ref field of arg_.
  } else if (arg_->type() != DT_INVALID) {
    *dt = arg_->type();
  } else if (!arg_->type_attr().empty()) {
    Status status = GetNodeAttr(*node_def_, arg_->type_attr(), dt);
    if (!status.ok()) {
      // Check if the type attr has a default
      const OpDef::AttrDef* attr = FindAttr(arg_->type_attr(), *op_def_);
      if (attr && attr->has_default_value()) {
        *dt = attr->default_value().type();
      } else {
        return errors::InvalidArgument("Could not infer type for input '",
                                       arg_->name(),
                                       "': ", status.error_message());
      }
    }
  } else {
    return errors::InvalidArgument("No type or type_attr field in arg '",
                                   arg_->name(), "'");
  }
  if (arg_->is_ref()) {
    *dt = MakeRefType(*dt);
  }
  return OkStatus();
}

void FakeInputImpl::NSources(int n, DataType dt) const {
  std::vector<NodeDefBuilder::NodeOut> srcs;
  srcs.reserve(n);
  for (int i = 0; i < n; ++i) {
    srcs.emplace_back(in_node_, i, dt);
  }
  builder_->Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
}

void FakeInputImpl::SourceList(DataTypeSlice dts) const {
  std::vector<NodeDefBuilder::NodeOut> srcs;
  srcs.reserve(dts.size());
  for (size_t i = 0; i < dts.size(); ++i) {
    srcs.emplace_back(in_node_, i, dts[i]);
  }
  builder_->Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
}

}  // namespace

// Public interface ------------------------------------------------------------

FakeInputFunctor FakeInput() {
  return [](const OpDef& op_def, int in_index, const NodeDef& node_def,
            NodeDefBuilder* builder) {
    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) {
    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetDataType(dt);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(int n) {
  return [n](const OpDef& op_def, int in_index, const NodeDef& node_def,
             NodeDefBuilder* builder) {
    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetN(n);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(int n, DataType dt) {
  return [n, dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
                 NodeDefBuilder* builder) {
    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetN(n);
    impl.SetDataType(dt);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(DataTypeSlice dts) {
  // Make a copy to ensure the data will still be around when the lambda is
  // called.
  DataTypeVector dtv(dts.begin(), dts.end());
  return [dtv](const OpDef& op_def, int in_index, const NodeDef& node_def,
               NodeDefBuilder* builder) {
    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetTypeList(dtv);
    return impl.AddInputToBuilder();
  };
}

}  // namespace tensorflow
