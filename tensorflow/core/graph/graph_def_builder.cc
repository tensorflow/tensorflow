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

#include "tensorflow/core/graph/graph_def_builder.h"

#include <utility>

#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

GraphDefBuilder::Options::Options(Graph* graph, Status* status)
    : graph_(graph), status_(status) {}
GraphDefBuilder::Options::~Options() {}

GraphDefBuilder::Options GraphDefBuilder::Options::WithName(
    StringPiece name) const {
  return Options(*this).WithNameImpl(name);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithDevice(
    StringPiece device) const {
  return Options(*this).WithDeviceImpl(device);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInput(
    Node* control_input) const {
  return Options(*this).WithControlInputImpl(control_input);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputs(
    gtl::ArraySlice<Node*> control_inputs) const {
  return Options(*this).WithControlInputsImpl(control_inputs);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithNameImpl(
    StringPiece name) {
  name_ = std::string(name);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithDeviceImpl(
    StringPiece device) {
  device_ = std::string(device);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputImpl(
    Node* control_input) {
  control_inputs_.push_back(control_input);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputsImpl(
    gtl::ArraySlice<Node*> control_inputs) {
  control_inputs_.insert(control_inputs_.end(), control_inputs.begin(),
                         control_inputs.end());
  return *this;
}

Status GraphDefBuilder::ToGraphDef(GraphDef* graph_def) const {
  if (status_.ok()) {
    graph_.ToGraphDef(graph_def);
  }
  return status_;
}

string GraphDefBuilder::Options::GetNameForOp(StringPiece op) const {
  if (name_.empty()) return graph_->NewName(op);
  return name_;
}

Node* GraphDefBuilder::Options::FinalizeBuilder(NodeBuilder* builder) const {
  builder->ControlInputs(control_inputs_);
  if (!device_.empty()) builder->Device(device_);
  for (const auto& attr : attrs_) {
    builder->Attr(attr.first, attr.second);
  }

  Node* returned_node;
  UpdateStatus(builder->Finalize(graph_, &returned_node));
  return returned_node;
}

void GraphDefBuilder::Options::UpdateStatus(const Status& status) const {
  if (status_ == nullptr) {
    TF_CHECK_OK(status);
  } else {
    status_->Update(status);
  }
}

namespace ops {

Node* SourceOp(const string& op_name, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  return opts.FinalizeBuilder(&node_builder);
}

Node* UnaryOp(const string& op_name, NodeOut input,
              const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(input));
  return opts.FinalizeBuilder(&node_builder);
}

Node* BinaryOp(const string& op_name, NodeOut a, NodeOut b,
               const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(a)).Input(std::move(b));
  return opts.FinalizeBuilder(&node_builder);
}

}  // end namespace ops
}  // end namespace tensorflow
