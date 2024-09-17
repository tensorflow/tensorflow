/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/cond_builder.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/graph_rewrite/incomplete_nodedef_builder.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {

CondBuilder::CondBuilder(std::string name, std::string device,
                         const NodeDebugInfo& debug, Graph* graph)
    : graph_(graph), name_(std::move(name)), device_(std::move(device)) {
  auto new_name = [graph, this](std::string suffix) {
    return graph->NewName(absl::StrCat(name_, "/", suffix));
  };
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("pred"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &pred_));
  Node* switch_pred;
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Switch(new_name("switch_pred"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_pred));
  graph_->AddEdge(pred(), 0, switch_pred, 0);
  graph_->AddEdge(pred(), 0, switch_pred, 1);
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("switch_f"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_f_));
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("switch_t"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_t_));
  graph_->AddEdge(switch_pred, kElseBranch, switch_f_, 0);
  graph_->AddEdge(switch_pred, kThenBranch, switch_t_, 0);
  Node* merge_pred;
  TF_CHECK_OK(IncompleteNodeDefBuilder::Merge(new_name("merge_pred"), DT_BOOL,
                                              debug, /*n=*/2)
                  .Device(device_)
                  .Build(graph_, &merge_pred));
  graph_->AddEdge(switch_f_, 0, merge_pred, kElseBranch);
  graph_->AddEdge(switch_t_, 0, merge_pred, kThenBranch);
  // Note: when additional return values are added then there should be a
  // control dependency between those merge nodes and control_successor_ to
  // ensure that it is control successor of conditional.
  control_successor_ = merge_pred;
}

Node* CondBuilder::pred() { return pred_; }

Node* CondBuilder::switch_f() { return switch_f_; }

Node* CondBuilder::switch_t() { return switch_t_; }

Node* CondBuilder::control_successor() { return control_successor_; }

Status CondBuilder::AddInput(const std::string& input_name,
                             const DataType& type, const std::string& device,
                             const NodeDebugInfo& debug, Node** input) {
  auto b = IncompleteNodeDefBuilder::Switch(
      graph_->NewName(absl::StrCat(name_, "/", input_name)), type, debug);
  TF_RETURN_IF_ERROR(b.Device(device).Build(graph_, input));
  graph_->AddEdge(pred(), 0, *input, 1);
  return absl::OkStatus();
}

}  // namespace tensorflow
