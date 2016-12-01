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

#include "tensorflow/cc/client/client_session.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

ClientSession::ClientSession(const Scope& scope, const string& target)
    : ClientSession(scope, MakeDefaultSessionOptions(target)) {}

ClientSession::ClientSession(const Scope& scope) : ClientSession(scope, "") {}

ClientSession::ClientSession(const Scope& scope,
                             const SessionOptions& session_options)
    : session_(NewSession(session_options)),
      graph_(scope.graph_as_shared_ptr()) {
  CHECK_NOTNULL(session_.get());
}

SessionOptions ClientSession::MakeDefaultSessionOptions(
    const string& target) const {
  SessionOptions options;
  options.env = Env::Default();
  options.target = target;
  return options;
}

Status ClientSession::Run(const std::vector<ops::Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(FeedType{}, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<ops::Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(inputs, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<ops::Output>& fetch_outputs,
                          const std::vector<ops::Operation>& run_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(RunOptions(), inputs, fetch_outputs, run_outputs, outputs,
             nullptr);
}

Status ClientSession::MaybeExtendGraph() const {
  mutex_lock l(mu_);
  int num_nodes = graph_->num_node_ids();
  if (num_nodes > last_num_graph_nodes_) {
    GraphDef graph_def;
    graph_->ToGraphDefSubRange(&graph_def, last_num_graph_nodes_);
    last_num_graph_nodes_ = num_nodes;
    return session_->Extend(graph_def);
  }
  return Status::OK();
}

Status ClientSession::Run(const RunOptions& run_options, const FeedType& inputs,
                          const std::vector<ops::Output>& fetch_outputs,
                          const std::vector<ops::Operation>& run_outputs,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) const {
  std::vector<std::pair<string, Tensor>> feeds;
  for (auto const& feed : inputs) {
    TF_RETURN_IF_ERROR(feed.second.status);
    feeds.emplace_back(feed.first.name(), feed.second.tensor);
  }
  std::vector<string> output_tensor_names;
  for (auto const& output : fetch_outputs) {
    output_tensor_names.push_back(output.name());
  }
  std::vector<string> target_node_names;
  for (auto const& output : run_outputs) {
    target_node_names.push_back(output.node()->name());
  }
  TF_RETURN_IF_ERROR(MaybeExtendGraph());
  return session_->Run(run_options, feeds, output_tensor_names,
                       target_node_names, outputs, run_metadata);
}

}  // end namespace tensorflow
