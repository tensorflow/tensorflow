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
#include <utility>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ClientSession::Impl {
 private:
  friend class ClientSession;

  Impl(Session* session, std::shared_ptr<Graph> graph)
      : session_(session), graph_(std::move(graph)) {}

  static SessionOptions MakeDefaultSessionOptions(const string& target);
  Status MaybeExtendGraph() const;

  std::unique_ptr<Session> session_;
  std::shared_ptr<Graph> graph_;

  mutable mutex mu_;
  mutable int last_num_graph_nodes_ GUARDED_BY(mu_) = 0;
};

ClientSession::ClientSession(const Scope& scope, const string& target)
    : ClientSession(scope, Impl::MakeDefaultSessionOptions(target)) {}

ClientSession::ClientSession(const Scope& scope) : ClientSession(scope, "") {}

ClientSession::ClientSession(const Scope& scope,
                             const SessionOptions& session_options) {
  Session* new_session;
  Status status = NewSession(session_options, &new_session);
  TF_CHECK_OK(status) << status;
  impl_.reset(new Impl(new_session, scope.graph_as_shared_ptr()));
  CHECK_NOTNULL(impl()->session_.get());
}

// Define destructor here so we can forward declare `Impl` in client_session.h.
// If we define a dtor in the header file or use the default dtor,
// unique_ptr<Impl> needs the complete type.
ClientSession::~ClientSession() {}

SessionOptions ClientSession::Impl::MakeDefaultSessionOptions(
    const string& target) {
  SessionOptions options;
  options.env = Env::Default();
  options.target = target;
  return options;
}

Status ClientSession::Run(const std::vector<Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(FeedType{}, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(inputs, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<Output>& fetch_outputs,
                          const std::vector<Operation>& run_outputs,
                          std::vector<Tensor>* outputs) const {
  return Run(RunOptions(), inputs, fetch_outputs, run_outputs, outputs,
             nullptr);
}

Status ClientSession::Impl::MaybeExtendGraph() const {
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
                          const std::vector<Output>& fetch_outputs,
                          const std::vector<Operation>& run_outputs,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) const {
  std::vector<std::pair<string, Tensor>> feeds;
  for (auto const& feed : inputs) {
    TF_RETURN_IF_ERROR(feed.second.status);
    feeds.emplace_back(feed.first.name(), feed.second.tensor);
  }
  std::vector<string> output_tensor_names;
  output_tensor_names.reserve(fetch_outputs.size());
  for (auto const& output : fetch_outputs) {
    output_tensor_names.push_back(output.name());
  }
  std::vector<string> target_node_names;
  target_node_names.reserve(run_outputs.size());
  for (auto const& output : run_outputs) {
    target_node_names.push_back(output.node()->name());
  }
  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->Run(run_options, feeds, output_tensor_names,
                               target_node_names, outputs, run_metadata);
}

Status ClientSession::Run(
    const RunOptions& run_options, const FeedType& inputs,
    const std::vector<Output>& fetch_outputs,
    const std::vector<Operation>& run_outputs, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options) const {
  std::vector<std::pair<string, Tensor>> feeds;
  for (auto const& feed : inputs) {
    TF_RETURN_IF_ERROR(feed.second.status);
    feeds.emplace_back(feed.first.name(), feed.second.tensor);
  }
  std::vector<string> output_tensor_names;
  output_tensor_names.reserve(fetch_outputs.size());
  for (auto const& output : fetch_outputs) {
    output_tensor_names.push_back(output.name());
  }
  std::vector<string> target_node_names;
  target_node_names.reserve(run_outputs.size());
  for (auto const& output : run_outputs) {
    target_node_names.push_back(output.node()->name());
  }
  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->Run(run_options, feeds, output_tensor_names,
                               target_node_names, outputs, run_metadata,
                               threadpool_options);
}

Status ClientSession::MakeCallable(const CallableOptions& callable_options,
                                   CallableHandle* out_handle) {
  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->MakeCallable(callable_options, out_handle);
}

Status ClientSession::RunCallable(CallableHandle handle,
                                  const std::vector<Tensor>& feed_tensors,
                                  std::vector<Tensor>* fetch_tensors,
                                  RunMetadata* run_metadata) {
  return impl()->session_->RunCallable(handle, feed_tensors, fetch_tensors,
                                       run_metadata);
}

Status ClientSession::RunCallable(CallableHandle handle,
                                  const std::vector<Tensor>& feed_tensors,
                                  std::vector<Tensor>* fetch_tensors,
                                  RunMetadata* run_metadata,
                                  const thread::ThreadPoolOptions& options) {
  return impl()->session_->RunCallable(handle, feed_tensors, fetch_tensors,
                                       run_metadata, options);
}

Status ClientSession::ReleaseCallable(CallableHandle handle) {
  return impl()->session_->ReleaseCallable(handle);
}

}  // end namespace tensorflow
