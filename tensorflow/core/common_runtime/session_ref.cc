/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/session_ref.h"

#include <utility>

namespace tensorflow {

namespace {

// Scope helper to track active calls and manage session lifetime.
struct RunCounter {
  std::shared_ptr<Session> session;
  uint64* value;
  mutex* m;
  condition_variable* cv;

  explicit RunCounter(std::shared_ptr<Session> s, uint64* v, mutex* m,
                      condition_variable* cv)
      : session(std::move(s)), value(v), m(m), cv(cv) {
    mutex_lock l(*m);
    ++*value;
  }

  ~RunCounter() {
    mutex_lock l(*m);
    if (--*value == 0) {
      cv->notify_all();
    }
  }
};

}  // namespace

Status SessionRef::CheckNotClosed() {
  mutex_lock l(run_lock_);
  if (session_ == nullptr) return errors::Cancelled("Session has been closed.");
  return ::tensorflow::Status::OK();
}

Status SessionRef::Run(const RunOptions& run_options,
                       const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs,
                       RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Run(run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
}

Status SessionRef::Create(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Create(graph);
}

Status SessionRef::Create(const RunOptions& run_options,
                          const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Create(run_options, graph);
}

Status SessionRef::Extend(const RunOptions& run_options,
                          const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Extend(run_options, graph);
}

Status SessionRef::Extend(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Extend(graph);
}

Status SessionRef::Close(const RunOptions& run_options) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status = session_->Close(run_options);
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

Status SessionRef::Close() {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status = session_->Close();
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

Status SessionRef::Run(const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->Run(inputs, output_tensor_names, target_node_names,
                         outputs);
}

Status SessionRef::ListDevices(std::vector<DeviceAttributes>* response) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->ListDevices(response);
}

Status SessionRef::PRunSetup(const std::vector<string>& input_names,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             string* handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->PRunSetup(input_names, output_names, target_nodes, handle);
}

Status SessionRef::PRun(const string& handle,
                        const std::vector<std::pair<string, Tensor> >& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->PRun(handle, inputs, output_names, outputs);
}

Status SessionRef::MakeCallable(const CallableOptions& callable_options,
                                CallableHandle* out_handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->MakeCallable(callable_options, out_handle);
}

Status SessionRef::RunCallable(CallableHandle handle,
                               const std::vector<Tensor>& feed_tensors,
                               std::vector<Tensor>* fetch_tensors,
                               RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->RunCallable(handle, feed_tensors, fetch_tensors,
                                 run_metadata);
}

Status SessionRef::ReleaseCallable(CallableHandle handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_);
  return rc.session->ReleaseCallable(handle);
}

}  // namespace tensorflow
