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
#ifndef TENSORFLOW_PYTHON_CLIENT_SESSION_REF_H_
#define TENSORFLOW_PYTHON_CLIENT_SESSION_REF_H_

#include <memory>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class SessionLogger;

// A `SessionRef` manages the lifetime of a wrapped `Session` pointer.
//
// SessionRef blocks the return of Close() until all pending operations have
// been completed or cancelled and underlying session has been freed.  Any
// subsequent operations on the SessionRef object will return errors::Cancelled.
class SessionRef : public Session {
 public:
  explicit SessionRef(Session* session);
  ~SessionRef() override;

  Status Create(const GraphDef& graph) override;
  Status Extend(const GraphDef& graph) override;
  Status Create(const RunOptions& run_options, const GraphDef& graph) override;
  Status Extend(const RunOptions& run_options, const GraphDef& graph) override;
  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override;

  Status Close() override;
  Status Close(const RunOptions& run_options) override;

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  Status PRunSetup(const std::vector<string>& input_names,
                   const std::vector<string>& output_names,
                   const std::vector<string>& target_nodes,
                   string* handle) override;

  Status PRun(const string& handle,
              const std::vector<std::pair<string, Tensor> >& inputs,
              const std::vector<string>& output_names,
              std::vector<Tensor>* outputs) override;

  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle) override;

  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata) override;

  Status ReleaseCallable(CallableHandle handle) override;

 private:
  mutex run_lock_;
  condition_variable run_finished_;
  uint64 run_count_ TF_GUARDED_BY(run_lock_) = {0};
  std::shared_ptr<Session> session_;

  // Borrowed reference to global session logger.
  SessionLogger* logger_;

  Status CheckNotClosed();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_CLIENT_SESSION_REF_H_
