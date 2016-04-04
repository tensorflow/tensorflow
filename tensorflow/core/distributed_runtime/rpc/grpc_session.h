/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class MasterInterface;

// A Session instance lets the caller drive a TensorFlow graph
// computation on potentially remote sets of devices. This is a thin
// wrapper around tensorflow::grpc::MasterService.
//
// Multiple threads must synchronize their accesses to a single
// session.
class GrpcSession : public Session {
 protected:
  explicit GrpcSession(const SessionOptions& options);

 public:
  static Status Create(const SessionOptions& options,
                       std::unique_ptr<GrpcSession>* out_session);

  ~GrpcSession() override;

  // Creates a session with the "target". The session carries out
  // the graph computation defined by "graph", and will have version
  // number "initial_version".
  Status Create(const GraphDef& graph) override;
  Status Create(const RunOptions& run_options, const GraphDef& graph) override;

  // Runs with and without RunOptions.
  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata);

  Status Extend(const GraphDef& graph) override;
  Status Extend(const RunOptions& run_options, const GraphDef& graph) override;

  Status Close() override;

  // NOTE: This API is still experimental and may change.
  ::tensorflow::Status PRunSetup(const std::vector<string>& input_names,
                                 const std::vector<string>& output_names,
                                 const std::vector<string>& target_nodes,
                                 string* handle) override;

  // NOTE: This API is still experimental and may change.
  ::tensorflow::Status PRun(
      const string& handle,
      const std::vector<std::pair<string, Tensor> >& inputs,
      const std::vector<string>& output_names,
      std::vector<Tensor>* outputs) override;

  std::vector<DeviceAttributes> ListDevices();

 protected:
  // Takes ownership of `*master`.
  void SetRemoteMaster(MasterInterface* master);

 private:
  SessionOptions options_;
  std::unique_ptr<MasterInterface> master_;
  mutex mu_;

  // handle_ returned by the master to identify this session.
  string handle_;

  // The current version of the graph.
  int64 current_graph_version_ GUARDED_BY(mu_);

  Status RunProto(CallOptions* call_options, RunStepRequest* req,
                  RunStepResponse* resp);

  // Implementations for all the public interfaces.
  Status CreateImpl(CallOptions* call_options, const GraphDef& graph);
  Status ExtendImpl(CallOptions* call_options, const GraphDef& graph);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcSession);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_
