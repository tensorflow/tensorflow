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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_

#include <memory>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class GrpcServer : public ServerInterface {
 protected:
  GrpcServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  virtual ~GrpcServer();

  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Stop() override;
  Status Join() override;
  const string target() const override;

 protected:
  Status Init();

  // A subclass can override this method to support secure credentials.
  virtual std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
      const ServerDef& server_def) const;

  virtual ChannelCreationFunction GetChannelCreationFunction(
      const ServerDef& server_def) const;

  // Returns the port to which this server is bound.
  // This method may only be called after `this->Init()` returns successfully.
  int bound_port() const { return bound_port_; }

 private:
  // The overall server configuration.
  const ServerDef server_def_;
  Env* env_;

  // The port requested for this server.
  int requested_port_;
  // The port to which this server is bound.
  int bound_port_ = 0;

  // Guards state transitions.
  mutex mu_;

  // Represents the current state of the server, which changes as follows:
  //
  //                 Join()            Join()
  //                  ___               ___
  //      Start()     \ /    Stop()     \ /
  // NEW ---------> STARTED --------> STOPPED
  //   \                          /
  //    \________________________/
  //            Stop(), Join()
  enum State { NEW, STARTED, STOPPED };
  State state_ GUARDED_BY(mu_);

  // Implementation of a TensorFlow master, and RPC polling thread.
  MasterEnv master_env_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::unique_ptr<Thread> master_thread_ GUARDED_BY(mu_);

  // Implementation of a TensorFlow worker, and RPC polling thread.
  WorkerEnv worker_env_;
  AsyncServiceInterface* worker_service_ = nullptr;
  std::unique_ptr<Thread> worker_thread_ GUARDED_BY(mu_);

  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
