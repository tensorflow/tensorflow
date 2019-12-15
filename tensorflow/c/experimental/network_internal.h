/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_NETWORK_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_NETWORK_INTERNAL_H_

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/network.h"
#include "tensorflow/c/experimental/rendezvous.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

// GrpcServer implementation that forwards calls to callbacks.
class CGrpcServer : public GrpcServer {
 protected:
  CGrpcServer(const ServerDef& server_def,
              void (*start_function)(const TF_GrpcServer*, void*, TF_Status*),
              void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
              void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
              void (*delete_function)(void*))
      : GrpcServer(server_def, ::tensorflow::Env::Default()),
        start_function_(start_function),
        stop_function_(stop_function),
        join_function_(join_function),
        delete_function_(delete_function),
        context_(nullptr) {}

 public:
  static Status Create(
      const ServerDef& server_def,
      void* (*init_function)(const TF_GrpcServer*, TF_Status*),
      void (*start_function)(const TF_GrpcServer*, void*, TF_Status*),
      void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
      void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
      void (*delete_function)(void*),
      TF_RemoteRendezvousBuilder* rendezvous_builder,
      std::unique_ptr<ServerInterface>* out_server);

  Status Start() override;
  Status Stop() override;
  Status Join() override;

  ~CGrpcServer() override { delete_function_(context_); }

 protected:
  void SetContext(void* context) { context_ = context; }

 private:
  void (*start_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*stop_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*join_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*delete_function_)(void*);
  void* context_;

  friend class NetworksTest;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_C_EXPERIMENTAL_NETWORK_INTERNAL_H_
