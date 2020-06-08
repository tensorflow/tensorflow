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
#include "tensorflow/c/experimental/network.h"

#include <memory>
#include <string>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/experimental/network_internal.h"
#include "tensorflow/c/experimental/rendezvous_internal.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

using tensorflow::ServerFactory;

namespace tensorflow {

/* static */ Status CGrpcServer::Create(
    const ServerDef& server_def,
    void* (*init_function)(const TF_GrpcServer*, TF_Status*),
    void (*start_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*delete_function)(void*),
    TF_RemoteRendezvousBuilder* rendezvous_builder,
    std::unique_ptr<ServerInterface>* out_server) {
  auto* grpc_server = new CGrpcServer(server_def, start_function, stop_function,
                                      join_function, delete_function);

  GrpcServerOptions options;
  options.rendezvous_mgr_func = [rendezvous_builder](const WorkerEnv* env) {
    return new CRendezvousMgr(env, rendezvous_builder);
  };
  TF_RETURN_IF_ERROR(grpc_server->Init(options));
  TF_Status* tf_status = TF_NewStatus();
  grpc_server->SetContext(init_function(
      reinterpret_cast<const TF_GrpcServer*>(grpc_server), tf_status));
  TF_RETURN_IF_ERROR(tf_status->status);
  TF_DeleteStatus(tf_status);

  out_server->reset(grpc_server);
  return Status::OK();
}

Status CGrpcServer::Start() {
  Status status = GrpcServer::Start();
  TF_Status* tf_status = TF_NewStatus();
  (*start_function_)(reinterpret_cast<const TF_GrpcServer*>(this), context_,
                     tf_status);
  status.Update(tf_status->status);
  TF_DeleteStatus(tf_status);
  return status;
}

Status CGrpcServer::Stop() {
  Status status = GrpcServer::Stop();
  TF_Status* tf_status = TF_NewStatus();
  (*stop_function_)(reinterpret_cast<const TF_GrpcServer*>(this), context_,
                    tf_status);
  status.Update(tf_status->status);
  TF_DeleteStatus(tf_status);
  return status;
}

Status CGrpcServer::Join() {
  Status status = GrpcServer::Join();
  TF_Status* tf_status = TF_NewStatus();
  (*join_function_)(reinterpret_cast<const TF_GrpcServer*>(this), context_,
                    tf_status);
  status.Update(tf_status->status);
  TF_DeleteStatus(tf_status);
  return status;
}

namespace {
// Factory that creates CGrpcServer instances.
class CServerFactory : public ServerFactory {
 public:
  CServerFactory(bool (*accept_function)(const char*),
                 void* (*init_function)(const TF_GrpcServer*, TF_Status*),
                 void (*start_function)(const TF_GrpcServer*, void*,
                                        TF_Status*),
                 void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
                 void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
                 void (*delete_function)(void*),
                 TF_RemoteRendezvousBuilder* rendezvous_builder)
      : accept_function_(accept_function),
        init_function_(init_function),
        start_function_(start_function),
        stop_function_(stop_function),
        join_function_(join_function),
        delete_function_(delete_function),
        rendezvous_builder_(rendezvous_builder) {}

  Status NewServer(const ServerDef& server_def, const Options& options,
                   std::unique_ptr<ServerInterface>* out_server) override {
    TF_RETURN_IF_ERROR(CGrpcServer::Create(
        server_def, init_function_, start_function_, stop_function_,
        join_function_, delete_function_, rendezvous_builder_, out_server));
    return Status::OK();
  }

  // Returns true if and only if this factory can create a server
  // based on the given `server_def`.
  bool AcceptsOptions(const ServerDef& server_def) override {
    return (*accept_function_)(server_def.protocol().c_str());
  }

 private:
  bool (*accept_function_)(const char* protocol);
  void* (*init_function_)(const TF_GrpcServer*, TF_Status*);
  void (*start_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*stop_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*join_function_)(const TF_GrpcServer*, void*, TF_Status*);
  void (*delete_function_)(void*);
  TF_RemoteRendezvousBuilder* rendezvous_builder_;
};
}  // namespace
}  // namespace tensorflow

// Server factory representation to use in C API.
// Holds CServerFactory pointer.
struct TF_GrpcServerFactory {
  ::tensorflow::CServerFactory* factory;
};

TF_GrpcServerFactory* TF_NewGrpcServerFactory(
    bool (*accept_function)(const char*),
    void* (*init_function)(const TF_GrpcServer*, TF_Status*),
    void (*start_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*delete_function)(void*),
    TF_RemoteRendezvousBuilder* rendezvous_builder) {
  TF_GrpcServerFactory* server_factory = new TF_GrpcServerFactory;
  server_factory->factory = new ::tensorflow::CServerFactory(
      accept_function, init_function, start_function, stop_function,
      join_function, delete_function, rendezvous_builder);
  return server_factory;
}

void TF_DeleteGrpcServerFactory(TF_GrpcServerFactory* server_factory) {
  DCHECK_NE(server_factory, nullptr);
  delete server_factory;
}

void TF_RegisterGrpcServerFactory(const char* server_type,
                                  TF_GrpcServerFactory* server_factory) {
  ServerFactory::Register(server_type, server_factory->factory);
}
