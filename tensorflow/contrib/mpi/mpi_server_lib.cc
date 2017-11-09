/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_MPI

#include "tensorflow/contrib/mpi/mpi_server_lib.h"

#include <string>
#include <utility>

#include "grpc/support/alloc.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {
// static utility function
RendezvousMgrInterface* NewMPIRendezvousMgr(const WorkerEnv* env) {
  // Runtime check to disable the MPI path
  const char* mpienv = getenv("MPI_DISABLED");
  if (mpienv && mpienv[0] == '1') {
    LOG(INFO) << "MPI path disabled by environment variable\n";
    return new RpcRendezvousMgr(env);
  } else {
    return new MPIRendezvousMgr(env);
  }
}

}  // namespace

MPIServer::MPIServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env) {}

MPIServer::~MPIServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());
}

Status MPIServer::Init(ServiceInitFunction service_func,
                       RendezvousMgrCreationFunction rendezvous_mgr_func) {
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  return s;
}

Status MPIServer::Start() {
  Status s = GrpcServer::Start();
  return s;
}

Status MPIServer::Join() {
  Status s = GrpcServer::Join();
  return s;
}

/* static */
Status MPIServer::Create(const ServerDef& server_def, Env* env,
                         std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<MPIServer> ret(new MPIServer(server_def, Env::Default()));
  ServiceInitFunction service_func = nullptr;
  TF_RETURN_IF_ERROR(ret->Init(service_func, NewMPIRendezvousMgr));
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class MPIServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+mpi";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return MPIServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `MPIServer` instances.
class MPIServerRegistrar {
 public:
  MPIServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("MPI_SERVER", new MPIServerFactory());
  }
};
static MPIServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
