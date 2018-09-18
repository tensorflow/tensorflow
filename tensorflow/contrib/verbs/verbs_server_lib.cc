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

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/verbs_server_lib.h"

#include "grpc/support/alloc.h"

#include "tensorflow/contrib/verbs/rdma_mgr.h"
#include "tensorflow/contrib/verbs/rdma_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {
// static utility function
RendezvousMgrInterface* NewRdmaRendezvousMgr(const WorkerEnv* env) {
  return new RdmaRendezvousMgr(env);
}

}  // namespace

VerbsServer::VerbsServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env), verbs_state_(DISCONNECTED) {}

VerbsServer::~VerbsServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());
  delete rdma_mgr_;
  delete verbs_service_;
  delete channel_cache_;
}

Status VerbsServer::ChannelCacheFactory(const ServerDef& server_def,
                                        GrpcChannelCache** channel_cache) {
  string name_prefix =
      strings::StrCat("/job:", server_def.job_name(), "/replica:0",
                      "/task:", server_def.task_index());

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(server_def, &channel_spec));

  *channel_cache =
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction());

  const string host_port = (*channel_cache)->TranslateTask(name_prefix);
  int requested_port;

  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            (*channel_cache)->TranslateTask(name_prefix),
                            "\".");
  }
  if (requested_port != bound_port()) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ",
                                   bound_port());
  }

  return Status::OK();
}

Status VerbsServer::Init(ServiceInitFunction service_func,
                         RendezvousMgrCreationFunction rendezvous_mgr_func) {
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  {
    mutex_lock l(mu_);
    CHECK_EQ(verbs_state_, DISCONNECTED);
    CHECK(ChannelCacheFactory(server_def(), &channel_cache_).ok());
    rdma_mgr_ = new RdmaMgr(worker_env(), channel_cache_);
    // set rdma_mgr for verbs_service and rdma_rendezvous_mgr
    verbs_service_->SetRdmaMgr(rdma_mgr_);
    dynamic_cast<RdmaRendezvousMgr*>(worker_env()->rendezvous_mgr)
        ->SetRdmaMgr(rdma_mgr_);
  }
  return s;
}

Status VerbsServer::Start() {
  Status s = GrpcServer::Start();
  {
    mutex_lock l(mu_);
    if (verbs_state_ == DISCONNECTED) {
      // verbs_thread needs to be initiated
      // before rdma_mgr sets up the rdma channels.
      verbs_thread_.reset(worker_env()->env->StartThread(
          ThreadOptions(), "TF_verbs_service",
          [this] { verbs_service_->HandleRPCsLoop(); }));
      rdma_mgr_->SetupChannels();
      CHECK(rdma_mgr_->ConnectivityCheck()) << "Connectivity check failed!";
      rdma_mgr_->InitAllocators();
      verbs_state_ = CONNECTED;
    }
  }
  return s;
}

Status VerbsServer::Join() {
  Status s = GrpcServer::Join();
  {
    mutex_lock l(mu_);
    if (verbs_state_ == CONNECTED) {
      verbs_state_ = DISCONNECTED;
      verbs_thread_.reset();
    }
  }
  return s;
}

/* static */
Status VerbsServer::Create(const ServerDef& server_def, Env* env,
                           std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<VerbsServer> ret(new VerbsServer(server_def, Env::Default()));
  ServiceInitFunction service_func = [&ret](const WorkerEnv* worker_env,
                                            ::grpc::ServerBuilder* builder) {
    return SetNewVerbsService(&ret->verbs_service_, worker_env, builder);
  };
  TF_RETURN_IF_ERROR(ret->Init(service_func, NewRdmaRendezvousMgr));
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class VerbsServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+verbs";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return VerbsServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `VerbsServer` instances.
class VerbsServerRegistrar {
 public:
  VerbsServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("VERBS_SERVER", new VerbsServerFactory());
  }
};
static VerbsServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow

#endif
