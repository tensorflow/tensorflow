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

#include "tensorflow/contrib/gdr/gdr_server_lib.h"

#include "grpc/support/alloc.h"
#include "tensorflow/contrib/gdr/gdr_collective_executor_mgr.h"
#include "tensorflow/contrib/gdr/gdr_memory_manager.h"
#include "tensorflow/contrib/gdr/gdr_rendezvous_mgr.h"
#include "tensorflow/contrib/gdr/gdr_worker.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"

namespace tensorflow {

GdrServer::GdrServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env) {
  string host;
  string port;
  for (const auto& job : server_def.cluster().job()) {
    if (job.name() == server_def.job_name()) {
      auto iter = job.tasks().find(server_def.task_index());
      if (iter != job.tasks().end()) {
        const std::vector<string> hostname_port =
            str_util::Split(iter->second, ':');
        if (hostname_port.size() == 2) {
          host = hostname_port[0];
          port = hostname_port[1];
        }
      }
    }
  }
  remote_memory_manager_ = std::unique_ptr<RemoteMemoryManager>(
      CreateRemoteMemoryManager(host, port));
}

GdrServer::~GdrServer() {}

Status GdrServer::Init() {
  RendezvousMgrCreationFunction rendezvous_mgr_func =
      [this](const WorkerEnv* env) {
        return new GdrRendezvousMgr(env, remote_memory_manager_.get());
      };
  WorkerCreationFunction worker_func = [this](WorkerEnv* env,
                                              const ConfigProto& config) {
    return std::unique_ptr<GdrWorker>(
        new GdrWorker(env, config, remote_memory_manager_.get()));
  };
  CollectiveMgrCreationFunction collective_mgr_func = [this](
      const ConfigProto& config, const WorkerEnv* env,
      WorkerCacheInterface* worker_cache) {

    string unused;
    string default_worker_name;
    DeviceNameUtils::SplitDeviceName(env->device_mgr->ListDevices()[0]->name(),
                                     &default_worker_name, &unused);

    std::unique_ptr<DeviceResolverDistributed> dev_resolver(
        new DeviceResolverDistributed(env->device_mgr, worker_cache,
                                      default_worker_name));
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
        new CollectiveParamResolverDistributed(config, env->device_mgr,
                                               dev_resolver.get(), worker_cache,
                                               default_worker_name));
    return new GdrCollectiveExecutorMgr(
        config, env->device_mgr, std::move(dev_resolver),
        std::move(param_resolver), worker_cache, default_worker_name,
        remote_memory_manager_.get());

  };
  TF_RETURN_IF_ERROR(remote_memory_manager_->Init());

  return GrpcServer::Init(nullptr, rendezvous_mgr_func, collective_mgr_func,
                          worker_func);
}

Status GdrServer::Start() {
  {
    mutex_lock l(mu_);
    gdr_thread_.reset(worker_env()->env->StartThread(
        ThreadOptions(), "TF_gdr_service",
        [this] { remote_memory_manager_->Run(); }));
  }
  return GrpcServer::Start();
}

Status GdrServer::Stop() {
  remote_memory_manager_->Stop();
  return GrpcServer::Stop();
}

Status GdrServer::Join() {
  {
    mutex_lock l(mu_);
    gdr_thread_.reset();
  }
  return GrpcServer::Join();
}

/* static */
Status GdrServer::Create(const ServerDef& server_def, Env* env,
                         std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GdrServer> ret(
      new GdrServer(server_def, env == nullptr ? Env::Default() : env));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class GdrServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+gdr";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return GdrServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `GdrServer` instances.
class GdrServerRegistrar {
 public:
  GdrServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("GDR_SERVER", new GdrServerFactory());
  }
};
static GdrServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
