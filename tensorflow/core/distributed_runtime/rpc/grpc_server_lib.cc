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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <limits>
#include <memory>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) {}

GrpcServer::~GrpcServer() {
  Stop();
  Join();

  delete master_service_;
  delete worker_service_;

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.
  delete master_env_.worker_cache;  // Shared with worker_env.worker_cache.

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  delete worker_env_.graph_mgr;
  delete worker_env_.device_mgr;

  delete worker_env_.rendezvous_mgr;

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

Status GrpcServer::Init() {
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  SessionOptions sess_opts;
  sess_opts.config = server_def_.default_session_config();

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0", "/task:",
                      server_def_.task_index());
  DeviceFactory::AddDevices(sess_opts, name_prefix, &master_env_.local_devices);
  worker_env_.device_mgr = new DeviceMgr(master_env_.local_devices);
  string unused;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &worker_env_.worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // Look up the port that has been requested for this task in `server_def_`.
  requested_port_ = -1;
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
      const std::vector<string> hostname_port =
          str_util::Split(iter->second, ':');
      if (hostname_port.size() != 2 ||
          !strings::safe_strto32(hostname_port[1], &requested_port_)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\"");
      } else {
        break;
      }
    }
  }
  if (requested_port_ == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }

  // N.B. The order of initialization here is intricate, because we
  // wish to allow `requested_port_ == 0` (for choosing any port,
  // mostly for testing). Therefore, the construction of the channel
  // and worker caches depends on `bound_port_`, which is not set
  // until we call `builder.BuildAndStart()`. We must create the
  // service objects before calling `builder.BuildAndStart()`, but
  // `master_env_` and `worker_env_` are only partially
  // configured. However, this is not dangerous, because we do not
  // start serving requests until `this->Start()` is called, which
  // happens after this method returns.
  //
  // TODO(mrry): Provide a general mechanism for dynamically setting
  // the identities of tasks in the worker pool after the service is
  // running.
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());
  master_service_ = NewGrpcMasterService(&master_env_, &builder);
  worker_service_ = NewGrpcWorkerService(&worker_env_, &builder);
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Internal("Could not start gRPC server");
  }

  GrpcChannelSpec channel_spec;
  for (const auto& job : server_def_.cluster().job()) {
    int max_task_id = -1;
    for (const auto& task : job.tasks()) {
      max_task_id = std::max(max_task_id, task.first);
    }
    std::vector<string> host_ports(max_task_id + 1);
    for (const auto& task : job.tasks()) {
      if (job.name() == server_def_.job_name() &&
          task.first == server_def_.task_index()) {
        host_ports[task.first] = strings::StrCat("localhost:", bound_port_);
      } else {
        host_ports[task.first] = task.second;
      }
    }
    channel_spec.AddHostPortsJob(job.name(), host_ports, host_ports.size());
  }

  std::unique_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(
      channel_spec, GetChannelCreationFunction(server_def_)));
  const string host_port = channel_cache->TranslateTask(name_prefix);
  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port_)) {
    return errors::Internal("Could not parse port for local server from \"",
                            channel_cache->TranslateTask(name_prefix), "\".");
  }
  worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_env_.worker_cache;
  master_env_.master_session_factory = internal::NewMasterSession;

  // Finish setting up worker environment.
  worker_env_.graph_mgr = new GraphMgr(&worker_env_);
  worker_env_.compute_pool = ComputePool(sess_opts);
  worker_env_.rendezvous_mgr = new RpcRendezvousMgr(&worker_env_);

  return Status::OK();
}

Status GrpcServer::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      worker_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_worker_service",
                            [this] { worker_service_->HandleRPCsLoop(); }));
      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      CHECK(false);
  }
}

Status GrpcServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      server_->Shutdown();
      master_service_->Shutdown();
      worker_service_->Shutdown();
      state_ = STOPPED;
      return Status::OK();
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      CHECK(false);
  }
}

Status GrpcServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      return Status::OK();
    default:
      CHECK(false);
  }
}

const string GrpcServer::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction(
    const ServerDef& server_def) const {
  return NewHostPortGrpcChannel;
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GrpcServer> ret(new GrpcServer(server_def, Env::Default()));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return GrpcServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `GrpcServer` instances.
class GrpcServerRegistrar {
 public:
  GrpcServerRegistrar() {
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
  }
};
static GrpcServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
