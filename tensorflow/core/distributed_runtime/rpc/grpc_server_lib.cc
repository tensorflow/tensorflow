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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <memory>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {
class TensorFlowServer : public ServerInterface {
 public:
  TensorFlowServer(const ServerDef& server_def, Env* env)
      : server_def_(server_def), env_(env), state_(NEW) {}

  ~TensorFlowServer() {
    Stop();
    Join();

    delete master_service_;
    delete worker_service_;

    // TODO(mrry): Refactor the *Env classes so that it is less fiddly
    // to destroy them.
    delete master_env_.worker_cache;  // Shared with worker_env.worker_cache.

    delete worker_env_.device_mgr;
    delete worker_env_.graph_mgr;
    delete worker_env_.rendezvous_mgr;

    // Do not delete (as these are not owned by the server):
    // - master_env_.env
    // - worker_env_.env
    // - worker_env_.compute_pool
  }

  Status Init() {
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
    DeviceFactory::AddDevices(sess_opts, name_prefix,
                              &master_env_.local_devices);
    worker_env_.device_mgr = new DeviceMgr(master_env_.local_devices);
    string unused;
    if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                          &worker_env_.worker_name, &unused)) {
      return errors::Internal("Could not parse worker name.");
    }

    GrpcChannelSpec channel_spec;
    for (const auto& job : server_def_.cluster().job()) {
      int max_task_id = -1;
      for (const auto& task : job.tasks()) {
        max_task_id = std::max(max_task_id, task.first);
      }
      std::vector<string> host_ports(max_task_id + 1);
      for (const auto& task : job.tasks()) {
        host_ports[task.first] = task.second;
      }
      channel_spec.AddHostPortsJob(job.name(), host_ports, host_ports.size());
    }

    std::unique_ptr<GrpcChannelCache> channel_cache(
        NewGrpcChannelCache(channel_spec));
    const string host_port = channel_cache->TranslateTask(name_prefix);
    if (!str_util::NumericParse32(str_util::Split(host_port, ':')[1],
                                  &requested_port_)) {
      return errors::Internal("Could not parse port for local server from \"",
                              channel_cache->TranslateTask(name_prefix), "\".");
    }
    target_ = strings::StrCat("grpc://", host_port);

    worker_env_.worker_cache = NewGrpcWorkerCache(channel_cache.release());

    // Finish setting up master environment.
    master_env_.ops = OpRegistry::Global();
    master_env_.worker_cache = worker_env_.worker_cache;
    master_env_.master_session_factory = internal::NewMasterSession;

    // Finish setting up worker environment.
    worker_env_.graph_mgr = new GraphMgr(&worker_env_);
    worker_env_.rendezvous_mgr = new RpcRendezvousMgr(&worker_env_);
    worker_env_.compute_pool = ComputePool(sess_opts);

    return Status::OK();
  }

  Status Start() override {
    mutex_lock l(mu_);
    switch (state_) {
      case NEW: {
        ::grpc::ServerBuilder builder;
        builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                                 ::grpc::InsecureServerCredentials());
        master_service_ = NewGrpcMasterService(&master_env_, &builder);
        worker_service_ = NewGrpcWorkerService(&worker_env_, &builder);
        server_ = builder.BuildAndStart();
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

  Status Stop() override {
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

  Status Join() override {
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

  const string& target() const override { return target_; }

 private:
  // The overall server configuration.
  const ServerDef server_def_;
  Env* env_;

  // The port requested for this server.
  // TODO(mrry): Support requested_port_ == 0 to bind to any available port.
  int requested_port_;

  // The `SessionOptions.target` to be used when connecting to this
  // server (as a master).
  string target_;

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
  AsyncServiceInterface* master_service_;
  std::unique_ptr<Thread> master_thread_ GUARDED_BY(mu_);

  // Implementation of a TensorFlow worker, and RPC polling thread.
  WorkerEnv worker_env_;
  AsyncServiceInterface* worker_service_;
  std::unique_ptr<Thread> worker_thread_ GUARDED_BY(mu_);

  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);
};
}  // namespace

Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<TensorFlowServer> ret(
      new TensorFlowServer(server_def, Env::Default()));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

}  // namespace tensorflow
