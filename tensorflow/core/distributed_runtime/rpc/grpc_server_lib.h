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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_

// GrpcServer manages the lifecycle of an Eager, Worker and Master service.

#include <memory>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"

namespace tensorflow {

class GrpcWorker;
class Master;

// function that creates a RendezvousMgr.
typedef std::function<RendezvousMgrInterface*(const WorkerEnv*)>
    RendezvousMgrCreationFunction;

// function that creates a CollectiveExecutorMgr.
typedef std::function<CollectiveExecutorMgrInterface*(
    const ConfigProto&, const WorkerEnv*, WorkerCacheInterface*)>
    CollectiveMgrCreationFunction;

// function that registers a service to the server. The service needs to
// be registered before builder.BuildAndStart().
typedef std::function<void(const WorkerEnv*, ::grpc::ServerBuilder*)>
    ServiceInitFunction;

// function that creates a grpc based worker implementation.
typedef std::function<std::unique_ptr<GrpcWorker>(WorkerEnv*,
                                                  const ConfigProto& config)>
    WorkerCreationFunction;

struct GrpcServerOptions {
  ServiceInitFunction service_func = nullptr;
  RendezvousMgrCreationFunction rendezvous_mgr_func = nullptr;
  CollectiveMgrCreationFunction collective_mgr_func = nullptr;
  WorkerCreationFunction worker_func = nullptr;
  StatsPublisherFactory stats_factory = CreateNoOpStatsPublisher;
  GrpcWorkerServiceOptions worker_service_options;
  DeviceMgr* local_device_mgr = nullptr;
};

class GrpcServer : public ServerInterface {
 protected:
  GrpcServer(const ServerDef& server_def, Env* env);
  GrpcServer(const ServerDef& server_def, DeviceMgr* local_device_mgr,
             Env* env);
  // Allow children classes to override this and provide custom args to the
  // server before it is constructed. Default behavior is to do nothing.
  // requested_port provides the port requested by caller as bound_port() is
  // not available till BuildAndStart has been called.
  virtual void MaybeMutateBuilder(::grpc::ServerBuilder* builder,
                                  int requested_port) {}

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<GrpcServer>* out_server);
  // Reuse the local_device_mgr.
  static Status Create(const ServerDef& server_def, Env* env,
                       DeviceMgr* local_device_mgr,
                       std::unique_ptr<ServerInterface>* out_server);

  // Destruction is only supported in the factory method. Clean
  // shutdown is not currently implemented for this server type.
  virtual ~GrpcServer();

  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Stop() override;
  Status Join() override;
  const string target() const override;

  WorkerEnv* worker_env() { return &worker_env_; }
  MasterEnv* master_env() { return &master_env_; }

  // Add master eager context to local eager service in order to handle enqueue
  // requests from remote workers.
  Status AddMasterEagerContextToEagerService(
      const tensorflow::uint64 context_id, tensorflow::EagerContext* context);
  // Update the set of workers that can be reached by the GRPC server
  Status UpdateServerDef(const ServerDef& server_def);

 protected:
  virtual Status GetHostAndPort(const ServerDef& server_def, string* host_name,
                                int* port) const;
  Status Init(const GrpcServerOptions& opts = GrpcServerOptions());

  // A subclass can override this method to support secure credentials.
  virtual std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
      const ServerDef& server_def) const;

  virtual ChannelCreationFunction GetChannelCreationFunction() const;

  virtual std::unique_ptr<Master> CreateMaster(MasterEnv* master_env);

  // Creates a WorkerCacheInterface for a session.
  virtual Status WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                    WorkerCacheInterface** worker_cache);

  // Override to return extra services to be brought up and managed along with
  // the standard {master, worker, eager} services. The map key is an aribtrary
  // string and the value is a pointer to the service to be brought up.
  // Ownership of the pointer is transferred to GrpcServer after this call
  // returns, and the service will be destroyed during the destruction of
  // GrpcServer. Each service will have its HandleRPCsLoop called in a separate
  // thread. An example usage would be to add a RDMA based partial worker
  // service to offload tensor and data buffer transfers.
  virtual std::map<std::string, AsyncServiceInterface*> ExtraServices(
      ::grpc::ServerBuilder*) {
    return {};
  }

  // Parses a WorkerCacheFactoryOptions into a GrpcChannelSpec.
  Status ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                          GrpcChannelSpec* channel_spec);

  // Returns the port to which this server is bound.
  // This method may only be called after `this->Init()` returns successfully.
  int bound_port() const { return bound_port_; }

  // Returns hostname.
  const string& host_name() const { return host_name_; }

  const ServerDef& server_def() const { return server_def_; }
  GrpcWorker* worker_impl() const { return worker_impl_.get(); }
  GrpcWorkerEnv* grpc_worker_env() const { return grpc_worker_env_.get(); }

 private:
  Env* env_;

  // The port to which this server is bound.
  int bound_port_ = 0;

  // The host name of this server
  string host_name_;

  // Guards server configuration, server, and state.
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
  State state_ TF_GUARDED_BY(mu_);

  // Implementation of a TensorFlow master, and RPC polling thread.
  MasterEnv master_env_;
  std::unique_ptr<Master> master_impl_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::unique_ptr<Thread> master_thread_ TF_GUARDED_BY(mu_);

  std::map<std::string, AsyncServiceInterface*> extra_services_;
  std::vector<std::unique_ptr<Thread>> extra_service_threads_
      TF_GUARDED_BY(mu_);

  // Implementation of a TensorFlow worker, and RPC polling thread.
  WorkerEnv worker_env_;
  std::unique_ptr<const DeviceMgr> owned_device_manager_;
  std::unique_ptr<GrpcWorker> worker_impl_;
  AsyncServiceInterface* worker_service_ = nullptr;
  std::unique_ptr<Thread> worker_thread_ TF_GUARDED_BY(mu_);
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env_;

  // TensorFlow Eager implementation, and RPC polling thread.
  AsyncServiceInterface* eager_service_ = nullptr;
  std::unique_ptr<Thread> eager_thread_ TF_GUARDED_BY(mu_);
  std::shared_ptr<WorkerSession> worker_session_;

  // TensorFlow profiler service implementation.
  std::unique_ptr<grpc::ProfilerService::Service> profiler_service_ = nullptr;

  // The overall server configuration.
  ServerDef server_def_ TF_GUARDED_BY(mu_);

  std::unique_ptr<::grpc::Server> server_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
