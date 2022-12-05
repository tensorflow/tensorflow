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

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/distributed_runtime/rpc/async_service_interface.h"

namespace tensorflow {

namespace {

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

// Define an option subclass in order to enable SO_REUSEPORT for the
// server socket.
class ReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 1);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

// static utility function
RendezvousMgrInterface* NewRpcRendezvousMgr(const WorkerEnv* env) {
  return new RpcRendezvousMgr(env);
}

}  // namespace

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : env_(env), state_(NEW), server_def_(server_def) {}

GrpcServer::~GrpcServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete master_service_;
  delete worker_service_;
  delete eager_service_;

  for (auto& kv : extra_services_) {
    tsl::AsyncServiceInterface* service = kv.second;
    delete service;
  }

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.

  // Shut down all outstanding rendezvous.
  delete worker_env_.rendezvous_mgr;

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  }

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

// Look up the requested host name and port for this task in `server_def`.
Status GrpcServer::GetHostAndPort(const ServerDef& server_def,
                                  string* host_name, int* port) const {
  *port = -1;
  *host_name = "localhost";
  for (const auto& job : server_def.cluster().job()) {
    if (job.name() == server_def.job_name()) {
      auto iter = job.tasks().find(server_def.task_index());
      if (iter == job.tasks().end()) {
        return errors::Internal("Task ", server_def.task_index(),
                                " was not defined in job \"",
                                server_def.job_name(), "\"");
      }

      if (server_def.port() != 0) {
        *port = server_def.port();
      } else {
        auto colon_index = iter->second.find_last_of(':');
        if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                   port)) {
          return errors::InvalidArgument(
              "Could not parse port for local server from \"", iter->second,
              "\".");
        }

        if (colon_index != string::npos &&
            !iter->second.substr(0, colon_index).empty()) {
          *host_name = iter->second.substr(0, colon_index);
        }
      }
      break;
    }
  }
  if (*port == -1) {
    return errors::Internal("Job \"", server_def.job_name(),
                            "\" was not defined in cluster");
  }

  return OkStatus();
}

Status GrpcServer::Init(const GrpcServerOptions& opts) {
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  // Check parameters before DeviceFactory::AddDevices,
  // otherwise if 'task_index=-1' the program will abort.

  int requested_port;
  TF_RETURN_IF_ERROR(GetHostAndPort(server_def_, &host_name_, &requested_port));

  SessionOptions sess_opts;
  VLOG(3) << "Grpc Server Init Definition: " << server_def_.DebugString();
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  if (opts.local_device_mgr == nullptr) {
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(
        DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
    worker_env_.device_mgr = new DynamicDeviceMgr(std::move(devices));
    owned_device_manager_.reset(worker_env_.device_mgr);
  } else {
    worker_env_.device_mgr = opts.local_device_mgr;
    owned_device_manager_.reset(nullptr);
  }
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();

  int num_tasks = 0;
  for (auto& job : server_def_.cluster().job()) {
    num_tasks += job.tasks_size();
  }
  master_env_.experimental_num_shards = std::max(1, num_tasks);
  worker_env_.experimental_num_shards = master_env_.experimental_num_shards;

  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // N.B. The order of initialization here is intricate, because we
  // wish to allow `requested_port == 0` (for choosing any port,
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
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  bool reuse_port = false;
  const Status status =
      ReadBoolFromEnvVar("TF_GRPC_REUSE_PORT", false, &reuse_port);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  auto server_build_option =
      reuse_port
          ? std::unique_ptr<::grpc::ServerBuilderOption>(new ReusePortOption)
          : std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption);
  builder.SetOption(std::move(server_build_option));

  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder, requested_port);
  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);
  thread::ThreadPool* compute_pool = ComputePool(sess_opts);
  coordination_service_ =
      new GrpcCoordinationServiceImpl(compute_pool, &builder);

  profiler_service_ = profiler::CreateProfilerService();
  builder.RegisterService(profiler_service_.get());

  // Add any extra services to be started.
  extra_services_ = ExtraServices(&builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }
  // Create the execution environment for the GRPC workers cache.
  grpc_worker_env_.reset(CreateGrpcWorkerEnv());

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr.reset(
        opts.collective_mgr_func(config, &worker_env_, worker_cache));
    if (worker_env_.collective_executor_mgr == nullptr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    worker_env_.collective_executor_mgr = CreateProdRpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, MaybeCreateNcclCommunicator(config),
        worker_cache, default_worker_name);
  }

  auto* grpc_coordination_service =
      static_cast<GrpcCoordinationServiceImpl*>(coordination_service_);
  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      },
      grpc_coordination_service->GetRpcHandler());
  worker_env_.compute_pool = compute_pool;

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr =
      worker_env_.collective_executor_mgr.get();
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  return OkStatus();
}

Status GrpcServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                    GrpcChannelSpec* channel_spec) {
  for (const auto& job : options.cluster_def->job()) {
    std::map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument("JobDef for job \"", job.name(),
                                       "\" specified two addresses for task \"",
                                       task.first, "\": ", host_port, " and ",
                                       task.second);
      }
      if (job.name() == *options.job_name && task.first == options.task_index) {
        host_port = strings::StrCat(host_name_, ":", bound_port_);
      } else {
        host_port = task.second;
      }
    }
    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return OkStatus();
}

Status GrpcServer::WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                      WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  if (options.rpc_options == nullptr) {
    return errors::InvalidArgument(
        "rpc_options not set in WorkerCacheFactoryOptions");
  }
  std::shared_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(
      channel_spec, GetChannelCreationFunction(), *options.rpc_options));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }
  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }
  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(
      channel_cache, grpc_worker_env(), worker_impl(), name_prefix);
  return OkStatus();
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
      eager_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_eager_service",
                            [this] { eager_service_->HandleRPCsLoop(); }));
      coordination_thread_.reset(env_->StartThread(
          ThreadOptions(), "TF_coordination_service",
          [this] { coordination_service_->HandleRPCsLoop(); }));

      for (const auto& kv : extra_services_) {
        const std::string& service_name = kv.first;
        tsl::AsyncServiceInterface* service = kv.second;
        std::unique_ptr<Thread> extra_service_thread;
        extra_service_thread.reset(env_->StartThread(
            ThreadOptions(), service_name,
            [service = service] { service->HandleRPCsLoop(); }));
        extra_service_threads_.push_back(std::move(extra_service_thread));
        VLOG(3) << "Started extra service: " << service_name;
      }

      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return OkStatus();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return OkStatus();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::AddMasterEagerContextToEagerService(
    const tensorflow::uint64 context_id, tensorflow::EagerContext* context) {
  auto* eager_service =
      static_cast<eager::GrpcEagerServiceImpl*>(eager_service_);
  return eager_service->CreateMasterContext(context_id, context);
}

Status GrpcServer::UpdateServerDef(const ServerDef& server_def) {
  mutex_lock l(mu_);
  server_def_ = server_def;
  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  if (worker_cache == nullptr) {
    return errors::InvalidArgument(
        "Failed to build worker cache with the provided server def.");
  }
  // Transfer ownership of worker_cache to worker_env_.session_mgr.
  worker_env_.session_mgr->ResetDefaultWorkerCache(worker_cache);

  string default_worker_name;
  string unused;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }
  worker_env_.collective_executor_mgr = CreateProdRpcCollectiveExecutorMgr(
      server_def_.default_session_config(), worker_env_.device_mgr,
      MaybeCreateNcclCommunicator(server_def_.default_session_config()),
      worker_cache, default_worker_name);

  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr =
      worker_env_.collective_executor_mgr.get();
  return OkStatus();
}

// TODO(haoyuzhang): Remove this method once we have a mechanism to directly set
// field inside the RPC coordination service handler.
Status GrpcServer::SetCoordinationServiceAgentInstance(
    tsl::CoordinationServiceAgent* agent) {
  auto* coord_service =
      static_cast<GrpcCoordinationServiceImpl*>(coordination_service_);
  coord_service->SetCoordinationServiceAgentInstance(agent);
  return OkStatus();
}

Status GrpcServer::SetCoordinationServiceInstance(
    tsl::CoordinationServiceInterface* service) {
  auto* coord_service =
      static_cast<GrpcCoordinationServiceImpl*>(coordination_service_);
  coord_service->SetCoordinationServiceInstance(service);
  return OkStatus();
}

Status GrpcServer::StopCoordinationService() {
  // Note: the sequence of events is important here.
  // 1. Agent must be torn down before the service as it needs to notify the
  // service.
  // 2. Remove RPC handlers' access to agent/service first before destructing
  // them within the session manager to prevent data races.
  TF_RETURN_IF_ERROR(SetCoordinationServiceAgentInstance(nullptr));
  worker_env()->session_mgr->TeardownCoordinationServiceAgent();
  TF_RETURN_IF_ERROR(SetCoordinationServiceInstance(nullptr));
  coordination_service_->Shutdown();
  worker_env()->session_mgr->TeardownCoordinationService();
  return OkStatus();
}

Status GrpcServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return OkStatus();
    case STARTED:
      return errors::Unimplemented(
          "Clean shutdown is not currently implemented");
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return OkStatus();
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return OkStatus();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      eager_thread_.reset();
      for (auto& thread : extra_service_threads_) {
        thread.reset();
      }
      return OkStatus();
    default:
      LOG(FATAL);
  }
}

const string GrpcServer::target() const {
  return strings::StrCat("grpc://", host_name_, ":", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  // We can do this because SparseGrpcChannelCache is robust to nullptr being
  // returned by the channel creation function
  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
}

std::unique_ptr<Master> GrpcServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          DeviceMgr* local_device_mgr,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  options.local_device_mgr = local_device_mgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return OkStatus();
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  return Create(server_def, env, nullptr, out_server);
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<GrpcServer>* out_server) {
  std::unique_ptr<ServerInterface> server;
  Status s = Create(server_def, env, nullptr, &server);
  if (!s.ok()) {
    return s;
  }
  out_server->reset(dynamic_cast<GrpcServer*>(server.release()));
  return OkStatus();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc";
  }

  Status NewServer(const ServerDef& server_def, const Options& options,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return GrpcServer::Create(server_def, Env::Default(),
                              options.local_device_mgr, out_server);
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
