#include <fstream>
#include <map>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"
#include "grpc/support/alloc.h"

#include "tensorflow/contrib/seastar/seastar_channel_cache.h"
#include "tensorflow/contrib/seastar/seastar_engine.h"
#include "tensorflow/contrib/seastar/seastar_rendezvous_mgr.h"
#include "tensorflow/contrib/seastar/seastar_server_lib.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/contrib/seastar/seastar_worker_cache.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

const char* kEndpointMapFile = ".endpoint_map";

class SeastarPortMgr {
public:
  explicit SeastarPortMgr(const ServerDef& server_def) {
    ParseGrpcServerDef(server_def);
    LoadEndpointMapForFile();
  }

  std::string GetSeastarIpPort(const std::string& job_name, int task_index) {
    const auto it_job = grpc_cluster_spec_.find(job_name);
    if (it_job == grpc_cluster_spec_.end()) {
      LOG(FATAL) << "Job name: " << job_name
                 << " does not exist in cluster spec.";
    }
    const std::map<int, std::string>& task_map = it_job->second;

    const auto it_task = task_map.find(task_index);
    if (it_task == task_map.end()) {
      LOG(FATAL) << "Job name: " << job_name << ", task index: " << task_index
                 << " does not exist in cluster spec.";
    }
    const std::string& grpc_ip_port = it_task->second;

    const auto it_seastar = endpoint_grpc2seastar_.find(grpc_ip_port);
    if (it_seastar == endpoint_grpc2seastar_.end()) {
      LOG(FATAL) << "Seastar ip and port not found for job name: " << job_name
                 << "task index: " << task_index << ".";
    }

    return it_seastar->second;
  }

  int GetLocalSeastarPort() {
    const auto it = endpoint_grpc2seastar_.find(local_grpc_ip_port_);
    if (it == endpoint_grpc2seastar_.end()) {
      LOG(FATAL) << "Seastar ip and port not found for job name: " << job_name_
                 << "task index: " << task_index_ << ".";
    }
    const std::string& local_seastar_ip_port = it->second;
    std::vector<std::string> vec = str_util::Split(local_seastar_ip_port, ":");
    CHECK_EQ(vec.size(), 2);

    int local_seastar_port = -1;
    strings::safe_strto32(vec[1], &local_seastar_port);
    CHECK_GT(local_seastar_port, 0);

    return local_seastar_port;
  }

  std::string get_job_name() const {
    return job_name_;
  }

private:
  void ParseGrpcServerDef(const ServerDef& server_def) {
    job_name_ = server_def.job_name();
    task_index_ = server_def.task_index();

    for (const auto& job : server_def.cluster().job()) {
      std::map<int, std::string>& task_map = grpc_cluster_spec_[job.name()];
      for (const auto& task : job.tasks()) {
        task_map[task.first] = task.second;
        if (job.name() == job_name_ && task.first == task_index_) {
          local_grpc_ip_port_ = task.second;
        }
      }
    }

    if (local_grpc_ip_port_.empty()) {
      LOG(FATAL) << "Job name: " << job_name_ << ", task index: " << task_index_
                 << " not found in cluter spec.";
    }
  }

  void LoadEndpointMapForFile() {
    std::ifstream fin(kEndpointMapFile, std::ios::in);
    if (!fin.good()) {
      LOG(FATAL) << "Load endpoint map file failed.";
    }

    string str;
    while (getline(fin, str)) {
      std::vector<std::string> vec = str_util::Split(str, '=');
      CHECK_EQ(vec.size(), 2);
      endpoint_grpc2seastar_[vec[0]] = vec[1];
    }
  }

private:
  std::map<std::string, std::string> endpoint_grpc2seastar_;
  std::map<std::string, std::map<int, std::string> > grpc_cluster_spec_;
  std::string job_name_;
  int task_index_;
  std::string local_grpc_ip_port_;
};

SeastarServer::SeastarServer(const ServerDef& server_def, Env* env)
  : server_def_(server_def), env_(env), state_(NEW) {
  seastar_port_mgr_ = new SeastarPortMgr(server_def_);
}

SeastarServer::~SeastarServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete seastar_engine_;
  delete master_service_;
  delete worker_service_;

  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;
  } else {
    delete worker_env_.device_mgr;
  }

  delete seastar_port_mgr_;
}

Status SeastarServer::Init() {
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  string name_prefix =
    strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
        "/task:", server_def_.task_index());

  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(sess_opts, name_prefix,
        &devices));
  worker_env_.device_mgr = new DeviceMgr(std::move(devices));
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = new SeastarRendezvousMgr(&worker_env_);

  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  int requested_port = -1;
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

      if (hostname_port.size() != 2) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\"");
      }

      if (!strings::safe_strto32(hostname_port[1], &requested_port)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\"");
      }
      break;
    }
  }

  if (requested_port == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
        "\" was not defined in cluster");
  }

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
      GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(
      master_impl_.get(), config, &builder);

  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  LOG(INFO) << "starting grpc server, bind port:" << bound_port_;

  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  worker_impl_ = NewSeastarWorker(&worker_env_);
  worker_service_ = NewSeastarWorkerService(worker_impl_.get()).release();

  worker_env_.compute_pool = ComputePool(sess_opts);
  seastar_bound_port_ = seastar_port_mgr_->GetLocalSeastarPort();
  size_t server_number = ParseServers(worker_cache_factory_options);
  seastar_engine_ = new SeastarEngine(server_number, seastar_bound_port_,
                                      worker_service_);

  WorkerCacheInterface* worker_cache;
  TF_RETURN_IF_ERROR(
      SeastarWorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return SeastarWorkerCacheFactory(options, worker_cache);
      });

  // master intialize
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  StatsPublisherFactory stats_factory;
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
      return SeastarWorkerCacheFactory(options, worker_cache);
    };
  LocalMaster::Register(target(), master_impl_.get(),
      config.operation_timeout_in_ms());

  return Status::OK();
}

Status SeastarServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                       SeastarChannelSpec* channel_spec) {
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
        host_port = strings::StrCat("localhost:", seastar_bound_port_);
      } else {
        host_port = task.second;
        int grpc_port = -1;
        const std::vector<string> vec = str_util::Split(host_port, ':');
        if (vec.size() != 2 ||
            !strings::safe_strto32(vec[1], &grpc_port)) {
          LOG(ERROR) << "error host port schema " << host_port;
          return errors::Cancelled("error host port schema ", host_port);
        }

        std::string seastar_host_port
          = seastar_port_mgr_->GetSeastarIpPort(job.name(), task.first);
        LOG(INFO) << "host port: " << host_port
                  << ", remote seastar host port: " << seastar_host_port;
        host_port = seastar_host_port;
      }
    }

    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return Status::OK();
}

size_t SeastarServer::ParseServers(const WorkerCacheFactoryOptions& options) {
  size_t hosts_count = 0;
  for (const auto& job : options.cluster_def->job()) {
    hosts_count += job.tasks().size();
  }
  return hosts_count;
}

Status SeastarServer::SeastarWorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                                WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  SeastarChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));
  std::unique_ptr<SeastarChannelCache> channel_cache(
      NewSeastarChannelCache(seastar_engine_, channel_spec));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                           channel_cache->TranslateTask(name_prefix), "\".");
  }

  LOG(INFO) << "SeastarWorkerCacheFactory, name_prefix:" << name_prefix;
  *worker_cache = NewSeastarWorkerCacheWithLocalWorker(
      channel_cache.release(), worker_impl_.get(), name_prefix, &worker_env_);

  return Status::OK();
}

Status SeastarServer::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
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
      LOG(FATAL);
  }
  return Status::OK();
}

Status SeastarServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      LOG(WARNING) << "Clean shutdown is not currently implemented";
      server_->Shutdown();
      master_service_->Shutdown();
      state_ = STOPPED;
      return Status::OK();
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      LOG(FATAL);
  }
  return Status::OK();
}

Status SeastarServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      return Status::OK();
    default:
      LOG(FATAL);
  }
  return Status::OK();
}

const string SeastarServer::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> SeastarServer::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

std::unique_ptr<Master> SeastarServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

Status SeastarServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {

  std::unique_ptr<SeastarServer> ret(
      new SeastarServer(server_def, env == nullptr ? Env::Default() : env));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {
class SeastarServerFactory : public ServerFactory {
public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+seastar";
  }

  Status NewServer(const ServerDef& server_def,
      std::unique_ptr<ServerInterface>* out_server) override {
    return SeastarServer::Create(server_def, Env::Default(), out_server);
  }
};

class SeastarServerRegistrar {
public:
  SeastarServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("SEASTAR_SERVER", new SeastarServerFactory());
}
};

static SeastarServerRegistrar registrar;
}
} // namespace tensorflow
