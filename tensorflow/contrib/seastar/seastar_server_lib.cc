#include <fstream>
#include <map>
#include "grpc/support/alloc.h"
#include "tensorflow/contrib/seastar/seastar_channel_cache.h"
#include "tensorflow/contrib/seastar/seastar_engine.h"
#include "tensorflow/contrib/seastar/seastar_rendezvous_mgr.h"
#include "tensorflow/contrib/seastar/seastar_server_lib.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/contrib/seastar/seastar_worker_cache.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

namespace {
const char* kEndpointMapFile = ".endpoint_map";
} // namespace

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
    const auto& task_map = it_job->second;

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
      auto& task_map = grpc_cluster_spec_[job.name()];
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
  std::unordered_map<std::string, std::string> endpoint_grpc2seastar_;
  std::unordered_map<std::string,
      std::unordered_map<int, std::string>> grpc_cluster_spec_;
  std::string job_name_;
  int task_index_;
  std::string local_grpc_ip_port_;
};

SeastarServer::SeastarServer(const ServerDef& server_def, Env* env)
  : GrpcServer(server_def, env) {
  seastar_port_mgr_ = new SeastarPortMgr(server_def);
}

SeastarServer::~SeastarServer() {
  delete seastar_worker_service_;
  delete seastar_engine_;
  delete seastar_port_mgr_;
}

Status SeastarServer::Init() {
  seastar_worker_impl_ = NewSeastarWorker(worker_env());
  seastar_worker_service_ =
    NewSeastarWorkerService(seastar_worker_impl_.get()).release();
  seastar_bound_port_ = seastar_port_mgr_->GetLocalSeastarPort();
  seastar_engine_ = new SeastarEngine(seastar_bound_port_,
                                      seastar_worker_service_);

  RendezvousMgrCreationFunction rendezvous_mgr_func =
      [this](const WorkerEnv* env) {
        return new SeastarRendezvousMgr(env);
      };

  GrpcServerOptions opts;
  opts.rendezvous_mgr_func = rendezvous_mgr_func;
  return GrpcServer::Init(opts);
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

Status SeastarServer::WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
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
      channel_cache.release(), seastar_worker_impl_.get(), name_prefix, worker_env());

  return Status::OK();
}

Status SeastarServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<SeastarServer> ret(
      new SeastarServer(server_def, env == nullptr ? Env::Default() : env));
  Status s = ret->Init();
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
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
