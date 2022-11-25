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

#include "tensorflow/tsl/distributed_runtime/rpc/grpc_channel.h"

#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>

#include "grpcpp/create_channel.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_channel_common.h"
#include "tensorflow/tsl/lib/gtl/map_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/numbers.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/str_util.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/thread_annotations.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/protobuf/rpc_options.pb.h"
#include "tensorflow/tsl/util/device_name_utils.h"

namespace tsl {

namespace {

string MakeAddress(const string& job, int task) {
  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}

// Allows the host to be a raw IP (either v4 or v6).
Status ValidateHostPortPair(const string& host_port) {
  string bns_prefix = "/bns/";
  if (host_port.substr(0, bns_prefix.length()) == bns_prefix) {
    return OkStatus();
  }
  uint32 port;
  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strtou32(host_port.substr(colon_index + 1), &port) ||
      host_port.substr(0, colon_index).find('/') != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return OkStatus();
}

::grpc::ChannelArguments* CreateDefaultChannelArguments() {
  ::grpc::ChannelArguments* args = new ::grpc::ChannelArguments();
  const char* env = std::getenv("TF_GRPC_DEFAULT_OPTIONS");
  if (env != nullptr) {
    for (auto& grpc_option : absl::StrSplit(env, ',')) {
      std::vector<string> name_value = absl::StrSplit(grpc_option, '=');
      if (name_value.size() != 2) {
        LOG(ERROR) << "Invalid GRPC options format: " << grpc_option;
        continue;
      }
      VLOG(3) << "Setting GRPC default for '" << name_value[0] << "' to '"
              << name_value[1] << "'";
      if (name_value[1].size() >= 2 && name_value[1][0] == '"') {
        string ue_value = name_value[1].substr(1, name_value[1].size() - 2);
        string value;
        string error;
        if (!absl::CUnescape(ue_value, &value, &error)) {
          LOG(ERROR) << "Failed to parse escaped string for " << grpc_option
                     << ": " << error;
        } else {
          args->SetString(name_value[0], value);
        }
      } else {
        int64_t value;
        if (strings::safe_strto64(name_value[1], &value)) {
          args->SetInt(name_value[0], value);
        } else {
          LOG(ERROR) << "Invalid integer value: " << grpc_option;
        }
      }
    }
  }
  return args;
}

const ::grpc::ChannelArguments* GetDefaultChannelArguments() {
  static const ::grpc::ChannelArguments* args = CreateDefaultChannelArguments();
  return args;
}

}  // namespace

::grpc::ChannelArguments GetChannelArguments(const RPCOptions* rpc_options) {
  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args = *GetDefaultChannelArguments();
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  // NOTE(mrry): Some versions of gRPC use a 20-second minimum backoff
  // on connection failure, which makes our tests time out.
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
  if (rpc_options != nullptr) {
    if (rpc_options->compression_algorithm() == "deflate") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_DEFLATE);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (rpc_options->compression_algorithm() == "gzip") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (!rpc_options->compression_algorithm().empty()) {
      LOG(ERROR) << "Invalid compression algorithm: "
                 << rpc_options->compression_algorithm();
    }
    if (rpc_options->disable_session_connection_sharing()) {
      VLOG(5) << "Disabling TCP connection sharing";
      args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    }
  }
  return args;
}

Status NewHostPortGrpcChannel(const string& target,
                              const RPCOptions* rpc_options,
                              SharedGrpcChannelPtr* channel_pointer) {
  // Minimally ensure that the target is valid
  TF_RETURN_IF_ERROR(ValidateHostPortPair(target));

  ::grpc::ChannelArguments args = GetChannelArguments(rpc_options);
  *channel_pointer = ::grpc::CreateCustomChannel(
      "dns:///" + target, ::grpc::InsecureChannelCredentials(), args);
  return OkStatus();
}

ChannelCreationFunction ConvertToChannelCreationFunction(
    const std::function<Status(string, const RPCOptions*,
                               SharedGrpcChannelPtr*)>& new_channel_func_ptr) {
  return [new_channel_func_ptr](const string& target) -> SharedGrpcChannelPtr {
    SharedGrpcChannelPtr channel_ptr;
    if (new_channel_func_ptr(target, /*rpc_options=*/nullptr, &channel_ptr)
            .ok()) {
      return channel_ptr;
    } else {
      return nullptr;
    }
  };
}

Status GrpcChannelSpec::AddHostPortsJob(const string& job_id,
                                        const std::vector<string>& host_ports) {
  std::map<int, string> host_ports_map;
  for (size_t i = 0; i < host_ports.size(); ++i) {
    host_ports_map[i] = host_ports[i];
  }
  return AddHostPortsJob(job_id, host_ports_map);
}

Status GrpcChannelSpec::AddHostPortsJob(
    const string& job_id, const std::map<int, string>& host_ports) {
  if (!job_ids_.insert(job_id).second) {
    return errors::InvalidArgument(
        "Duplicate job ID in cluster specification: ", job_id);
  }
  for (const auto& id_host_port : host_ports) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(id_host_port.second));
  }
  host_ports_jobs_.emplace_back(job_id, host_ports);
  return OkStatus();
}

namespace {

// GrpcChannelCache that caches results to FindWorkerChannel() calls.
using CachingGrpcChannelCache = GenericCachingChannelCache<GrpcChannelCache>;

// A ChannelCache that is the union of multiple ChannelCaches.
// Takes ownership of the caches passed to the constructor.
class MultiGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  explicit MultiGrpcChannelCache(const std::vector<GrpcChannelCache*>& caches,
                                 int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target), caches_(caches) {}

  ~MultiGrpcChannelCache() override {
    for (GrpcChannelCache* cache : caches_) {
      delete cache;
    }
  }

  void ListWorkers(std::vector<string>* workers) override {
    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkers(workers);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkersInJob(job_name, workers);
    }
  }

  string TranslateTask(const string& target) override {
    mutex_lock l(mu_);  // could use reader lock
    GrpcChannelCache* cache = gtl::FindPtrOrNull(target_caches_, target);
    if (cache == nullptr) {
      for (GrpcChannelCache* c : caches_) {
        string r = c->TranslateTask(target);
        if (!r.empty()) {
          target_caches_.insert({target, c});
          cache = c;
          break;
        }
      }
    }
    CHECK(cache) << "Could not find GrpcChannelCache holding channel for "
                 << target;
    return cache->TranslateTask(target);
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    for (GrpcChannelCache* cache : caches_) {
      SharedGrpcChannelPtr ch(cache->FindWorkerChannel(target));
      if (ch) {
        mutex_lock l(mu_);
        target_caches_.insert({target, cache});
        return ch;
      }
    }
    return nullptr;
  }

 private:
  // List of channels used by this MultiGrpcChannelCache.
  const std::vector<GrpcChannelCache*> caches_;

  mutex mu_;
  // Cache of channels keyed by the target they are handling.
  // The same GrpcChannelCache can appear multiple times in the cache.
  std::unordered_map<string, GrpcChannelCache*> target_caches_
      TF_GUARDED_BY(mu_);
};

class SparseGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  SparseGrpcChannelCache(const string& job_id,
                         const std::map<int, string>& host_ports,
                         ChannelCreationFunction channel_func,
                         int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target),
        job_id_(job_id),
        host_ports_(host_ports),
        channel_func_(std::move(channel_func)) {
    VLOG(2) << "Initialize GrpcChannelCache for job " << ToString();
  }
  ~SparseGrpcChannelCache() override {}

  void ListWorkers(std::vector<string>* workers) override {
    workers->reserve(workers->size() + host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      workers->emplace_back(MakeAddress(job_id_, id_host_port.first));
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    if (job_name == job_id_) {
      ListWorkers(workers);
    }
  }

  string TranslateTask(const string& target) override {
    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(target, &parsed)) {
      LOG(WARNING) << "Invalid target: " << target;
      return "";
    }

    if (!parsed.has_job || parsed.job != job_id_) {
      return "";
    }
    if (!parsed.has_replica || parsed.replica != 0) {
      LOG(WARNING) << "Replica ID must be 0 in target: " << target;
      return "";
    }
    int32_t task = parsed.has_task ? parsed.task : -1;
    auto iter = host_ports_.find(task);
    if (iter == host_ports_.end()) {
      LOG(WARNING) << "Task " << task << " was not defined in sparse job "
                   << job_id_ << ": " << target;
      return "";
    }
    return iter->second;
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }
    auto chan_ptr = channel_func_(host_port);
    VLOG(5) << "Channel created for: job: " << job_id_
            << " host_port: " << host_port << " target : " << target
            << " Ptr: " << chan_ptr.get();
    return chan_ptr;
  }

 private:
  string ToString() {
    std::vector<string> task_strings;
    task_strings.reserve(host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", absl::StrJoin(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, string> host_ports_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseGrpcChannelCache);
};

}  // namespace

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func,
                                      const RPCOptions& options) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    VLOG(2) << "Creating Grpc Channel Cache for: " << job.job_id;
    caches.push_back(
        new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func,
                                   options.num_channels_per_target()));
  }
  return caches.size() == 1 ? caches[0]
                            : new MultiGrpcChannelCache(
                                  caches, options.num_channels_per_target());
}

}  // namespace tsl
