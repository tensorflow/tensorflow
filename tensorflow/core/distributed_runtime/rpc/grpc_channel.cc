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

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <limits>
#include <map>
#include <unordered_map>

#include "grpc++/create_channel.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

string MakeAddress(const string& job, int task) {
  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}

Status ValidateHostPortPair(const string& host_port) {
  uint32 port;
  std::vector<string> parts = str_util::Split(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/'.
  if (parts.size() != 2 || !strings::safe_strtou32(parts[1], &port) ||
      parts[0].find("/") != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}
}  // namespace

Status NewHostPortGrpcChannel(const string& target,
                              SharedGrpcChannelPtr* channel_pointer) {
  // Minimally ensure that the target is valid
  TF_RETURN_IF_ERROR(ValidateHostPortPair(target));

  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  // NOTE(mrry): Some versions of gRPC use a 20-second minimum backoff
  // on connection failure, which makes our tests time out.
  args.SetInt("grpc.testing.fixed_reconnect_backoff_ms", 1000);
  *channel_pointer = ::grpc::CreateCustomChannel(
      target, ::grpc::InsecureChannelCredentials(), args);
  return Status::OK();
}

ChannelCreationFunction ConvertToChannelCreationFunction(
    const std::function<Status(string, SharedGrpcChannelPtr*)>&
        new_channel_func_ptr) {
  return [new_channel_func_ptr](const string& target) -> SharedGrpcChannelPtr {
    SharedGrpcChannelPtr channel_ptr;
    if (new_channel_func_ptr(target, &channel_ptr).ok()) {
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
  return Status::OK();
}

namespace {

// GrpcChannelCache that caches results to FindWorkerChannel() calls.
class CachingGrpcChannelCache : public GrpcChannelCache {
 public:
  CachingGrpcChannelCache() {}

  ~CachingGrpcChannelCache() override {}

  SharedGrpcChannelPtr FindWorkerChannel(const string& target) override {
    SharedGrpcChannelPtr ch = nullptr;
    {
      mutex_lock l(mu_);  // could use reader lock
      ch = gtl::FindPtrOrNull(channels_, target);
      if (ch) {
        return ch;
      }
    }
    ch = FindChannelOnce(target);
    if (ch) {
      mutex_lock l(mu_);
      channels_.insert({target, ch});
    }
    return ch;
  }

 protected:
  // Find the ClientChannel for "target".  Only called when no channel was
  // found in the channels_ cache for "target".  A non nullptr result will be
  // cached in channels_.
  virtual SharedGrpcChannelPtr FindChannelOnce(const string& target) = 0;

 private:
  // TODO(zhifengc): Eviction when the map becomes too big.
  mutex mu_;
  std::unordered_map<string, SharedGrpcChannelPtr> channels_ GUARDED_BY(mu_);
};

// A ChannelCache that is the union of multiple ChannelCaches.
// Takes ownership of the caches passed to the constructor.
class MultiGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  explicit MultiGrpcChannelCache(const std::vector<GrpcChannelCache*>& caches)
      : CachingGrpcChannelCache(), caches_(caches) {}

  ~MultiGrpcChannelCache() override {
    for (GrpcChannelCache* cache : caches_) {
      delete cache;
    }
  }

  void ListWorkers(std::vector<string>* workers) const override {
    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkers(workers);
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
  std::unordered_map<string, GrpcChannelCache*> target_caches_ GUARDED_BY(mu_);
};

class SparseGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  SparseGrpcChannelCache(const string& job_id,
                         const std::map<int, string>& host_ports,
                         ChannelCreationFunction channel_func)
      : job_id_(job_id),
        host_ports_(host_ports),
        channel_func_(std::move(channel_func)) {
    LOG(INFO) << "Initialize GrpcChannelCache for job " << ToString();
  }
  ~SparseGrpcChannelCache() override {}

  void ListWorkers(std::vector<string>* workers) const override {
    workers->reserve(workers->size() + host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      workers->emplace_back(MakeAddress(job_id_, id_host_port.first));
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
    int32 task = parsed.has_task ? parsed.task : -1;
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
    return channel_func_(host_port);
  }

 private:
  string ToString() {
    std::vector<string> task_strings;
    task_strings.reserve(host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", str_util::Join(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, string> host_ports_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseGrpcChannelCache);
};

}  // namespace

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    caches.push_back(
        new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func));
  }
  return caches.size() == 1 ? caches[0] : new MultiGrpcChannelCache(caches);
}

}  // end namespace tensorflow
