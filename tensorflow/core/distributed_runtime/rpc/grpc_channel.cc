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

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <limits>
#include <unordered_map>

#include "grpc++/create_channel.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
RE2* kTargetRE = new RE2("^/job:([^/]+)/replica:([0-9]+)/task:([0-9]+)$");
RE2* kHostPortRE = new RE2("([^:/]+):(\\d+)");
RE2* kSparseHostPortRE = new RE2("(\\d+):([^:/]+):(\\d+)");

string MakeAddress(const string& job, int replica, int task) {
  return strings::StrCat("/job:", job, "/replica:", replica, "/task:", task);
}

}  // namespace

SharedGrpcChannelPtr NewHostPortGrpcChannel(const string& target) {
  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  return ::grpc::CreateCustomChannel(
      target, ::grpc::InsecureChannelCredentials(), args);
}

Status GrpcChannelSpec::AddHostPortsJob(const string& job_id,
                                        const std::vector<string>& host_ports,
                                        int tasks_per_replica) {
  if (!job_ids_.insert(job_id).second) {
    return errors::InvalidArgument(
        "Duplicate job ID in cluster specification: ", job_id);
  }
  HostPortsJob job;
  job.job_id = job_id;
  for (const string& host_port : host_ports) {
    string host;
    int port;
    if (!RE2::FullMatch(host_port, *kHostPortRE, &host, &port)) {
      return errors::InvalidArgument("Could not interpret \"", host_port,
                                     "\" as a host-port pair.");
    }
  }
  job.host_ports = host_ports;
  job.tasks_per_replica = tasks_per_replica;
  host_ports_jobs_.push_back(job);
  return Status::OK();
}

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (const GrpcChannelSpec::HostPortsJob& job : spec.host_ports_jobs()) {
    caches.push_back(NewHostPortsGrpcChannelCache(
        job.job_id, job.host_ports, job.tasks_per_replica, channel_func));
  }
  return caches.size() == 1 ? caches[0] : NewMultiGrpcChannelCache(caches);
}

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

  void ListWorkers(std::vector<string>* workers) override {
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

GrpcChannelCache* NewMultiGrpcChannelCache(
    const std::vector<GrpcChannelCache*>& caches) {
  return new MultiGrpcChannelCache(caches);
}

class HostPortsGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  HostPortsGrpcChannelCache(const string& job_id,
                            const std::vector<string>& host_ports,
                            int tasks_per_replica,
                            ChannelCreationFunction channel_func)
      : job_id_(job_id),
        host_ports_(BuildDenseHostPortsList(host_ports, tasks_per_replica)),
        tasks_per_replica_(tasks_per_replica),
        channel_func_(channel_func) {
    LOG(INFO) << "Initialize HostPortsGrpcChannelCache for job " << job_id
              << " -> {" << str_util::Join(host_ports, ", ") << "}";
  }
  ~HostPortsGrpcChannelCache() override {}

  void ListWorkers(std::vector<string>* workers) override {
    int num_host_ports = 0;
    for (size_t i = 0; i < host_ports_.size(); ++i) {
      if (!host_ports_[i].empty()) {
        ++num_host_ports;
      }
    }
    workers->reserve(workers->size() + num_host_ports);
    for (size_t i = 0; i < host_ports_.size(); ++i) {
      if (!host_ports_[i].empty()) {
        workers->emplace_back(MakeAddress(job_id_, i / tasks_per_replica_,
                                          i % tasks_per_replica_));
      }
    }
  }

  string TranslateTask(const string& target) override {
    RegexpStringPiece job;
    int32 replica;
    int32 task;
    if (!RE2::FullMatch(target, *kTargetRE, &job, &replica, &task)) {
      LOG(WARNING) << "Invalid target: " << target;
      return "";
    }
    if (job != job_id_) {
      return "";
    }
    if (task >= tasks_per_replica_) {
      LOG(WARNING) << "Task out of bounds for job " << job_id_ << ": " << task;
      return "";
    }
    const size_t i = replica * tasks_per_replica_ + task;
    if (i >= host_ports_.size()) {
      LOG(WARNING) << "Replica/task out of bounds for job " << job_id_ << ": "
                   << target;
      return "";
    }
    if (host_ports_[i].empty()) {
      LOG(WARNING) << "Replica/task not in sparse index:host:port list for job "
                   << job_id_ << ": " << target;
      return "";
    }
    return host_ports_[i];
  }

 protected:
  static std::vector<string> BuildDenseHostPortsList(
      const std::vector<string>& host_ports, int tasks_per_replica) {
    std::map<int, string> sparse_host_ports;
    for (const string& host_port : host_ports) {
      int i = -1;
      string host;
      int port = -1;
      if (RE2::FullMatch(host_port, *kSparseHostPortRE, &i, &host, &port)) {
        CHECK_LE(0, i);
        CHECK_LE(0, port);
        CHECK(sparse_host_ports.find(i) == sparse_host_ports.end())
            << "Duplicate index " << i << ": {"
            << str_util::Join(host_ports, ", ") << "}";
        sparse_host_ports[i] = strings::StrCat(host, ":", port);
      } else {
        CHECK(RE2::FullMatch(host_port, *kHostPortRE, &host, &port))
            << host_port
            << " does not look like a host:port or an index:host:port";
      }
    }

    if (sparse_host_ports.empty()) {
      // The input is a dense list; return it directly.
      return host_ports;
    }

    // The input is a sparse list. Convert it to a dense list.
    CHECK_EQ(host_ports.size(), sparse_host_ports.size())
        << "Mix of host:port and index:host:port: {"
        << str_util::Join(host_ports, ", ") << "}";
    int num_tasks = sparse_host_ports.rbegin()->first + 1;
    if (num_tasks % tasks_per_replica != 0) {
      num_tasks = (num_tasks / tasks_per_replica + 1) * tasks_per_replica;
    }
    std::vector<string> dense_host_ports;
    dense_host_ports.resize(num_tasks);
    for (const auto& p : sparse_host_ports) {
      dense_host_ports[p.first] = p.second;
    }
    return dense_host_ports;
  }

  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }
    return channel_func_(host_port);
  }

 private:
  const string job_id_;
  const std::vector<string> host_ports_;
  const int tasks_per_replica_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(HostPortsGrpcChannelCache);
};

GrpcChannelCache* NewHostPortsGrpcChannelCache(
    const string& job_id, const std::vector<string>& host_ports,
    int tasks_per_replica, ChannelCreationFunction channel_func) {
  return new HostPortsGrpcChannelCache(job_id, host_ports, tasks_per_replica,
                                       channel_func);
}

}  // end namespace tensorflow
