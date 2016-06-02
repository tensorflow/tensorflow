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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "grpc++/grpc++.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace tensorflow {

// Consolidated parameter structure to ease use of generic interfaces.
//
// Each job_id requires:
// - a list of host:port (or sparse list of index:host:port)
// - the number of tasks per replica
class GrpcChannelSpec {
 public:
  struct HostPortsJob {
    string job_id;
    std::vector<string> host_ports;
    int tasks_per_replica;
  };

  Status AddHostPortsJob(const string& job_id,
                         const std::vector<string>& host_ports,
                         int tasks_per_replica);

  const std::vector<HostPortsJob>& host_ports_jobs() const {
    return host_ports_jobs_;
  }

 private:
  std::vector<HostPortsJob> host_ports_jobs_;
  std::set<string> job_ids_;
};

class GrpcChannelCache {
 public:
  virtual ~GrpcChannelCache() {}

  // Populates *workers with names of all workers which this object
  // was created to handle.  Worker names are in the format
  //  /job:<job identifier>/task:<task id>
  // e.g. /job:mnist/task:2
  virtual void ListWorkers(std::vector<string>* workers) = 0;

  // If found, returns a gRPC channel that is connected to the remote
  // worker named by 'target'. 'target' is of the following
  // format: /job:<job identifier>/task:<task id>
  // E.g., /job:mnist/task:2
  virtual SharedGrpcChannelPtr FindWorkerChannel(const string& target) = 0;

  // Translates a string in the form `/job:X/task:Z` into a host_port.
  virtual string TranslateTask(const string& task) = 0;
};

typedef std::function<SharedGrpcChannelPtr(string)> ChannelCreationFunction;

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& p,
                                      ChannelCreationFunction channel_func);

// Below here are internal-only functions.

SharedGrpcChannelPtr NewHostPortGrpcChannel(const string& target);

// Returns a ChannelCache that uses a set of known host:port pairs. E.g., say,
// job_id = 'mnist', 'host_ports' = {"h0:0", "h1:1", ..., "h11:11", "h12:12"},
// tasks_per_replica = 8, /job:mnist/replica:1/task:3 is mapped to host:port
// "h11:11" (11 = 8 * 1 + 3).
//
// The caller takes ownership of the returned object.
GrpcChannelCache* NewHostPortsGrpcChannelCache(
    const string& job_id, const std::vector<string>& host_ports,
    int tasks_per_replica, ChannelCreationFunction channel_func);

// Returns a ChannelCache that is the union of a number of other ChannelCaches.
GrpcChannelCache* NewMultiGrpcChannelCache(
    const std::vector<GrpcChannelCache*>& caches);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
