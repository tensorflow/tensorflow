/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_
#define TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_

#ifdef TENSORFLOW_USE_VERBS

#include <string>
#include <unordered_map>

#include "tensorflow/contrib/verbs/rdma.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"

namespace tensorflow {

class RdmaMgr {
  friend class RdmaChannel;
  friend class RdmaAdapter;

 public:
  explicit RdmaMgr(const WorkerEnv* const worker_env,
                   GrpcChannelCache* const channel_cache);
  ~RdmaMgr();
  RdmaChannel* FindChannel(const string& key);
  void SetupChannels();
  bool ConnectivityCheck();
  void InitAllocators();
  const string& local_worker() { return local_worker_; }

 private:
  string local_worker_;
  size_t num_remote_workers_;
  const WorkerEnv* const worker_env_;
  GrpcChannelCache* const channel_cache_;
  RdmaAdapter* rdma_adapter_;
  typedef std::unordered_map<string, RdmaChannel*> ChannelTable;
  ChannelTable channel_table_;
  TF_DISALLOW_COPY_AND_ASSIGN(RdmaMgr);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_
