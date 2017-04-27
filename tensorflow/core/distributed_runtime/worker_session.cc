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

#include "tensorflow/core/distributed_runtime/worker_session.h"

namespace tensorflow {

WorkerSession::WorkerSession(
    const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    std::unique_ptr<RendezvousMgrInterface> rendezvous_mgr,
    std::unique_ptr<GraphMgr> graph_mgr)
    : worker_name(worker_name),
      worker_cache(std::move(worker_cache)),
      rendezvous_mgr(std::move(rendezvous_mgr)),
      graph_mgr(std::move(graph_mgr)) {}

}  // namespace tensorflow
