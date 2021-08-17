/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RESPONSE_CACHE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RESPONSE_CACHE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/mutex.h"

// gRPC response caching.  Most WorkerService methods cannot be retried directly
// as they will fail or deadlock.  To enable retrying, we can instead cache
// responses and reply to duplicate requests from the cache. The cache will be
// cleaned when the MarkRecvFinishedRequest is received from the receiver or the
// session step is completed.
namespace tensorflow {

// Track and cache the state of worker service RPCs.  An RPC can be in 3 states:
//
// * PENDING: this is the first call of the RPC, and it will transition to
// * ACTIVE: another thread is active processing this RPC
// * FINISHED: the worker has finished processing the method

class GrpcResponseCache {
 public:
  using FinishResponseCB = std::function<void(
      const Tensor& tensor, bool is_dead, const Status& status)>;

  // Add the given request to the cache.
  // If the request is in the cache,
  //    If it is finished, invoke `cb` immediately
  //    If active, cb will be invoked when the current call completes.
  //    In either case, return true.
  // Otherwise, store the request and cb in the cache, and return false.
  // Note FinishResponseCB is assumed to be thread-safe.
  bool QueueRequest(int64_t request_id, int64_t step_id,
                    const FinishResponseCB& cb);

  // Fill the response cache for the given request_id and respond to all
  // pending request.
  void OnRequestFinished(int64_t request_id, const Tensor& tensor, bool is_dead,
                         const Status& status);

  // Erase the cache entry with the given request_id
  void EraseRequestId(int64_t request_id);

  // Erase cache entries with the given step_id
  void CleanEntriesForStep(int64_t step_id);

 private:
  struct ResponseCacheEntry {
    enum class State {
      PENDING = 0,
      ACTIVE = 1,
      FINISHED = 2,
    };

    State state = State::PENDING;
    int64_t step_id = -1;
    Tensor tensor;
    bool is_dead = false;
    Status response_status;

    void FinishResponse(const FinishResponseCB& cb) const {
      cb(tensor, is_dead, response_status);
    }
    std::vector<FinishResponseCB> callbacks;
  };

  mutex mu_;
  // response_cache_ is expected to be small, as entries are cleared immediately
  // on ack from the receiver.
  gtl::FlatMap<int64_t, ResponseCacheEntry> response_cache_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RESPONSE_CACHE_H_
