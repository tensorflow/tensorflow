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

#include "tensorflow/core/distributed_runtime/rpc/grpc_response_cache.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

bool GrpcResponseCache::QueueRequest(int64_t request_id, int64_t step_id,
                                     const FinishResponseCB& cb) {
  VLOG(1) << "GrpcResponseCache Lookup " << request_id;

  mu_.lock();

  ResponseCacheEntry& entry = response_cache_[request_id];

  if (entry.state == ResponseCacheEntry::State::FINISHED) {
    VLOG(1) << "Reuse cached response for " << request_id;
    // Make a copy of the ResponseCacheEntry so that we can run FinishResponse
    // outside the critical section. FinishResponse can be potentially
    // expensive.
    auto entry_copy = entry;

    mu_.unlock();
    entry_copy.FinishResponse(cb);
    return true;
  }

  entry.callbacks.emplace_back(cb);

  if (entry.state == ResponseCacheEntry::State::ACTIVE) {
    VLOG(1) << "Found active request for " << request_id
            << ".  Adding entry to response queue.";
    mu_.unlock();
    return true;
  } else {
    VLOG(2) << "No cache entry for " << request_id
            << ", running user computation.";
    entry.step_id = step_id;
    entry.state = ResponseCacheEntry::State::ACTIVE;
    mu_.unlock();
    return false;
  }
}

void GrpcResponseCache::OnRequestFinished(int64_t request_id,
                                          const Tensor& tensor, bool is_dead,
                                          const Status& status) {
  absl::optional<ResponseCacheEntry> entry_copy;

  {
    mutex_lock m(mu_);

    auto it = response_cache_.find(request_id);
    if (it == response_cache_.end()) {
      LOG(ERROR) << "Unexpected missing response cache entry for request "
                 << request_id;
      return;
    }
    ResponseCacheEntry& entry = it->second;

    VLOG(1) << "Operation for " << request_id << " finished. "
            << "Status: " << status << ", tensor size " << tensor.TotalBytes()
            << " bytes, " << entry.callbacks.size() << " pending callbacks.";

    entry.tensor = tensor;
    entry.is_dead = is_dead;
    entry.response_status = status;
    entry.state = ResponseCacheEntry::State::FINISHED;

    // We copy the extra work out of the critical section in order to avoid
    // serializing the work for sending response.
    entry_copy = entry;

    entry.callbacks.clear();
  }

  for (auto& cb : entry_copy->callbacks) {
    entry_copy->FinishResponse(cb);
  }
}

void GrpcResponseCache::EraseRequestId(int64_t request_id) {
  mutex_lock m(mu_);
  response_cache_.erase(request_id);
}

void GrpcResponseCache::CleanEntriesForStep(int64_t step_id) {
  mutex_lock m(mu_);
  // Remove all cache entries whose step id is the given step_id
  for (auto it = response_cache_.begin(), last = response_cache_.end();
       it != last;) {
    if (it->second.step_id == step_id) {
      VLOG(1) << "Erase stale GrpcResponseCache entry " << it->first;
      it = response_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace tensorflow
