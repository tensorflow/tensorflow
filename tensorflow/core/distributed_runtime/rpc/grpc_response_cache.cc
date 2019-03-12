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
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

struct WorkerCacheEntry {
  enum class State {
    PENDING = 0,
    ACTIVE = 1,
    FINISHED = 2,
  };

  State state = State::PENDING;
  int64 expires_seconds;

  ::grpc::ByteBuffer response_buf;
  Status response_status;

  // Additional retries may arrive while a request is still executing.  The
  // callbacks for these calls are queued in `callbacks` and evaluated after
  // the original request is completed.
  std::vector<std::pair<RPCResponse, StatusCallback>> callbacks;
};

void RPCResponse::Encode(::grpc::ByteBuffer* tgt) const {
  if (buf_ != nullptr) {
    *tgt = *buf_;
  } else {
    CHECK(msg_ != nullptr);
    ::grpc::Slice slice(msg_->ByteSizeLong());
    msg_->SerializeWithCachedSizesToArray(
        const_cast<uint8*>(reinterpret_cast<const uint8*>(slice.begin())));
    ::grpc::ByteBuffer tmp(&slice, 1);
    tgt->Swap(&tmp);
  }
}

void RPCResponse::CopyFrom(const ::grpc::ByteBuffer& src) {
  if (buf_ != nullptr) {
    *buf_ = src;
    return;
  }

  CHECK(msg_ != nullptr);
  // We create a single slice when encoding protocol messages.
  std::vector<::grpc::Slice> slices;
  if (src.Dump(&slices).ok()) {
    msg_->ParseFromArray(slices[0].begin(), slices[0].size());
  } else {
    LOG(ERROR) << "Failed to decode cached buffer.";
  }
}

void GrpcResponseCache::LookupOrCompute(const string& key, RPCResponse response,
                                        ComputeFunc compute_func,
                                        StatusCallback done_cb) {
  VLOG(1) << "Lookup " << key;
  std::shared_ptr<WorkerCacheEntry> req;
  MaybeCleanup();
  {
    mutex_lock m(mu_);

    if (requests_.find(key) != requests_.end()) {
      req = requests_[key];
    } else {
      req.reset(new WorkerCacheEntry);
      requests_[key] = req;
    }

    if (req->state == WorkerCacheEntry::State::FINISHED) {
      if (req->expires_seconds > Env::Default()->NowSeconds()) {
        VLOG(1) << "Reuse cached response for " << key;
        response.CopyFrom(req->response_buf);
        done_cb(req->response_status);
        return;
      }
      VLOG(1) << "Found expired cache entry for " << key;
      req->state = WorkerCacheEntry::State::PENDING;
      req->response_buf.Clear();
    }

    req->callbacks.push_back(std::make_pair(response, done_cb));

    if (req->state == WorkerCacheEntry::State::ACTIVE) {
      VLOG(1) << "Found active request for " << key
              << ".  Adding entry to response queue.";
      return;
    }

    VLOG(2) << "No cache entry for " << key << ", running user computation.";
    req->state = WorkerCacheEntry::State::ACTIVE;
    req->expires_seconds = Env::Default()->NowSeconds() + expire_time_seconds_;
  }

  compute_func([this, key, req, response](Status status) {
    mutex_lock m(mu_);
    response.Encode(&req->response_buf);
    current_bytes_ += req->response_buf.Length();

    req->response_status = status;
    req->state = WorkerCacheEntry::State::FINISHED;

    VLOG(1) << "Operation for " << key << " finished. "
            << "Status: " << status << ", " << req->response_buf.Length()
            << " response bytes, " << req->callbacks.size()
            << " pending callbacks.";
    for (auto& cb : req->callbacks) {
      cb.first.CopyFrom(req->response_buf);
      cb.second(req->response_status);
    }
    req->callbacks.clear();
  });
}

// Remove all stale or expired cache entries if the cache is full.
void GrpcResponseCache::MaybeCleanup() {
  mutex_lock m(mu_);
  if (current_bytes_ < max_bytes_) {
    return;
  }

  VLOG(1) << "Cleanup: " << current_bytes_ << " -> " << max_bytes_;
  std::vector<std::pair<string, std::shared_ptr<WorkerCacheEntry>>>
      ordered_entries;
  ordered_entries.reserve(requests_.size());
  for (const auto& p : requests_) {
    ordered_entries.push_back(std::make_pair(p.first, p.second));
  }

  std::sort(ordered_entries.begin(), ordered_entries.end(),
            [](const std::pair<string, std::shared_ptr<WorkerCacheEntry>>& a,
               const std::pair<string, std::shared_ptr<WorkerCacheEntry>>& b) {
              return a.second->expires_seconds > b.second->expires_seconds;
            });

  std::unordered_map<string, std::shared_ptr<WorkerCacheEntry>> kept;
  int64 now = Env::Default()->NowSeconds();
  int64 bytes_used = 0;

  // Always keep active requests.
  for (auto& pair : ordered_entries) {
    if (pair.second->state != WorkerCacheEntry::State::FINISHED) {
      kept.insert(pair);
    }
  }

  // Keep unexpired, finished requests up to half of max_bytes_.  This reduces
  // chances of overfilling the cache when active requests complete and
  // amortizes cache cleanup cost.
  for (auto& pair : ordered_entries) {
    if (pair.second->expires_seconds < now || bytes_used >= max_bytes_ / 2) {
      break;
    }

    if (pair.second->state == WorkerCacheEntry::State::FINISHED) {
      kept.insert(pair);
      bytes_used += pair.second->response_buf.Length();
    }
  }

  VLOG(1) << "Cleaned cache.  Bytes used: " << current_bytes_ << " -> "
          << bytes_used << ". Cache size: " << requests_.size() << " -> "
          << kept.size();
  current_bytes_ = bytes_used;
  std::swap(requests_, kept);
}

}  // namespace tensorflow
