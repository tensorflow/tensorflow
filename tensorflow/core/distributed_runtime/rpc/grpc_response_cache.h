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

#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

// gRPC response caching.  Most WorkerService methods cannot be retried directly
// as they will fail or deadlock.  To enable retrying, we can instead cache
// responses for a short period of time and reply to duplicate requests from the
// cache.
namespace tensorflow {

// Union type to aid caching of either raw buffers (for RecvTensor RPCs) and
// protocol buffer messages (for all other RPCs).
class RPCResponse {
 public:
  explicit RPCResponse() : buf_(nullptr), msg_(nullptr) {}
  explicit RPCResponse(::grpc::ByteBuffer* b) : buf_(b), msg_(nullptr) {}
  explicit RPCResponse(protobuf::Message* m) : buf_(nullptr), msg_(m) {}

  // Encode this response into the target buffer.
  void Encode(::grpc::ByteBuffer* tgt) const;

  // Copy from `src`: if this is a buffer, make a shallow copy.
  // For protocol messages, parse the response from `src`.
  void CopyFrom(const ::grpc::ByteBuffer& src);

 private:
  ::grpc::ByteBuffer* buf_;
  protobuf::Message* msg_;
};

typedef std::function<void(StatusCallback)> ComputeFunc;
struct WorkerCacheEntry;

// Track and cache the state of worker service RPCs.  An RPC can be in 3 states:
//
// * PENDING: this is the first call of the RPC, and it will transition to
// * ACTIVE: another thread is active processing this RPC
// * FINISHED: the worker has finished processing the method
//
// The response from completed RPCs are LRU cached until either `max_bytes`
// bytes are in use by the cache or they expire (according to `expire_time`).
class GrpcResponseCache {
 public:
  GrpcResponseCache(int64 max_bytes, absl::Duration expire_duration)
      : max_bytes_(max_bytes), expire_duration_(expire_duration) {}

  // Lookup the result for key.
  // If it is finished, invoke `done_cb` immediately after filling `response`.
  // If active, done_db will be invoked when the current call completes.
  // Otherwise, invoke `compute_func` to fill the cache and invoke done_cb.
  void LookupOrCompute(const string& key, RPCResponse response,
                       ComputeFunc compute_func, StatusCallback done_cb);

  // Remove all stale or expired cache entries if the cache is full.
  void MaybeCleanup();

 private:
  int64 current_bytes_ GUARDED_BY(mu_) = 0;
  const int64 max_bytes_;
  const absl::Duration expire_duration_;

  std::unordered_map<string, std::shared_ptr<WorkerCacheEntry>> requests_
      GUARDED_BY(mu_);
  mutex mu_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RESPONSE_CACHE_H_
