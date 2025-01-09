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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// RecentRequestIds tracks recent 64-bit request_ids. When maximum capacity is
// reached, the oldest request_id is evicted. Thread safe.
//
// Some RPCs like RecvTensor are unsafe to retry. For example, RecvTensor pairs
// one sender and one receiver, and the receiver waits for the sender's tensor.
// Retried RecvTensor requests are problematic, because the original RecvTensor
// request may have consumed the sender's tensor, so a retried request might
// block forever. RecentRequestIds identifies retried requests, so we can fail
// them instead of blocking forever.
//
// Internally, recent request_ids are stored in two data structures: a set and a
// circular buffer. The set is used for efficient lookups, and the circular
// buffer tracks the oldest request_id. When the buffer is full, the new
// request_id replaces the oldest request_id in the circular buffer, and the
// oldest request_id is removed from the set.
class RecentRequestIds {
 public:
  // num_tracked_request_ids should be much larger than the number of RPCs that
  // can be received in a small time window. For example, we observed a peak RPC
  // rate of ~700 RecvTensor RPC/s when training inception v3 on TPUs, so we
  // currently set num_tracked_request_ids to 100,000 for RecvTensor.
  // Having a large `num_shars` can prevent run into lock contention in this
  // class.
  explicit RecentRequestIds(int num_tracked_request_ids, int num_shards = 1);

  // Returns OK iff request_id has not been seen in the last
  // num_tracked_request_ids insertions. For backwards compatibility, this
  // always returns OK for request_id 0. The method_name and the request's
  // ShortDebugString are added to returned errors.
  absl::Status TrackUnique(int64_t request_id, const string& method_name,
                           const protobuf::Message& request);
  // Overloaded version of the above function for wrapped protos.
  template <typename RequestWrapper>
  absl::Status TrackUnique(int64_t request_id, const string& method_name,
                           const RequestWrapper* wrapper);

 private:
  bool Insert(int64_t request_id);

  struct IndexBucket {
    mutex mu;
    // next_index indexes into circular_buffer_, and points to the next storage
    // space to use. When the buffer is full, next_index_ points at the oldest
    // request_id.
    int next_index TF_GUARDED_BY(mu) = 0;
    std::vector<int64_t> circular_buffer TF_GUARDED_BY(mu);
    absl::flat_hash_set<int64_t> set TF_GUARDED_BY(mu);
  };

  // This vector is immutable so we don't need to use a mutex to protect it.
  std::vector<IndexBucket> index_buckets_;
};

// Implementation details

template <typename RequestWrapper>
absl::Status RecentRequestIds::TrackUnique(int64_t request_id,
                                           const string& method_name,
                                           const RequestWrapper* wrapper) {
  if (Insert(request_id)) {
    return absl::OkStatus();
  } else {
    return errors::Aborted("The same ", method_name,
                           " request was received twice. ",
                           wrapper->ToProto().ShortDebugString());
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
