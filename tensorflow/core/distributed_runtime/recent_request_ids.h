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

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/mutex.h"
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
  RecentRequestIds(int num_tracked_request_ids);

  // Returns OK iff request_id has not been seen in the last
  // num_tracked_request_ids insertions. For backwards compatibility, this
  // always returns OK for request_id 0. The method_name and the request's
  // ShortDebugString are added to returned errors.
  Status TrackUnique(int64 request_id, const string& method_name,
                     const protobuf::Message& request);

 private:
  mutex mu_;
  // next_index_ indexes into circular_buffer_, and points to the next storage
  // space to use. When the buffer is full, next_index_ points at the oldest
  // request_id.
  int next_index_ GUARDED_BY(mu_) = 0;
  std::vector<int64> circular_buffer_ GUARDED_BY(mu_);
  gtl::FlatSet<int64> set_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
