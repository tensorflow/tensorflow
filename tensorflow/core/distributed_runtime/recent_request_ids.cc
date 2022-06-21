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

#include "tensorflow/core/distributed_runtime/recent_request_ids.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

RecentRequestIds::RecentRequestIds(int num_tracked_request_ids)
    : circular_buffer_(num_tracked_request_ids) {
  set_.reserve(num_tracked_request_ids);
}

bool RecentRequestIds::Insert(int64_t request_id) {
  if (request_id == 0) {
    // For backwards compatibility, allow all requests with request_id 0.
    return true;
  }

  mutex_lock l(mu_);
  const bool inserted = set_.insert(request_id).second;
  if (!inserted) {
    // Note: RecentRequestIds is not strict LRU because we don't update
    // request_id's age in the circular_buffer_ if it's tracked again. Strict
    // LRU is not useful here because returning this error will close the
    // current Session.
    return false;
  }

  // Remove the oldest request_id from the set_. circular_buffer_ is
  // zero-initialized, and zero is never tracked, so it's safe to do this even
  // when the buffer is not yet full.
  set_.erase(circular_buffer_[next_index_]);
  circular_buffer_[next_index_] = request_id;
  next_index_ = (next_index_ + 1) % circular_buffer_.size();
  return true;
}

Status RecentRequestIds::TrackUnique(int64_t request_id,
                                     const string& method_name,
                                     const protobuf::Message& request) {
  if (Insert(request_id)) {
    return OkStatus();
  } else {
    return errors::Aborted("The same ", method_name,
                           " request was received twice. ",
                           request.ShortDebugString());
  }
}

}  // namespace tensorflow
