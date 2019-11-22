/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

// Activity IDs: To avoid contention over a counter, the top 32 bits identify
// the originating thread, the bottom 32 bits name the event within a thread.
// IDs may be reused after 4 billion events on one thread, or 4 billion threads.
static std::atomic<uint32> thread_counter(1);  // avoid kUntracedActivity
uint64 NewActivityId() {
  const thread_local static uint32 thread_id = thread_counter.fetch_add(1);
  thread_local static uint32 per_thread_activity_id = 0;
  return static_cast<uint64>(thread_id) << 32 | per_thread_activity_id++;
}

/* static */ uint64 TraceMe::ActivityStartImpl(
    absl::string_view activity_name) {
  uint64 activity_id = NewActivityId();
  TraceMeRecorder::Record({activity_id, string(activity_name),
                           /*start_time=*/EnvTime::NowNanos(),
                           /*end_time=*/0});
  return activity_id;
}

/* static */ void TraceMe::ActivityEndImpl(uint64 activity_id) {
  TraceMeRecorder::Record({activity_id, /*name=*/"", /*start_time=*/0,
                           /*end_time=*/EnvTime::NowNanos()});
}

}  // namespace profiler
}  // namespace tensorflow
