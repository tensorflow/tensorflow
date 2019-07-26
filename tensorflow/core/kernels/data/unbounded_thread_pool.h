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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_UNBOUNDED_THREAD_POOL_H_
#define TENSORFLOW_CORE_KERNELS_DATA_UNBOUNDED_THREAD_POOL_H_

#include <deque>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/thread_factory.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
namespace data {

// An `UnboundedThreadPool` provides a mechanism for temporally multiplexing a
// potentially large number of "logical" threads onto a smaller number of
// "physical" threads. The multiplexing is achieved by using an
// `UnboundedWorkQueue`.
class UnboundedThreadPool {
 public:
  UnboundedThreadPool(Env* env, const string& thread_name)
      : unbounded_work_queue_(env, thread_name) {}
  ~UnboundedThreadPool() = default;

  // Returns an implementation of `ThreadFactory` that can be used to create
  // logical threads in this pool.
  std::shared_ptr<ThreadFactory> get_thread_factory();

 private:
  class LogicalThreadFactory;
  class LogicalThreadWrapper;

  std::unique_ptr<Thread> ScheduleOnWorkQueue(std::function<void()> fn);

  UnboundedWorkQueue unbounded_work_queue_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_UNBOUNDED_THREAD_POOL_H_
