/* Copyright 2017 Graphcore Ltd
 */

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

#ifndef TENSORFLOW_COMPILER_POPLAR_STREAM_EXECUTOR_POPLAR_STREAM_H_
#define TENSORFLOW_COMPILER_POPLAR_STREAM_EXECUTOR_POPLAR_STREAM_H_

#include <functional>
#include <memory>

#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace poplar {

class PoplarStream : public internal::StreamInterface {
 public:
  PoplarStream();
  ~PoplarStream() override;

  bool EnqueueTask(std::function<void()> task);

  void BlockUntilDone();

 private:
  // Use only one thread and own task queue to preserve FIFO ordering
  // for the operations enqueued by any given stream.
  static const int kExecutorThreads = 1;
  std::unique_ptr<port::ThreadPool> host_executor_;

  mutex mu_;
  int pending_tasks_ GUARDED_BY(mu_) = 0;
  condition_variable completion_condition_;
};

}  // namespace poplar
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_COMPILER_POPLAR_STREAM_EXECUTOR_POPLAR_STREAM_H_
