/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
#define TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class RunHandler;

// RunHandlerPool is a fixed size pool of pre-allocated RunHandlers
// that can be used for tracking inter-op work for a given Session::Run().
// RunHandler(s) in the pool are initially 'inactive'. A RunHandler becomes
// 'active' when its unique_ptr is returned by Get() and is being used by a
// client. It becomes 'inactive' once more when its unique_ptr gets destroyed.
//
// Expected usage:
//
// * Create a single RunHandlerPool (say run_handler_pool_).
//
// * When a Session::Run() is invoked, obtain a handler by:
// auto handler = run_handler_pool_->Get();
//
// * Use handler for scheduling all inter-op work by:
// handler->ScheduleInterOpClosure(closure);
//
// This class is thread safe.
class RunHandlerPool {
 public:
  explicit RunHandlerPool(int num_inter_op_threads);
  ~RunHandlerPool();

  // Returns an inactive RunHandler from the pool.
  //
  // RunHandlers in RunHandlerPool are initially 'inactive'.
  // A RunHandler becomes 'active' when its unique_ptr its returned by Get()
  // and is being used by a client.  It becomes 'inactive' once more when the
  // unique_ptr is destroyed.
  //
  // Will block unless there is an inactive handler.
  std::unique_ptr<RunHandler> Get();

 private:
  class Impl;
  friend class RunHandler;

  std::unique_ptr<Impl> impl_;
};

// RunHandler can be used to schedule inter-op closures to run on a global pool
// shared across all Session::Run(s).
//
// It can only be created via RunHandlerPool::Get().
//
// This class can be used instead of directly scheduling closures on a global
// pool since it maintains a global view across all sessions and optimizes pool
// scheduling to improve (median and tail) latency.
//
// This class is thread safe.
class RunHandler {
 public:
  void ScheduleInterOpClosure(std::function<void()> fn);

  ~RunHandler();

 private:
  class Impl;
  friend class RunHandlerPool::Impl;

  explicit RunHandler(Impl* impl);

  Impl* impl_;  // NOT OWNED.
};

}  // end namespace tensorflow.

#endif  // TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
