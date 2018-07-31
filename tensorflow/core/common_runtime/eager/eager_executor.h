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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// A unit of execution for the EagerExecutor class below. Example subclasses
// encapsulate execution of a TFE_Op, or copying a TFE_TensorHandle from one
// device to another.
class EagerNode {
 public:
  explicit EagerNode(uint64 id);

  virtual ~EagerNode() {}

  // Runs the computation corresponding to this node and blocks till the
  // execution is done.
  virtual Status Run() = 0;

  // An id unique to the TFE_Context under which this node is created. Allocated
  // monotonically.
  const uint64 id;
};

// A class for handling async execution (see TFE_ContextSetAsync).
// Note that this class is thread-safe.
// TODO(agarwal): TFE_OpAddInput may currently block if it tries to access the
// device of the input handle. Fix that.
// TODO(agarwal): On error, mark all affected handles as corrupted.
// TODO(agarwal): Implement support for control dependencies.
// TODO(agarwal): Support out-of-order execution and dispatching multiple
// EagerNode in parallel.
// TODO(agarwal): Implement optimizations over EagerNode traces.
class EagerExecutor {
 public:
  ~EagerExecutor();

  // This is called whenever async mode is enabled. Note that it may be called
  // multiple times as different calling threads may switch async mode on or off
  // independently.
  void EnableAsync();

  // Helper function to create monotonically increasing ids unique to this
  // object.
  uint64 NextId();

  // Schedules `node` for execution.
  // Note that Add must be called in monotonically increasing order of node->id.
  void Add(EagerNode* node);

  // Causes the caller to block till node with id `node_id` has finished
  // execution.
  Status WaitFor(uint64 node_id);

  // Blocks till all currently pending ops are done.
  Status WaitForAllPendingNodes();

  // Clears all currently set errors which re-enables async execution.
  void ClearError();

  // Returns Status based on any errors that occurred during async execution.
  Status status();

 private:
  // Starts execution of pending EagerNodes. This function loops till
  // thread_done_ is set to true. If any errors are encontered, these are set
  // inside `status_`. The loop blocks anytime there are no pending nodes, or if
  // `status_` is not ok.
  void Run();

  Status WaitImpl(bool wait_all, uint64 node_id);

  mutex node_queue_mutex_;

  // Used to signal that some EagerNodes are pending execution.
  condition_variable nodes_pending_ GUARDED_BY(node_queue_mutex_);

  // Queue of pending EagerNodes.
  std::queue<EagerNode*> node_queue_ GUARDED_BY(node_queue_mutex_);

  // `status_` is set based on any errors raised during execution of a
  // EagerNode.  It remains set until ClearError is called.
  Status status_ GUARDED_BY(node_queue_mutex_);

  // Map from id of a EagerNode to condition_variables (not owned by the map).
  // These condition_variables are notified and removed when that EagerNode is
  // done executing, or if an error is found in execution of any EagerNode.
  std::multimap<uint64, condition_variable*> node_done_notifications_
      GUARDED_BY(node_queue_mutex_);

  // Thread object that calls the `Run` method. Currently we use only one thread
  // for executing the EagerNodes one-by-one.
  std::unique_ptr<Thread> thread_ GUARDED_BY(node_queue_mutex_);

  // Indicates that `thread_` should stop as soon as it is done executing the
  // current EagerNode.
  bool thread_done_ GUARDED_BY(node_queue_mutex_) = false;

  mutex next_id_mutex_;
  uint64 next_id_ GUARDED_BY(next_id_mutex_) = 1;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
