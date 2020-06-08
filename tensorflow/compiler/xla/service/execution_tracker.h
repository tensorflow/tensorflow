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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_TRACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_TRACKER_H_

#include <map>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Represents an asynchronously launched execution. Owns the stream (from the
// passed run_options->stream()) on which the execution is launched and releases
// the stream when destructed.
class AsyncExecution {
 public:
  AsyncExecution(Backend* backend, std::vector<StreamPool::Ptr> streams,
                 const ExecutionProfile& profile, GlobalDataHandle result);

  Status BlockUntilDone() const;

  const GlobalDataHandle& result() const { return result_; }

  const ExecutionProfile& profile() const { return profile_; }

 private:
  // Backend to execute the computation on.
  Backend* backend_;

  // Stream on which the execution is launched.
  std::vector<StreamPool::Ptr> streams_;

  // Profile object of the execution to be returned to the user.
  ExecutionProfile profile_;

  // Data handle to the result of the execution. Data represented by this handle
  // is valid only after BlockUntilDone() is called.
  GlobalDataHandle result_;
};

// Tracks asynchronously launched executions for the XLA service.
class ExecutionTracker {
 public:
  ExecutionTracker();

  // Registers an execution with its backend, streams, and data handle to the
  // execution result. Returns a handle for the registered execution.
  ExecutionHandle Register(Backend* backend,
                           std::vector<StreamPool::Ptr> stream,
                           const ExecutionProfile& profile,
                           GlobalDataHandle data);

  // Unregisters the execution for the given handle.
  Status Unregister(const ExecutionHandle& handle);

  // Resolves the given ExecutionHandle to an AsyncExecution. Returns an
  // error status if the given handle is not found, which means that the
  // execution is not yet registered or already unregistered.
  StatusOr<const AsyncExecution*> Resolve(const ExecutionHandle& handle);

 private:
  // The next handle to assign to an execution.
  int64 next_handle_ TF_GUARDED_BY(execution_mutex_);

  // Mapping from ExecutionHandle handle to the corresponding registered
  // AsyncExecution object.
  std::map<int64, std::unique_ptr<AsyncExecution>> handle_to_execution_
      TF_GUARDED_BY(execution_mutex_);

  tensorflow::mutex execution_mutex_;  // Guards the execution mapping.

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutionTracker);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_TRACKER_H_
