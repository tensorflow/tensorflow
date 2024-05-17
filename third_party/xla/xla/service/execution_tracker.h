/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_EXECUTION_TRACKER_H_
#define XLA_SERVICE_EXECUTION_TRACKER_H_

#include <map>
#include <memory>
#include <utility>

#include "xla/executable_run_options.h"
#include "xla/service/backend.h"
#include "xla/service/stream_pool.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

// Represents an asynchronously launched execution. Owns the stream (from the
// passed run_options->stream()) on which the execution is launched and releases
// the stream when destructed.
class AsyncExecution {
 public:
  AsyncExecution(Backend* backend, std::vector<StreamPool::Ptr> streams,
                 const ExecutionProfile& profile, GlobalDataHandle result);

  absl::Status BlockUntilDone() const;

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
  absl::Status Unregister(const ExecutionHandle& handle);

  // Resolves the given ExecutionHandle to an AsyncExecution. Returns an
  // error status if the given handle is not found, which means that the
  // execution is not yet registered or already unregistered.
  absl::StatusOr<const AsyncExecution*> Resolve(const ExecutionHandle& handle);

 private:
  // The next handle to assign to an execution.
  int64_t next_handle_ ABSL_GUARDED_BY(execution_mutex_);

  // Mapping from ExecutionHandle handle to the corresponding registered
  // AsyncExecution object.
  std::map<int64_t, std::unique_ptr<AsyncExecution>> handle_to_execution_
      ABSL_GUARDED_BY(execution_mutex_);

  absl::Mutex execution_mutex_;  // Guards the execution mapping.

  ExecutionTracker(const ExecutionTracker&) = delete;
  ExecutionTracker& operator=(const ExecutionTracker&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_EXECUTION_TRACKER_H_
