/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_WORKER_THREAD_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_WORKER_THREAD_H_

#include <deque>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"

namespace xla::gpu {

// WorkerThread is a single, long-lived thread on which you can synchronously
// execute functions. For example:
//
//     WorkerThread t;
//     TF_RETURN_IF_ERROR(t.Run([]() -> absl::Status {
//         LOG(INFO) << "This runs on the worker thread.";
//         return absl::OkStatus();
//     }));
//     TF_RETURN_IF_ERROR(t.Run([]() -> absl::Status {
//         LOG(INFO) << "This also runs on the worker thread.";
//         return absl::OkStatus();
//     }));
//
// WorkerThread is thread-safe. Functions passed to Run are executed serially.
class WorkerThread {
 public:
  // Constructs a WorkerThread using the provided environment (see
  // Env::StartThread) and name (for debugging).
  explicit WorkerThread(tsl::Env& env = *tsl::Env::Default(),
                        absl::string_view thread_name = "WorkerThread");

  // Destroys the WorkerThread. A WorkerThread should not be destroyed while
  // there are still pending invocations of the Run method.
  ~WorkerThread();

  // Synchronously executes the provided function on the worker thread and
  // returns the resulting status. Functions passed to Run are executed serially
  // on the worker thread.
  absl::Status Run(absl::AnyInvocable<absl::Status() &&> f);

  // WorkerThread is not copyable or movable.
  WorkerThread(const WorkerThread&) = delete;
  WorkerThread(WorkerThread&&) = delete;
  WorkerThread& operator=(const WorkerThread&) = delete;
  WorkerThread& operator=(WorkerThread&&) = delete;

 private:
  struct WorkItem {
    absl::AnyInvocable<absl::Status() &&> f;  // function to execute
    absl::Status status;                      // the result of executing f
    absl::Notification done;                  // notified after f is executed
  };

  // Underlying thread that executes functions.
  std::unique_ptr<tsl::Thread> thread_;

  // Guards the following fields.
  absl::Mutex mu_;

  // A work queue of pending functions to execute.
  std::deque<WorkItem*> work_ ABSL_GUARDED_BY(mu_);

  // Set to true when the WorkerThread is being destroyed.
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_WORKER_THREAD_H_
