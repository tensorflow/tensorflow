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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_

#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TODO(annarev): Check if we can use a more general option representation here
// that could work for other device types as well.
class GPUOptions;

// The callback provided to EventMgr::ThenExecute must not block or take a long
// time.  If it does, performance may be impacted and device memory may be
// exhausted.  This macro is for checking that an EventMgr thread is not
// accidentally entering blocking parts of the code, e.g. the RPC subsystem.
//
// Intended use is something like
//
//   void RespondToAnRPC(Params* params) {
//      WARN_IF_IN_EVENT_MGR_THREAD;
//      if (params->status.ok()) { ...
//
namespace device_event_mgr {
// Logs a stack trace if current execution thread belongs to this EventMgr
// object.  If f is not nullptr, executes instead of  logging the stack trace.
// trace.
void WarnIfInCallback(std::function<void()> f);
}  // namespace device_event_mgr
#define WARN_IF_IN_EVENT_MGR_THREAD \
  ::tensorflow::device_event_mgr::WarnIfInCallback(nullptr)

// EventMgr lets you register a callback to be executed when a given
// StreamExecutor stream completes all the work that's thus-far been enqueued on
// the stream.
class EventMgr {
 public:
  virtual ~EventMgr();

  // Execute `func` when all pending stream actions have completed.  func must
  // be brief and non-blocking since it executes in the one thread used for all
  // such callbacks and also buffer deletions.
  void ThenExecute(se::Stream* stream, std::function<void()> func) {
    mutex_lock l(mu_);
    EnqueueCallback(stream, std::move(func));
    PollEvents(stream);
  }

 private:
  friend class TEST_EventMgr;
  friend class TEST_EventMgrHelper;
  friend class EventMgrFactory;

  se::StreamExecutor* const exec_;
  const int32 polling_active_delay_usecs_;
  mutex mu_;
  condition_variable events_pending_ TF_GUARDED_BY(mu_);

  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

  // Set up `func` to be called once `stream` completes all its outstanding
  // work.
  void EnqueueCallback(se::Stream* stream, std::function<void()> func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // This function should be called at roughly the same tempo as QueueTensors()
  // to check whether pending events have recorded, and then retire them.
  //
  // If `stream` is not null, we only poll events for that stream.  Otherwise we
  // poll events for all streams.
  void PollEvents(se::Stream* stream = nullptr)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An internal polling loop that runs at a low frequency to clear straggler
  // Events.
  void PollLoop();

  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<std::unique_ptr<se::Event>> free_events_ TF_GUARDED_BY(mu_);

  // Callbacks waiting on their events to complete.
  absl::flat_hash_map<
      se::Stream*,
      std::deque<std::pair<std::unique_ptr<se::Event>, std::function<void()>>>>
      callbacks_ TF_GUARDED_BY(mu_);

  bool stop_polling_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

// Manages all the EventMgr instances.
class EventMgrFactory {
 public:
  static EventMgrFactory* Singleton();

  EventMgr* GetEventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

 private:
  mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  absl::flat_hash_map<se::StreamExecutor*, EventMgr*> event_mgr_map_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_
