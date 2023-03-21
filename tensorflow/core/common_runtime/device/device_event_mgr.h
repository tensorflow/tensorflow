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
#include "tensorflow/tsl/platform/thread_annotations.h"

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

  // Executes `func` when all pending stream actions have completed.  func must
  // be brief and non-blocking since it executes in a shared threadpool used for
  // all such callbacks.
  void ThenExecute(se::Stream* stream, std::function<void()> func);

 private:
  friend class TEST_EventMgr;
  friend class TEST_EventMgrHelper;
  friend class EventMgrFactory;

  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

  mutex mu_;
  se::StreamExecutor* const exec_;
  thread::ThreadPool threadpool_;

  // Stacks of currently-unused events and streams.
  std::vector<std::unique_ptr<se::Event>> free_events_ TF_GUARDED_BY(mu_);
  std::vector<std::unique_ptr<se::Stream>> free_streams_ TF_GUARDED_BY(mu_);

  // Logically, we want a call to ThenExecute(stream, func) to enqueue a CUDA
  // host callback (StreamExecutor ThenDoHostCallback) that runs `func` onto
  // `stream`.
  //
  // But CUDA host callbacks execute synchronously with respect to the GPU work
  // on their stream -- in other words, they block the stream.  Blocking a
  // stream is potentially very expensive to the GPU work there; it essentially
  // flushes the work queue and can result in gaps in GPU execution.
  //
  // We therefore build an "asynchronous CUDA callback" as follows:
  //  - A user calls ThenExecute(stream, func).
  //  - We enqueue a CUDA event onto `stream`; this lets us find out when
  //    `stream` completes all the work that's currently pending.
  //  - A *separate* stream, managed by EventMgr, waits on the event and then
  //    enqueues a synchronous CUDA callback that runs `func`.
  //
  // This way we're never blocking the stream that actually executes GPU work.
  //
  // callback_streams_ maps the first stream (that was passed to ThenExecute) to
  // the second one (owned by EventMgr).
  absl::flat_hash_map<se::Stream* /*user stream*/,
                      std::pair<std::unique_ptr<se::Stream> /*callback stream*/,
                                int64_t /*num_pending_events*/>>
      callback_streams_ TF_GUARDED_BY(mu_);
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
