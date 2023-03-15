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

#include "tensorflow/core/common_runtime/device/device_event_mgr.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {

namespace {

// The EventMgr has two threads to execute event callback functions. Issues for
// reconsideration:
//  - Is this the right number of threads?
//  - Should EventMgrs be shared between devices on a machine with multiple
//  devices of the same type?
static const int kNumThreads = 2;
}  // namespace

namespace device_event_mgr {
class ThreadLabel {
 public:
  static const char* GetValue() { return value_; }

  // v must be a static const because value_ will capture and use its value
  // until reset or thread terminates.
  static void SetValue(const char* v) { value_ = v; }

 private:
  static thread_local const char* value_;
};
thread_local const char* ThreadLabel::value_ = "";

void WarnIfInCallback(std::function<void()> f) {
  const char* label = ThreadLabel::GetValue();
  if (label && !strcmp(label, "device_event_mgr")) {
    if (f) {
      f();
    } else {
      LOG(WARNING) << "Executing inside EventMgr callback thread: "
                   << CurrentStackTrace();
    }
  }
}

void InitThreadpoolLabels(thread::ThreadPool* threadpool) {
  static const char* label = "device_event_mgr";
  mutex mu;
  int init_count = 0;
  condition_variable all_initialized;
  int exit_count = 0;
  condition_variable ready_to_exit;
  const int num_threads = threadpool->NumThreads();
  for (int i = 0; i < num_threads; ++i) {
    threadpool->Schedule([num_threads, &mu, &init_count, &all_initialized,
                          &exit_count, &ready_to_exit]() {
      device_event_mgr::ThreadLabel::SetValue(label);
      mutex_lock l(mu);
      ++init_count;
      if (init_count == num_threads) {
        all_initialized.notify_all();
      }
      while (init_count < num_threads) {
        all_initialized.wait(l);
      }
      if (++exit_count == num_threads) {
        ready_to_exit.notify_all();
      }
    });
  }
  {
    mutex_lock l(mu);
    while (exit_count < num_threads) {
      ready_to_exit.wait(l);
    }
  }
}
}  // namespace device_event_mgr

EventMgr::EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      threadpool_(Env::Default(), "Device_Event_Manager", kNumThreads) {
  device_event_mgr::InitThreadpoolLabels(&threadpool_);
}

EventMgr::~EventMgr() {
  // Wait for all streams to complete.  All of the streams have completed when
  // `callback_streams_` is empty, or when all of the remaining streams are in
  // an error state.
  mutex_lock lock(mu_);
  mu_.Await(tsl::Condition(
      +[](decltype(callback_streams_)* callback_streams) {
        // std::all_of returns true if the container is empty.
        return absl::c_all_of(*callback_streams,
                              [](auto& kv) { return !kv.first->ok(); });
      },
      &callback_streams_));

  // The threadpool's destructor will block waiting for all outstanding
  // callbacks to complete.
}

void EventMgr::ThenExecute(se::Stream* stream, std::function<void()> func) {
  // Ensure the correct GPU is active before making any CUDA calls.
  //
  // This shouldn't be necessary!  StreamExecutor uses the CUDA driver API, and
  // all calls there take a GPU context as a parameter; the "current GPU" from
  // the perspective of the runtime API shouldn't matter.  But we can verify
  // that it in fact *does* matter, perhaps specifically because of the
  // ThenHostCallback calls, though it's hard to tell.
  //
  // This library is only available when GOOGLE_CUDA is defined.
#if GOOGLE_CUDA
  stream_executor::gpu::ScopedActivateExecutorContext scoped_activation{exec_};
#endif

  // tl;dr: Don't make CUDA calls while holding the lock on mu_.
  //
  // There are three mutexes at play here.  We need to be careful to avoid a
  // cycle in the order in which we acquire them.  The mutexes are:
  //
  //  1. Any mutexes inside the CUDA runtime or driver.  We can consider these
  //     to be a single mutex that wraps all CUDA calls and is also held while
  //     running the host callback, because that's the worst case for our
  //     analysis.
  //  2. Any mutexes inside threadpool_, which again we can consier to be a
  //     single mutex wrapping all calls to the object.
  //  3. EventMgr::mu_.
  //
  // The CUDA host callback needs to schedule func on threadpool_ (that's the
  // whole point of all this).  It also needs to modify internal state of the
  // EventMgr, e.g. to push an elements back onto free_events_ and
  // free_streams_.  Thus it's unavoidable that we acquire mutexes (2) and (3)
  // while holding (1).  This means that to avoid a deadlock, we must drop the
  // lock on the EventMgr (3) before making any CUDA API calls (1)!

  // Get an event and stream off the free list, lazily creating them if
  // necessary.  There's currently no limit on the number of allocated events
  // and streams.
  //
  // If we have to create a new stream/event, don't call Init() while holding
  // mu_, because that's what touches the CUDA API and can cause deadlocks.
  std::unique_ptr<se::Event> event;
  bool is_new_event = false;
  se::Stream* callback_stream;
  bool is_new_stream = false;
  {
    mutex_lock lock(mu_);

    // Get an event off the free list.
    if (free_events_.empty()) {
      free_events_.push_back(std::make_unique<se::Event>(exec_));
      is_new_event = true;
    }
    event = std::move(free_events_.back());
    free_events_.pop_back();

    // Get the internal stream associated with `stream`, or grab one off the
    // free list.
    //
    // Disable thread-safety analysis on this lambda because tsl::Mutex
    // currently lacks an AssertHeld function.  :(
    auto it = callback_streams_.lazy_emplace(
        stream, [&](const auto& ctor) ABSL_NO_THREAD_SAFETY_ANALYSIS {
          if (free_streams_.empty()) {
            free_streams_.push_back(std::make_unique<se::Stream>(exec_));
            is_new_stream = true;
          }
          ctor(stream, std::make_pair(std::move(free_streams_.back()),
                                      /*num_pending_events=*/0));
          free_streams_.pop_back();
        });
    callback_stream = it->second.first.get();
    it->second.second++;  // increment num_pending_events
  }
  if (is_new_event) {
    event->Init();
  }
  if (is_new_stream) {
    callback_stream->Init();
  }

  // Set callback_stream to run `func` when `stream` finishes the work that's
  // currently pending.
  stream->ThenRecordEvent(event.get());
  callback_stream->ThenWaitFor(event.get());

  // `mutable` is needed on the lambda so we can move `event` and `func`.
  // Without `mutable`, these variables are const and can't be moved.
  callback_stream->ThenDoHostCallbackWithStatus(
      [this, stream, event = std::move(event),
       func = std::move(func)]() mutable {
        threadpool_.Schedule(std::move(func));

        mutex_lock lock(mu_);
        free_events_.push_back(std::move(event));

        // Update the number of pending events on `stream` and erase it from
        // callback_streams_ if no events are pending any longer.
        auto callback_stream_it = callback_streams_.find(stream);
        if (callback_stream_it == callback_streams_.end()) {
          return tsl::errors::Internal(
              "Invariant violation in EventMgr: callback_streams_ does not "
              "contain stream ",
              stream);
        }
        auto& [callback_stream, num_pending_events] =
            callback_stream_it->second;
        if (num_pending_events <= 0) {
          return tsl::errors::Internal(
              "Invariant violation in EventMgr: refcount for stream ", stream,
              "should be >= 1, but was ", num_pending_events);
        }
        num_pending_events--;

        if (num_pending_events == 0) {
          free_streams_.push_back(std::move(callback_stream));
          callback_streams_.erase(callback_stream_it);
        }
        return tsl::OkStatus();
      });
}

EventMgrFactory* EventMgrFactory::Singleton() {
  static EventMgrFactory* instance = new EventMgrFactory;
  return instance;
}

EventMgr* EventMgrFactory::GetEventMgr(se::StreamExecutor* se,
                                       const GPUOptions& gpu_options) {
  mutex_lock l(mu_);
  // TODO(laigd): consider making gpu_options part of the key. It's not
  // currently since EventMgr depends only rely on field deferred_deletion_bytes
  // and polling_active_delay_usecs from gpu_options which are not used or
  // rarely used.
  auto itr = event_mgr_map_.find(se);
  if (itr == event_mgr_map_.end()) {
    auto event_mgr = new EventMgr(se, gpu_options);
    event_mgr_map_[se] = event_mgr;
    return event_mgr;
  } else {
    return itr->second;
  }
}

}  // namespace tensorflow
