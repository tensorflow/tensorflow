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
#include <utility>

#include "absl/synchronization/notification.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {

namespace {
// The EventMgr has 1 thread for the polling loop and one to execute
// event callback functions. Issues for reconsideration:
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
      polling_active_delay_usecs_(gpu_options.polling_active_delay_usecs()
                                      ? gpu_options.polling_active_delay_usecs()
                                      : 10),
      threadpool_(Env::Default(), "Device_Event_Manager", kNumThreads) {
  device_event_mgr::InitThreadpoolLabels(&threadpool_);
  StartPollingLoop();
}

EventMgr::~EventMgr() {
  StopPollingLoop();

  for (auto& [stream, stream_callbacks] : callbacks_) {
    for (auto& [event, callback] : stream_callbacks) {
      threadpool_.Schedule(std::move(callback));
    }
  }
  // The threadpool's destructor will block waiting for all outstanding
  // callbacks to complete.
}

void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_ = std::make_unique<absl::Notification>();
  threadpool_.Schedule([this]() { PollLoop(); });
}

void EventMgr::StopPollingLoop() {
  if (polling_stopped_) {
    {
      mutex_lock l(mu_);
      stop_polling_ = true;
      events_pending_.notify_all();
    }
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
}

// A polling loop to detect completion of device events.
//
// While one or more events is outstanding, poll for completed events.  When no
// events are outstanding, we sleep until one is enqueued.
void EventMgr::PollLoop() {
  ToFreeVector to_free;
  while (true) {
    bool events_still_pending;
    {
      mutex_lock l(mu_);
      if (stop_polling_) {
        break;
      }
      if (callbacks_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(nullptr, &to_free);  // poll all streams
      events_still_pending = !callbacks_.empty();
    }
    FreeMemory(to_free);
    to_free.clear();

    if (events_still_pending) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}

void EventMgr::EnqueueCallback(se::Stream* stream, std::function<void()> func) {
  VLOG(2) << "EnqueueCallback with one or more callbacks pending on "
          << callbacks_.size() << " streams and " << free_events_.size()
          << " unused event objects.";
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
    free_events_.emplace_back(exec_->CreateEvent().value());
  }

  std::unique_ptr<se::Event> e = std::move(free_events_.back());
  free_events_.pop_back();
  stream->RecordEvent(e.get()).IgnoreError();

  bool was_empty = callbacks_.empty();
  callbacks_[stream].push_back({std::move(e), std::move(func)});

  // Wake up the polling thread if it was sleeping.
  if (was_empty) {
    events_pending_.notify_all();
  }
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
// spikes of up to several hundred outstanding.  (If GPUKernelTracker
// is used to cap pending kernels there should never be more than
// that many.)
void EventMgr::PollEvents(se::Stream* stream,
                          absl::InlinedVector<InUse, 4UL>* to_free) {
  VLOG(2) << "PollEvents with one or more callbacks pending on "
          << callbacks_.size() << " streams and " << free_events_.size()
          << " unused event objects.";

  // Polls the events for one stream.
  //
  // `stream_it` should be an iterator into callbacks_.  Modifies stream_it so
  // it points to the next element of callbacks_.
  auto poll_events_for_stream_it =
      [&](auto& stream_it) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        auto& stream_callbacks = stream_it->second;

        auto it = stream_callbacks.begin();
        while (it != stream_callbacks.end()) {
          auto& [event, callback] = *it;

          se::Event::Status s = event->PollForStatus();
          bool keep_looping = true;
          switch (s) {
            case se::Event::Status::kUnknown:
            case se::Event::Status::kError:
              // We don't expect to see these.  Someday maybe propagate
              // a Status error, but for now fail hard.
              LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
              break;
            case se::Event::Status::kPending:
              // If this event is still pending, then all events after it are
              // guaranteed to be pending as well, so we can stop looping.
              keep_looping = false;
              break;
            case se::Event::Status::kComplete:
              free_events_.push_back(std::move(event));
              to_free->push_back({nullptr, std::move(callback)});
              // std::deque::erase() does invalidate iterators, so we can't
              // erase `it` here.  Instead, we'll wait until the end of the loop
              // over stream_callbacks and erase all of the completed events at
              // that point.
              ++it;
              break;
          }

          if (!keep_looping) {
            break;
          }
        }

        // Erase all completed events from stream_callbacks.
        stream_callbacks.erase(stream_callbacks.begin(), it);

        if (stream_callbacks.empty()) {
          // absl::flat_hash_map::erase doesn't invalidate iterators, so this is
          // safe.
          callbacks_.erase(stream_it++);
        } else {
          stream_it++;
        }
      };

  // If `stream` is non-null, poll events just for that stream.  Otherwise, poll
  // events for all streams.
  if (stream != nullptr) {
    auto stream_it = callbacks_.find(stream);
    if (stream_it != callbacks_.end()) {
      poll_events_for_stream_it(stream_it);
    }
  } else {
    for (auto stream_it = callbacks_.begin(); stream_it != callbacks_.end();) {
      poll_events_for_stream_it(stream_it);
    }
  }
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
