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
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"

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

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
    used_events_.pop_front();
  }
}

void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_.reset(new Notification);
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
      if (used_events_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(true, &to_free);
      events_still_pending = !used_events_.empty();
    }
    FreeMemory(to_free);
    to_free.clear();

    if (events_still_pending) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}

void EventMgr::QueueInUse(se::Stream* stream, InUse in_use) {
  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
    free_events_.push_back(new se::Event(exec_));
    free_events_.back()->Init();
  }
  se::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->RecordEvent(e).IgnoreError();
  in_use.event = e;
  bool was_empty = used_events_.empty();
  used_events_.push_back(in_use);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
// spikes of up to several hundred outstanding.  (If GPUKernelTracker
// is used to cap pending kernels there should never be more than
// that many.)
//
// NOTE: If all events are on the same stream, no later event will
// complete before an earlier event, except possibly if the earlier
// event transitions to an error state, so there's no advantage in
// looking past the first kPending event.  However, if we're using
// multiple streams there may be some gain in looking deeper.
// As a compromise, PollEvent() calls that are triggered by the queueing
// of a single event never look past the first kPending event.  Consequently
// those calls do an expected constant amount of work, unaffected by the
// length of the pending queue.  Calls coming from the dedicated
// polling thread always sweep the full queue.
void EventMgr::PollEvents(bool is_dedicated_poller,
                          gtl::InlinedVector<InUse, 4>* to_free) {
  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
    se::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case se::Event::Status::kUnknown:
      case se::Event::Status::kError:
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
        break;
      case se::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case se::Event::Status::kComplete:
        // Make a copy of the InUse record so we can free it after releasing
        // the lock
        to_free->push_back(iu);
        free_events_.push_back(iu.event);
        // Mark this InUse record as completed.
        iu.event = nullptr;
    }
  }
  // Then clear any completed InUse records from the front of the queue.
  while (!used_events_.empty()) {
    InUse& iu = used_events_.front();
    if (iu.event == nullptr) {
      used_events_.pop_front();
    } else {
      break;
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
