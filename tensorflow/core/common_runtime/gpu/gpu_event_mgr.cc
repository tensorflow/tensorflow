/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/stream.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

EventMgr::EventMgr(gpu::StreamExecutor* se)
    : exec_(se),
      // threadpool_ has 1 thread for the polling loop, and one to execute
      // event callback functions. Maybe we should have more?
      threadpool_(Env::Default(), "GPU_Event_Manager", 2) {
  threadpool_.Schedule([this]() { PollLoop(); });
}

EventMgr::~EventMgr() {
  stop_polling_.Notify();
  // Shut down the backup polling loop.
  polling_stopped_.WaitForNotification();

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    delete ue->mem;
    if (ue->bufrec.buf) {
      ue->bufrec.alloc->DeallocateRaw(ue->bufrec.buf);
    }
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
    used_events_.pop_front();
  }
}

// This polling loop runs at a relatively low frequency. Most calls to
// PollEvents() should come directly from Compute() via
// ThenDeleteTensors().  This function's purpose is to ensure that
// even if no more GPU operations are being requested, we still
// eventually clear the queue. It seems to prevent some tensorflow
// programs from stalling for reasons not yet understood.
void EventMgr::PollLoop() {
  while (!stop_polling_.HasBeenNotified()) {
    Env::Default()->SleepForMicroseconds(1 * 1000);
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      PollEvents(true, &to_free);
    }
    FreeMemory(to_free);
  }
  polling_stopped_.Notify();
}

void EventMgr::QueueInUse(gpu::Stream* stream, InUse iu) {
  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
    free_events_.push_back(new gpu::Event(exec_));
    free_events_.back()->Init();
  }
  gpu::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->ThenRecordEvent(e);
  iu.event = e;
  used_events_.push_back(iu);
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
// spikes of up to several hundred outstanding.
//
// NOTE: If all events are on the same stream, no later event will
// complete before an earlier event, except possibly if the earlier
// event transitions to an error state, so there's no advantage in
// looking past the first kPending event.  However, if we're using
// multiple streams there may be some gain in looking deeper.
// As a compromise, PollEvent() calls that are triggered by the queueing
// of a single event never look past the first kPending event.  Calls
// coming from the dedicated polling thread always sweep the full queue.
//
// Note that allowing the queue to grow very long could cause overall
// GPU memory use to spike needlessly.  An alternative strategy would
// be to throttle new Op execution until the pending event queue
// clears.
void EventMgr::PollEvents(bool is_dedicated_poller,
                          gtl::InlinedVector<InUse, 4>* to_free) {
  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
    gpu::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case gpu::Event::Status::kUnknown:
      case gpu::Event::Status::kError:
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
        break;
      case gpu::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case gpu::Event::Status::kComplete:
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

}  // namespace tensorflow
