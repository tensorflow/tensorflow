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

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

EventMgr::EventMgr(gpu::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      deferred_bytes_threshold_(gpu_options.deferred_deletion_bytes()
                                    ? gpu_options.deferred_deletion_bytes()
                                    : 8 * 1048576),
      accumulated_stream_(nullptr),
      accumulated_tensors_(new TensorReferenceVector),
      accumulated_tensor_bytes_(0),
      // threadpool_ has 1 thread for the polling loop, and one to execute
      // event callback functions. Maybe we should have more?
      threadpool_(Env::Default(), "GPU_Event_Manager", 2) {
  StartPollingLoop();
}

EventMgr::~EventMgr() {
  StopPollingLoop();

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  for (auto& t : *(accumulated_tensors_)) {
    t.Unref();
  }
  delete accumulated_tensors_;
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->mem != nullptr) {
      for (auto& t : *(ue->mem)) {
        t.Unref();
      }
      delete ue->mem;
    }
    if (ue->bufrec.buf) {
      if (LogMemory::IsEnabled()) {
        LogMemory::RecordRawDeallocation(ue->bufrec.operation,
                                         ue->bufrec.step_id, ue->bufrec.buf,
                                         ue->bufrec.alloc, false);
      }
      ue->bufrec.alloc->DeallocateRaw(ue->bufrec.buf);
    }
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
    used_events_.pop_front();
  }
}

void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_.get() == nullptr);
  stop_polling_.reset(new Notification);
  polling_stopped_.reset(new Notification);
  threadpool_.Schedule([this]() { PollLoop(); });
}

void EventMgr::StopPollingLoop() {
  if (stop_polling_.get()) {
    stop_polling_->Notify();
    polling_stopped_->WaitForNotification();
    stop_polling_.reset(nullptr);
    polling_stopped_.reset(nullptr);
  }
}

void EventMgr::ThenDeleteTensors(perftools::gputools::Stream* stream,
                                 const TensorReferenceVector& tensors) {
  mutex_lock l(mu_);
  // TODO(jeff): We currently keep one accumulated_tensors_ object.
  // If we start to use multiple streams heavily, we might want to keep
  // separate vectors/byte counters per stream
  if (!accumulated_tensors_->empty() && stream != accumulated_stream_) {
    FlushAccumulatedTensors();
  }
  accumulated_stream_ = stream;
  for (const auto& t : tensors) {
    // accumulated_tensors_ takes over ownership of the reference to "t"
    accumulated_tensors_->push_back(t);
    accumulated_tensor_bytes_ += t.TotalBytes();
  }
  if (accumulated_tensor_bytes_ >= deferred_bytes_threshold_) {
    FlushAccumulatedTensors();
  }
}

void EventMgr::FlushAccumulatedTensors() {
  DCHECK(!accumulated_tensors_->empty());
  DCHECK(accumulated_stream_ != nullptr);
  QueueTensors(accumulated_stream_, accumulated_tensors_);
  accumulated_tensors_ = new TensorReferenceVector;
  accumulated_tensor_bytes_ = 0;
  accumulated_stream_ = nullptr;
}

// A polling loop to detect completion of GPU events.  There's a
// tradeoff between achieving low latency detection, which argues for
// little delay between calls, and minimizing CPU use and lock
// contention, which argue for longer delay.  The current strategy is
// to poll frequently when the queue is non-empty, and infrequently
// otherwise.
void EventMgr::PollLoop() {
  const int32 kPollingDelayUsecs = 10;
  const int32 kPollingSuspendMsecs = 1;
  bool queue_empty = false;
  while (!stop_polling_->HasBeenNotified()) {
    if (queue_empty) {
      mutex_lock l(mu_);
      WaitForMilliseconds(&l, &events_pending_, kPollingSuspendMsecs);
    } else {
      Env::Default()->SleepForMicroseconds(kPollingDelayUsecs);
    }
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      PollEvents(true, &to_free);
      queue_empty = used_events_.empty();
    }
    FreeMemory(to_free);
  }
  polling_stopped_->Notify();
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
  bool was_empty = used_events_.empty();
  used_events_.push_back(iu);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
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
