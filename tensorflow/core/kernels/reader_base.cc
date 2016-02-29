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

#include "tensorflow/core/kernels/reader_base.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// ReaderBase ------------------------------------------------------

ReaderBase::ReaderBase(const string& name) : name_(name) {}

int64 ReaderBase::NumRecordsProduced() {
  mutex_lock lock(mu_);
  return num_records_produced_;
}

int64 ReaderBase::NumWorkUnitsCompleted() {
  mutex_lock lock(mu_);
  return work_finished_;
}

Status ReaderBase::Reset() {
  mutex_lock lock(mu_);
  return ResetLocked();
}

Status ReaderBase::ResetLocked() {
  work_started_ = 0;
  work_finished_ = 0;
  num_records_produced_ = 0;
  work_.clear();
  return Status::OK();
}

Status ReaderBase::SerializeState(string* state) {
  mutex_lock lock(mu_);
  return SerializeStateLocked(state);
}

Status ReaderBase::SerializeStateLocked(string* state) {
  return errors::Unimplemented("Reader SerializeState");
}

Status ReaderBase::RestoreState(const string& state) {
  mutex_lock lock(mu_);
  Status status = RestoreStateLocked(state);
  if (!status.ok()) {
    ResetLocked();
  }
  return status;
}

Status ReaderBase::RestoreStateLocked(const string& state) {
  return errors::Unimplemented("Reader RestoreState");
}

void ReaderBase::Read(QueueInterface* queue, string* key, string* value,
                      OpKernelContext* context) {
  mutex_lock lock(mu_);
  while (true) {
    if (!work_in_progress()) {
      GetNextWorkLocked(queue, context);
      if (!context->status().ok()) return;
    }

    bool produced = false;
    bool at_end = false;
    Status status = ReadLocked(key, value, &produced, &at_end);

    if (!at_end && status.ok() && !produced) {
      status = errors::Internal(
          "ReadLocked() for ", name(),
          " must set *at_end=true, *produced=true, or return an error.");
    }
    if (!status.ok() && produced) {
      status = errors::Internal("ReadLocked() for ", name(),
                                " set *produced=true *and* returned an error: ",
                                status.ToString());
    }
    if (status.ok() && at_end) {
      status = OnWorkFinishedLocked();
      work_finished_ = work_started_;
    }
    if (!status.ok()) {
      context->SetStatus(status);
      return;
    }
    if (produced) {
      ++num_records_produced_;
      return;
    }
  }
}

void ReaderBase::GetNextWorkLocked(QueueInterface* queue,
                                   OpKernelContext* context) {
  Notification n;
  queue->TryDequeue(
      context, [this, context, &n](const QueueInterface::Tuple& tuple) {
        if (context->status().ok()) {
          if (tuple.size() != 1) {
            context->SetStatus(
                errors::InvalidArgument("Expected single component queue"));
          } else if (tuple[0].dtype() != DT_STRING) {
            context->SetStatus(errors::InvalidArgument(
                "Expected queue with single string component"));
          } else if (tuple[0].NumElements() != 1) {
            context->SetStatus(errors::InvalidArgument(
                "Expected to dequeue a one-element string tensor"));
          } else {
            work_ = tuple[0].flat<string>()(0);
            ++work_started_;
            Status status = OnWorkStartedLocked();
            if (!status.ok()) {
              context->SetStatus(status);
              --work_started_;
            }
          }
        }
        n.Notify();
      });
  n.WaitForNotification();
}

void ReaderBase::SaveBaseState(ReaderBaseState* state) const {
  state->Clear();
  state->set_work_started(work_started_);
  state->set_work_finished(work_finished_);
  state->set_num_records_produced(num_records_produced_);
  state->set_current_work(work_);
}

Status ReaderBase::RestoreBaseState(const ReaderBaseState& state) {
  work_started_ = state.work_started();
  work_finished_ = state.work_finished();
  num_records_produced_ = state.num_records_produced();
  work_ = state.current_work();
  if (work_started_ < 0 || work_finished_ < 0 || num_records_produced_ < 0) {
    return errors::InvalidArgument(
        "Unexpected negative value when restoring in ", name(), ": ",
        state.DebugString());
  }
  if (work_started_ > work_finished_) {
    return errors::InvalidArgument(
        "Inconsistent work started vs. finished when restoring in ", name(),
        ": ", state.DebugString());
  }
  return Status::OK();
}

}  // namespace tensorflow
