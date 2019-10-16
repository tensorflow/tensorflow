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

#include "tensorflow/core/framework/reader_base.h"

#include "tensorflow/core/framework/reader_base.pb.h"
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

Status ReaderBase::SerializeState(tstring* state) {
  mutex_lock lock(mu_);
  return SerializeStateLocked(state);
}

Status ReaderBase::SerializeStateLocked(tstring* state) {
  return errors::Unimplemented("Reader SerializeState");
}

Status ReaderBase::RestoreState(const tstring& state) {
  mutex_lock lock(mu_);
  Status status = RestoreStateLocked(state);
  if (!status.ok()) {
    ResetLocked().IgnoreError();
  }
  return status;
}

Status ReaderBase::RestoreStateLocked(const tstring& state) {
  return errors::Unimplemented("Reader RestoreState");
}

int64 ReaderBase::ReadUpTo(const int64 num_records, QueueInterface* queue,
                           std::vector<tstring>* keys,
                           std::vector<tstring>* values,
                           OpKernelContext* context) {
  mutex_lock lock(mu_);
  int64 records_produced_this_call = 0;
  while (true) {
    // Records produced by this iteration of the ReadUpToLocked call.
    int64 num_records_produced = 0;
    int64 remaining = num_records - records_produced_this_call;
    if (remaining == 0) {
      return records_produced_this_call;
    }
    if (!work_in_progress()) {
      work_ = GetNextWorkLocked(queue, context);
      if (!context->status().ok()) {
        return records_produced_this_call;
      }
      Status status = OnWorkStartedLocked();
      if (status.ok()) {
        work_started_++;
      } else {
        context->SetStatus(status);
        return records_produced_this_call;
      }
    }
    bool at_end = false;

    Status status =
        ReadUpToLocked(remaining, keys, values, &num_records_produced, &at_end);
    // This call so far.
    records_produced_this_call += num_records_produced;

    // In total, over the lifetime of the ReaderBase.
    num_records_produced_ += num_records_produced;

    if (!at_end && status.ok() && num_records_produced == 0) {
      status = errors::Internal(
          "ReadManyLocked() for ", name(),
          " must set *at_end=true, *num_produced > 0 or return an error.");
      context->SetStatus(status);
      return records_produced_this_call;
    }
    if (status.ok() && at_end) {
      status = OnWorkFinishedLocked();
      work_finished_ = work_started_;
      if (records_produced_this_call > 0) {
        return records_produced_this_call;
      }
    }
    if (!status.ok()) {
      context->SetStatus(status);
      return records_produced_this_call;
    }
  }
}

// Default implementation just reads one record at a time.
Status ReaderBase::ReadUpToLocked(int64 num_records, std::vector<tstring>* keys,
                                  std::vector<tstring>* values, int64* num_read,
                                  bool* at_end) {
  bool produced = false;
  tstring key;
  tstring value;
  Status status = ReadLocked(&key, &value, &produced, at_end);
  if (produced) {
    keys->push_back(std::move(key));
    values->push_back(std::move(value));
    *num_read = 1;
  } else {
    *num_read = 0;
  }
  return status;
}

void ReaderBase::Read(QueueInterface* queue, tstring* key, tstring* value,
                      OpKernelContext* context) {
  mutex_lock lock(mu_);
  while (true) {
    if (!work_in_progress()) {
      work_ = GetNextWorkLocked(queue, context);
      if (!context->status().ok()) {
        return;
      }
      Status status = OnWorkStartedLocked();
      if (status.ok()) {
        work_started_++;
      } else {
        context->SetStatus(status);
        return;
      }
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
      status = errors::Internal(
          "ReadLocked() for ", name(),
          " set *produced=true *and* returned an error: ", status.ToString());
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

string ReaderBase::GetNextWorkLocked(QueueInterface* queue,
                                     OpKernelContext* context) const {
  string work;
  Notification n;
  queue->TryDequeue(
      context, [context, &n, &work](const QueueInterface::Tuple& tuple) {
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
            work = tuple[0].flat<tstring>()(0);
          }
        }
        n.Notify();
      });
  n.WaitForNotification();
  return work;
}

void ReaderBase::SaveBaseState(ReaderBaseState* state) const {
  state->Clear();
  state->set_work_started(work_started_);
  state->set_work_finished(work_finished_);
  state->set_num_records_produced(num_records_produced_);
  // Unfortunately, external proto does not accept string_view.
#if defined(PLATFORM_GOOGLE)
  // TODO(dero): Remove NOLINT after USE_TSTRING is enabled.  The external proto
  // compiler does not create an overloaded set method that accepts
  // absl::string_view, and string_view to std::string is an explicit
  // conversion.
  state->set_current_work(StringPiece(work_));  // NOLINT
#else
  state->set_current_work(string(work_));
#endif
}

tstring ReaderBase::KeyName(const tstring& key) const {
  return strings::StrCat(current_work(), ":", key);
}

Status ReaderBase::RestoreBaseState(const ReaderBaseState& state) {
  work_started_ = state.work_started();
  work_finished_ = state.work_finished();
  num_records_produced_ = state.num_records_produced();
  work_ = state.current_work();
  if (work_started_ < 0 || work_finished_ < 0 || num_records_produced_ < 0) {
#if defined(__ANDROID__) || defined(__EMSCRIPTEN__)
    const string debug_string = "<debug state not available>";
#else
    const string debug_string = state.DebugString();
#endif
    return errors::InvalidArgument(
        "Unexpected negative value when restoring in ", name(), ": ",
        debug_string);
  }
  if (work_started_ > work_finished_) {
#if defined(__ANDROID__) || (__EMSCRIPTEN__)
    const string debug_string = "<debug state not available>";
#else
    const string debug_string = state.DebugString();
#endif
    return errors::InvalidArgument(
        "Inconsistent work started vs. finished when restoring in ", name(),
        ": ", debug_string);
  }
  return Status::OK();
}

}  // namespace tensorflow
