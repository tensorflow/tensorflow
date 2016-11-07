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

#include "tensorflow/core/util/events_writer.h"

#include <stddef.h>  // for NULL

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

EventsWriter::EventsWriter(const string& file_prefix)
    // TODO(jeff,sanjay): Pass in env and use that here instead of Env::Default
    : env_(Env::Default()),
      file_prefix_(file_prefix),
      num_outstanding_events_(0) {}

bool EventsWriter::Init() {
  if (recordio_writer_.get() != nullptr) {
    CHECK(!filename_.empty());
    if (FileHasDisappeared()) {
      // Warn user of data loss and let .reset() below do basic cleanup.
      if (num_outstanding_events_ > 0) {
        LOG(WARNING) << "Re-initialization, attempting to open a new file, "
                     << num_outstanding_events_ << " events will be lost.";
      }
    } else {
      // No-op: File is present and writer is initialized.
      return true;
    }
  }

  int64 time_in_seconds = env_->NowMicros() / 1000000;

  filename_ = strings::Printf(
      "%s.out.tfevents.%010lld.%s", file_prefix_.c_str(),
      static_cast<long long>(time_in_seconds), port::Hostname().c_str());
  port::AdjustFilenameForLogging(&filename_);

  Status s = env_->NewWritableFile(filename_, &recordio_file_);
  if (!s.ok()) {
    LOG(ERROR) << "Could not open events file: " << filename_ << ": " << s;
    return false;
  }
  recordio_writer_.reset(new io::RecordWriter(recordio_file_.get()));
  if (recordio_writer_.get() == NULL) {
    LOG(ERROR) << "Could not create record writer";
    return false;
  }
  num_outstanding_events_ = 0;
  VLOG(1) << "Successfully opened events file: " << filename_;
  {
    // Write the first event with the current version, and flush
    // right away so the file contents will be easily determined.

    Event event;
    event.set_wall_time(time_in_seconds);
    event.set_file_version(strings::StrCat(kVersionPrefix, kCurrentVersion));
    WriteEvent(event);
    Flush();
  }
  return true;
}

string EventsWriter::FileName() {
  if (filename_.empty()) {
    Init();
  }
  return filename_;
}

void EventsWriter::WriteSerializedEvent(StringPiece event_str) {
  if (recordio_writer_.get() == NULL) {
    if (!Init()) {
      LOG(ERROR) << "Write failed because file could not be opened.";
      return;
    }
  }
  num_outstanding_events_++;
  recordio_writer_->WriteRecord(event_str);
}

// NOTE(touts); This is NOT the function called by the Python code.
// Python calls WriteSerializedEvent(), see events_writer.i.
void EventsWriter::WriteEvent(const Event& event) {
  string record;
  event.AppendToString(&record);
  WriteSerializedEvent(record);
}

bool EventsWriter::Flush() {
  if (num_outstanding_events_ == 0) return true;
  CHECK(recordio_file_.get() != NULL) << "Unexpected NULL file";

  if (!recordio_writer_->Flush().ok()) {
    LOG(ERROR) << "Failed to flush " << num_outstanding_events_ << " events to "
               << filename_;
    return false;
  }

  // The FileHasDisappeared() condition is necessary because
  // recordio_writer_->Sync() can return true even if the underlying
  // file has been deleted.  EventWriter.FileDeletionBeforeWriting
  // demonstrates this and will fail if the FileHasDisappeared()
  // condition is removed.
  // Also, we deliberately attempt to Sync() before checking for a
  // disappearing file, in case for some file system File::Exists() is
  // false after File::Open() but before File::Sync().
  if (!recordio_file_->Flush().ok() || !recordio_file_->Sync().ok() ||
      FileHasDisappeared()) {
    LOG(ERROR) << "Failed to flush " << num_outstanding_events_ << " events to "
               << filename_;
    return false;
  }
  VLOG(1) << "Wrote " << num_outstanding_events_ << " events to disk.";
  num_outstanding_events_ = 0;
  return true;
}

bool EventsWriter::Close() {
  bool return_value = Flush();
  if (recordio_file_.get() != NULL) {
    Status s = recordio_file_->Close();
    if (!s.ok()) {
      LOG(ERROR) << "Error when closing previous event file: " << filename_
                 << ": " << s;
      return_value = false;
    }
    recordio_writer_.reset(NULL);
    recordio_file_.reset(NULL);
  }
  num_outstanding_events_ = 0;
  return return_value;
}

bool EventsWriter::FileHasDisappeared() {
  if (env_->FileExists(filename_).ok()) {
    return false;
  } else {
    // This can happen even with non-null recordio_writer_ if some other
    // process has removed the file.
    LOG(ERROR) << "The events file " << filename_ << " has disappeared.";
    return true;
  }
}

}  // namespace tensorflow
