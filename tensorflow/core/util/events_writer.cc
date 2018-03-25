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

#include "tensorflow/core/lib/core/errors.h"
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

EventsWriter::~EventsWriter() {
  Close().IgnoreError();  // Autoclose in destructor.
}

Status EventsWriter::Init() { return InitWithSuffix(""); }

Status EventsWriter::InitWithSuffix(const string& suffix) {
  file_suffix_ = suffix;
  return InitIfNeeded();
}

Status EventsWriter::InitIfNeeded() {
  if (recordio_writer_ != nullptr) {
    CHECK(!filename_.empty());
    if (!FileStillExists().ok()) {
      // Warn user of data loss and let .reset() below do basic cleanup.
      if (num_outstanding_events_ > 0) {
        LOG(WARNING) << "Re-initialization, attempting to open a new file, "
                     << num_outstanding_events_ << " events will be lost.";
      }
    } else {
      // No-op: File is present and writer is initialized.
      return Status::OK();
    }
  }

  int64 time_in_seconds = env_->NowMicros() / 1000000;

  filename_ =
      strings::Printf("%s.out.tfevents.%010lld.%s%s", file_prefix_.c_str(),
                      static_cast<int64>(time_in_seconds),
                      port::Hostname().c_str(), file_suffix_.c_str());

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      env_->NewWritableFile(filename_, &recordio_file_),
      "Creating writable file ", filename_);
  recordio_writer_.reset(new io::RecordWriter(recordio_file_.get()));
  if (recordio_writer_ == nullptr) {
    return errors::Unknown("Could not create record writer");
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
    TF_RETURN_WITH_CONTEXT_IF_ERROR(Flush(), "Flushing first event.");
  }
  return Status::OK();
}

string EventsWriter::FileName() {
  if (filename_.empty()) {
    InitIfNeeded().IgnoreError();
  }
  return filename_;
}

void EventsWriter::WriteSerializedEvent(StringPiece event_str) {
  if (recordio_writer_ == nullptr) {
    if (!InitIfNeeded().ok()) {
      LOG(ERROR) << "Write failed because file could not be opened.";
      return;
    }
  }
  num_outstanding_events_++;
  recordio_writer_->WriteRecord(event_str).IgnoreError();
}

// NOTE(touts); This is NOT the function called by the Python code.
// Python calls WriteSerializedEvent(), see events_writer.i.
void EventsWriter::WriteEvent(const Event& event) {
  string record;
  event.AppendToString(&record);
  WriteSerializedEvent(record);
}

Status EventsWriter::Flush() {
  if (num_outstanding_events_ == 0) return Status::OK();
  CHECK(recordio_file_ != nullptr) << "Unexpected NULL file";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(recordio_writer_->Flush(), "Failed to flush ",
                                  num_outstanding_events_, " events to ",
                                  filename_);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(recordio_file_->Sync(), "Failed to sync ",
                                  num_outstanding_events_, " events to ",
                                  filename_);

  // The FileStillExists() condition is necessary because
  // recordio_writer_->Sync() can return OK even if the underlying
  // file has been deleted.  EventWriter.FileDeletionBeforeWriting
  // demonstrates this and will fail if the FileHasDisappeared()
  // condition is removed.
  // Also, we deliberately attempt to Sync() before checking for a
  // disappearing file, in case for some file system File::Exists() is
  // false after File::Open() but before File::Sync().
  TF_RETURN_WITH_CONTEXT_IF_ERROR(FileStillExists(), "Failed to flush ",
                                  num_outstanding_events_, " events to ",
                                  filename_);
  VLOG(1) << "Wrote " << num_outstanding_events_ << " events to disk.";
  num_outstanding_events_ = 0;
  return Status::OK();
}

Status EventsWriter::Close() {
  Status status = Flush();
  if (recordio_file_ != nullptr) {
    Status close_status = recordio_file_->Close();
    if (!close_status.ok()) {
      status = close_status;
    }
    recordio_writer_.reset(nullptr);
    recordio_file_.reset(nullptr);
  }
  num_outstanding_events_ = 0;
  return status;
}

Status EventsWriter::FileStillExists() {
  if (env_->FileExists(filename_).ok()) {
    return Status::OK();
  }
  // This can happen even with non-null recordio_writer_ if some other
  // process has removed the file.
  return errors::Unknown("The events file ", filename_, " has disappeared.");
}

}  // namespace tensorflow
