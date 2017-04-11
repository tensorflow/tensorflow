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

#ifndef TENSORFLOW_UTIL_EVENTS_WRITER_H_
#define TENSORFLOW_UTIL_EVENTS_WRITER_H_

#include <memory>
#include <string>
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

class EventsWriter {
 public:
#ifndef SWIG
  // Prefix of version string present in the first entry of every event file.
  static constexpr const char* kVersionPrefix = "brain.Event:";
  static constexpr const int kCurrentVersion = 2;
#endif

  // Events files typically have a name of the form
  //   '/some/file/path/my.file.out.events.[timestamp].[hostname][suffix]'
  // To create and EventWriter, the user should provide file_prefix =
  //   '/some/file/path/my.file'
  // The EventsWriter will append '.out.events.[timestamp].[hostname][suffix]'
  // to the ultimate filename once Init() is called.
  // Note that it is not recommended to simultaneously have two
  // EventWriters writing to the same file_prefix.
  explicit EventsWriter(const string& file_prefix);
  ~EventsWriter() { Close(); }  // Autoclose in destructor.

  // Sets the event file filename and opens file for writing.  If not called by
  // user, will be invoked automatically by a call to FileName() or Write*().
  // Returns false if the file could not be opened.  Idempotent: if file exists
  // and is open this is a no-op.  If on the other hand the file was opened,
  // but has since disappeared (e.g. deleted by another process), this will open
  // a new file with a new timestamp in its filename.
  bool Init() { return InitWithSuffix(""); }
  bool InitWithSuffix(const string& suffix) {
    file_suffix_ = suffix;
    return InitIfNeeded();
  }

  // Returns the filename for the current events file:
  // filename_ = [file_prefix_].out.events.[timestamp].[hostname][suffix]
  string FileName();

  // Append "event" to the file.  The "tensorflow::" part is for swig happiness.
  void WriteEvent(const tensorflow::Event& event);

  // Append "event_str", a serialized Event, to the file.
  // Note that this function does NOT check that de-serializing event_str
  // results in a valid Event proto.  The tensorflow:: bit makes SWIG happy.
  void WriteSerializedEvent(tensorflow::StringPiece event_str);

  // EventWriter automatically flushes and closes on destruction, but
  // these two methods are provided for users who want to write to disk sooner
  // and/or check for success.
  //   Flush() pushes outstanding events to disk.  Returns false if the
  // events file could not be created, or if the file exists but could not
  // be written too.
  //   Close() calls Flush() and then closes the current events file.
  // Returns true only if both the flush and the closure were successful.
  bool Flush();
  bool Close();

 private:
  bool FileHasDisappeared();  // True if event_file_path_ does not exist.
  bool InitIfNeeded();

  Env* env_;
  const string file_prefix_;
  string file_suffix_;
  string filename_;
  std::unique_ptr<WritableFile> recordio_file_;
  std::unique_ptr<io::RecordWriter> recordio_writer_;
  int num_outstanding_events_;
  TF_DISALLOW_COPY_AND_ASSIGN(EventsWriter);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_EVENTS_WRITER_H_
