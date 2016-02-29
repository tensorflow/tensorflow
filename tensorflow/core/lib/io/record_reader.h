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

#ifndef TENSORFLOW_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_LIB_IO_RECORD_READER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

class RecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit RecordReader(RandomAccessFile* file);

  ~RecordReader();

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, string* record);

 private:
  RandomAccessFile* src_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecordReader);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RECORD_READER_H_
