/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/random_inputstream.h"
#include <memory>

namespace tensorflow {
namespace io {

RandomAccessInputStream::RandomAccessInputStream(RandomAccessFile* file)
    : file_(file) {}

Status RandomAccessInputStream::ReadNBytes(int64 bytes_to_read,
                                           string* result) {
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Cannot read negative number of bytes");
  }
  result->clear();
  result->resize(bytes_to_read);
  char* result_buffer = &(*result)[0];
  StringPiece data;
  Status s = file_->Read(pos_, bytes_to_read, &data, result_buffer);
  if (data.data() != result_buffer) {
    memmove(result_buffer, data.data(), data.size());
  }
  result->resize(data.size());
  if (s.ok() || errors::IsOutOfRange(s)) {
    pos_ += data.size();
  } else {
    return s;
  }
  // If the amount of data we read is less than what we wanted, we return an
  // out of range error. We need to catch this explicitly since file_->Read()
  // would not do so if at least 1 byte is read (b/30839063).
  if (data.size() < bytes_to_read) {
    return errors::OutOfRange("reached end of file");
  }
  return Status::OK();
}

Status RandomAccessInputStream::SkipNBytes(int64 bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can only skip forward, not backwards");
  }
  // Tries to read one byte at future location to see whether we've skipped
  // beyond the end of file or not.
  StringPiece data;
  char scratch;
  Status s = file_->Read(pos_ + bytes_to_skip, 1, &data, &scratch);
  // Advance pointer if OK or OutOfRange status is returned.
  if (s.ok() || errors::IsOutOfRange(s)) {
    pos_ += bytes_to_skip;
  }
  // Passing on the return status from file_->Read(). We can get the following
  //  * OK: We have skipped to some portion in the file. Return OK.
  //  * OutOfRange: Despite b/30839063 an OutOfRange return status means that
  //                no bytes were read and we've skipped beyond EOF. The
  //                return semantics of SkipNBytes dictate we return OutOfRange.
  //  * other: if some other error is encountered we just pass that along.
  return s;
}

}  // namespace io
}  // namespace tensorflow
