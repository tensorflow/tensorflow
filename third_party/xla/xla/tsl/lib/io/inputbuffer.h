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

#ifndef XLA_TSL_LIB_IO_INPUTBUFFER_H_
#define XLA_TSL_LIB_IO_INPUTBUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/coding.h"

namespace tsl {
namespace io {

// An InputBuffer provides a buffer on top of a RandomAccessFile.
// A given instance of an InputBuffer is NOT safe for concurrent use
// by multiple threads
class InputBuffer {
 public:
  // Create an InputBuffer for "file" with a buffer size of
  // "buffer_bytes" bytes.  'file' must outlive *this.
  InputBuffer(RandomAccessFile* file, size_t buffer_bytes);
  ~InputBuffer();

  // Read one text line of data into "*result" until end-of-file or a
  // \n is read.  (The \n is not included in the result.)  Overwrites
  // any existing data in *result.
  //
  // If successful, returns OK.  If we are already at the end of the
  // file, we return an OUT_OF_RANGE error.  Otherwise, we return
  // some other non-OK status.
  template <typename T>
  absl::Status ReadLine(T* result);

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // If successful, returns OK.  If we there are not enough bytes to
  // read before the end of the file, we return an OUT_OF_RANGE error.
  // Otherwise, we return some other non-OK status.
  absl::Status ReadNBytes(int64_t bytes_to_read, std::string* result);

  // An overload that writes to char*.  Caller must ensure result[0,
  // bytes_to_read) is valid to be overwritten.  Returns OK iff "*bytes_read ==
  // bytes_to_read".
  absl::Status ReadNBytes(int64_t bytes_to_read, char* result,
                          size_t* bytes_read);

  // Reads a single varint32.
  absl::Status ReadVarint32(uint32* result);

  // Reads a single varint64.
  absl::Status ReadVarint64(uint64* result);

  // Like ReadNBytes() without returning the bytes read.
  absl::Status SkipNBytes(int64_t bytes_to_skip);

  // Seek to this offset within the file.
  //
  // If we seek to somewhere within our pre-buffered data, we will re-use what
  // data we can.  Otherwise, Seek() throws out the current buffer and the next
  // read will trigger a File::Read().
  absl::Status Seek(int64_t position);

  // Provides a hint about future reads, which may improve their performance.
  absl::Status Hint(int64_t bytes_to_read);

  // Returns the position in the file.
  int64_t Tell() const { return file_pos_ - num_remaining_bytes(); }

  // Returns the underlying RandomAccessFile.
  RandomAccessFile* file() const { return file_; }

 private:
  absl::Status FillBuffer();

  // Internal slow-path routine used by ReadVarint32().
  absl::Status ReadVarint32Fallback(uint32* result);

  // Internal slow-path routine used by ReadVarint64().
  absl::Status ReadVarint64Fallback(uint64* result);

  // Helper method for reading a varint which can span at max `max_bytes`.
  // If the varint is longer, a DataLoss error status is returned.
  // If end of file is reached while reading, OutOfRange error is returned.
  template <typename T>
  absl::Status ReadVarintFallback(T* result, int max_bytes);

  // The buffer itself.
  char* buf() { return buf_.get(); }
  const char* buf() const { return buf_.get(); }

  // Number of bytes remaining in the buffer.
  int num_remaining_bytes() const { return limit_ - pos_; }

  RandomAccessFile* const file_ = nullptr;  // Not owned
  int64_t file_pos_ = 0;               // Next position to read from in "file_"
  const int size_ = 0;                 // Size of "buf_"
  const std::unique_ptr<char[]> buf_;  // The buffer itself. Must not be null.
  // [pos_,limit_) hold the "limit_ - pos_" bytes just before "file_pos_".
  char* pos_ =
      nullptr;  // Current position in "buf". Must be in [buf(), buf() + size_].
  char* limit_ = nullptr;  // Just past end of valid data in "buf". Must be in
                           // [buf(), buf() + size_].

  InputBuffer(const InputBuffer&) = delete;
  void operator=(const InputBuffer&) = delete;
};

// Implementation details.

// Explicit instantiations defined in inputbuffer.cc.
extern template Status InputBuffer::ReadLine<std::string>(std::string* result);
extern template Status InputBuffer::ReadLine<tstring>(tstring* result);

// Inlined for performance.
inline absl::Status InputBuffer::ReadVarint32(uint32* result) {
  if (pos_ + core::kMaxVarint32Bytes <= limit_) {
    // Fast path: directly parse from buffered data.
    // Reads strictly from the range [pos_, limit_).
    const char* const new_pos = core::GetVarint32Ptr(pos_, limit_, result);
    if (new_pos == nullptr) return errors::OutOfRange("Parsed past limit.");
    const int offset = new_pos - buf();
    pos_ = buf() + offset;
    return absl::OkStatus();
  } else {
    return ReadVarint32Fallback(result);
  }
}

// Inlined for performance.
inline absl::Status InputBuffer::ReadVarint64(uint64* result) {
  if (pos_ + core::kMaxVarint64Bytes <= limit_) {
    // Fast path: directly parse from buffered data.
    // Reads strictly from the range [pos_, limit_).
    const char* const new_pos = core::GetVarint64Ptr(pos_, limit_, result);
    if (new_pos == nullptr) return errors::OutOfRange("Parsed past limit.");
    const int offset = new_pos - buf();
    pos_ = buf() + offset;
    return absl::OkStatus();
  } else {
    return ReadVarint64Fallback(result);
  }
}

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_INPUTBUFFER_H_
