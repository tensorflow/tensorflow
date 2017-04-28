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

#ifndef TENSORFLOW_LIB_IO_SNAPPY_INPUTBUFFER_H_
#define TENSORFLOW_LIB_IO_SNAPPY_INPUTBUFFER_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

// An SnappyInputBuffer provides support for reading from a file compressed
// using snappy (https://github.com/google/snappy).
//
// A given instance of an SnappyInputBuffer is NOT safe for concurrent use
// by multiple threads
class SnappyInputBuffer : public InputStreamInterface {
 public:
  // Create a SnappyInputBuffer for `file` with a buffer of size
  // `input_buffer_bytes` bytes for reading contents from `file` and another
  // buffer with size `output_buffer_bytes` for caching decompressed contents.
  // Does *not* take ownership of "file".
  SnappyInputBuffer(RandomAccessFile* file, size_t input_buffer_bytes,
                    size_t output_buffer_bytes);

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // Return Status codes:
  // OK:
  //   If successful.
  // OUT_OF_RANGE:
  //   If there are not enough bytes to read before the end of the file.
  // DATA_LOSS:
  //   If uncompression failed or if the file is corrupted.
  // RESOURCE_EXHAUSTED:
  //   If input_buffer_ is smaller in size than a compressed block.
  // others:
  //   If reading from file failed.
  Status ReadNBytes(int64 bytes_to_read, string* result) override;

  int64 Tell() const override;

  Status Reset() override;

 private:
  // Reads data from `file_` and tries to fill up `input_buffer_` if enough
  // unread data is left in `file_`.
  //
  // Looks up `next_in_` to check how much data in `input_buffer_`
  // has already been read. The used data is removed and new data is added to
  // after any unread data in `input_buffer_`.
  // After this call `next_in` points to the start of `input_buffer_`
  // and `avail_in_` stores the number of readable bytes in
  // `input_buffer_`.
  //
  // Returns OutOfRange error if NO data could be read from file. Note that this
  // won't return an OutOfRange if there wasn't sufficient data in file to
  // completely fill up `input_buffer_`.
  Status ReadFromFile();

  // Reads the length of the next compressed block stored in the next 4 bytes at
  // `next_in_`. Uncompresses the next compressed block and writes the output
  // produced to the output_buffer_.
  // Should be called only after the cached output has been consumed.
  Status Inflate();

  // Starts reading bytes at `next_out_` till either `bytes_to_read`
  // bytes have been read or `next_out_` is reached.
  // Returns the number of bytes read and advances the `next_out_`
  // pointer to the next location to read from.
  size_t ReadBytesFromCache(size_t bytes_to_read, string* result);

  // Reads the length of the next *compressed* block and stores in `length`.
  // The length is stored in 4 bytes in little endian notation.
  Status ReadCompressedBlockLength(uint32* length);

  RandomAccessFile* file_;         // Not owned
  int64 file_pos_ = 0;             // Next position to read from in `file_`
  size_t input_buffer_capacity_;   // Size of `input_buffer_`.
                                   // Must be at least as big as the size of
                                   // the largest compressed block.
  size_t output_buffer_capacity_;  // Size of `output_buffer_`

  // Buffer for storing contents read from compressed file.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<char[]> input_buffer_;

  // Buffer for storing inflated contents of `file_`.
  std::unique_ptr<char[]> output_buffer_;

  // Next unread byte in `input_buffer_`.
  char* next_in_;

  // Next unread byte in `output_buffer_`
  char* next_out_;

  // Number of unread bytes bytes available at `next_in_` in `input_buffer_`.
  size_t avail_in_ = 0;

  // Number of unread bytes bytes available at `next_out_` in `output_buffer_`.
  size_t avail_out_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(SnappyInputBuffer);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_SNAPPY_INPUTBUFFER_H_
