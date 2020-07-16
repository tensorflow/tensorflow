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

#ifndef TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_OUTPUTBUFFER_H_
#define TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_OUTPUTBUFFER_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

// Compresses input data using Snappy (https://github.com/google/snappy) and
// writes to `file`.
//
// The input data is cached in a buffer of size `input_buffer_bytes`. When the
// buffer does not have enough available space to fit new data (in the call to
// `Write`), the contents of the buffer are compressed and sent to the output
// buffer.
//
// The compressed output is buffered in a buffer of size `output_buffer_bytes`
// which gets flushed to file when full.
//
// Output file format:
// The output file consists of a sequence of compressed blocks. Each block
// starts with a 4 byte header which stores the length (in bytes) of the
// _compressed_ block _excluding_ this header. The compressed
// block (excluding the 4 byte header) is a valid snappy block and can directly
// be uncompressed using Snappy_Uncompress.
class SnappyOutputBuffer : public WritableFile {
 public:
  // Create an SnappyOutputBuffer for `file` with two buffers that cache the
  // 1. input data to be deflated
  // 2. the deflated output
  // with sizes `input_buffer_bytes` and `output_buffer_bytes` respectively.
  // Does not take ownership of `file`.
  SnappyOutputBuffer(WritableFile* file, int32 input_buffer_bytes,
                     int32 output_buffer_bytes);

  // Per convention, the dtor does not call Flush() or Close(). We expect the
  // caller to call those manually when done.
  ~SnappyOutputBuffer() override;

  // Adds `data` to the compression pipeline.
  //
  // The input data is buffered internally and will be written to disk at a
  // later time. To immediately write contents to file call `Flush()`.
  Status Append(StringPiece data) override;

#if defined(PLATFORM_GOOGLE)
  Status Append(const absl::Cord& cord) override;
#endif

  // Compresses any buffered input and writes all output to file. This must be
  // called before the destructor to avoid any data loss.
  //
  // Contrary to `Flush()` this informs snappy that it should not expect any
  // further input.
  //
  // After calling this, any further calls to `Write()`, `Flush()` or `Close()`
  // will fail.
  Status Close() override;

  // Returns the name of the underlying file.
  Status Name(StringPiece* result) const override;

  // Deflates any cached input, writes all output to file and syncs it.
  Status Sync() override;

  // Returns the write position in the underlying file. The position does not
  // reflect buffered, un-flushed data.
  Status Tell(int64* position) override;

  // Adds `data` to the compression pipeline.
  //
  // The input data is buffered in `input_buffer_` and is compressed in bulk
  // when the buffer gets full. The compressed output may not be immediately
  // written to file but rather buffered in `output_buffer_` and gets written
  // to file when the buffer is full.
  //
  // To immediately write contents to file call `Flush()`.
  Status Write(StringPiece data);

  // Compresses any cached input and writes all output to file. This must be
  // called before the destructor to avoid any data loss.
  Status Flush();

 private:
  // Appends `data` to `input_buffer_`.
  // Throws if `data.size()` > AvailableInputSpace().
  void AddToInputBuffer(StringPiece data);

  // Appends `data` to `output_buffer_`. Flushes buffer contents to file when
  // buffer gets full.
  Status AddToOutputBuffer(const char* data, size_t length);

  // Returns the total space available in `input_buffer_`.
  int32 AvailableInputSpace() const;

  // Deflate contents in input_buffer_ and store results in output_buffer_.
  // The contents of output stream are written to file if more space is needed.
  //
  // Note: This method does not flush contents to file.
  // Returns non-ok status if writing contents to file fails.
  Status DeflateBuffered();

  // Appends contents of `output_buffer_` to `file_`.
  // Returns non-OK status if writing to file fails.
  Status FlushOutputBufferToFile();

  // Compresses `avail_in_` bytes at `next_in_` location in `input_buffer_` and
  // writes the results to output using `AddToOutputBuffer`.
  // Returns non-OK status if writing to file failed.
  Status Deflate();

  WritableFile* file_;  // Not owned

  // Buffer for storing contents read from input `file_`.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<char[]> input_buffer_;
  size_t input_buffer_capacity_;
  char* next_in_;
  size_t avail_in_ = 0;

  // Buffer for storing deflated contents of `file_`.
  std::unique_ptr<char[]> output_buffer_;
  size_t output_buffer_capacity_;
  char* next_out_;
  size_t avail_out_;

  TF_DISALLOW_COPY_AND_ASSIGN(SnappyOutputBuffer);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_OUTPUTBUFFER_H_
