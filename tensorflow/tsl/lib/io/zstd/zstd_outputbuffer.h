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

#ifndef TENSORFLOW_TSL_LIB_IO_ZSTD_ZSTD_OUTPUTBUFFER_H_
#define TENSORFLOW_TSL_LIB_IO_ZSTD_ZSTD_OUTPUTBUFFER_H_

#include <zstd.h>

#include <string>

#include "tensorflow/tsl/lib/io/zstd/zstd_compression_options.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/platform.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {
namespace io {

// forward declaration
struct ZstdStreamDef;

// Compresses input data using Zstd (https://github.com/facebook/zstd) and
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

class ZstdOutputBuffer : public WritableFile {
 public:
  // Create an ZstdOutputBuffer for `file` with two buffers that cache the
  // 1. input data to be deflated
  // 2. the deflated output
  // with sizes `input_buffer_bytes` and `output_buffer_bytes` respectively.
  // Does not take ownership of `file`.
  ZstdOutputBuffer(WritableFile* file, int32 input_buffer_bytes,
                   int32 output_buffer_bytes,
                   const ZstdCompressionOptions& zstd_options);

  // Per convention, the dtor does not call Flush() or Close(). We expect the
  // caller to call those manually when done.
  ~ZstdOutputBuffer() override;

  // Adds `data` to the compression pipeline.
  //
  // The input data is buffered internally and will be written to disk at a
  // later time. To immediately write contents to file call `Flush()`.
  Status Append(StringPiece data) override;

#if defined(TF_CORD_SUPPORT)
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

  // Compresses any cached input and writes all output to file. This must be
  // called before the destructor to avoid any data loss.
  Status Flush();

 private:
  void InitZstdBuffer();
  // Appends `data` to `input_buffer_`.
  // Throws if `data.size()` > AvailableInputSpace().
  void AddToInputBuffer(StringPiece data);

  // Returns the total space available in `input_buffer_`.
  int32 AvailableInputSpace() const;

  Status FlushOutputBufferToFile();

  Status DeflateBuffered(ZSTD_EndDirective end_directive);

  Status Deflate(ZSTD_EndDirective end_directive);

  WritableFile* file_;  // Not owned
  size_t input_buffer_capacity_;
  size_t output_buffer_capacity_;
  ZSTD_CCtx* context_;

  std::unique_ptr<ZstdStreamDef> zstd_stream_;

  const ZstdCompressionOptions zstd_options_;

  TF_DISALLOW_COPY_AND_ASSIGN(ZstdOutputBuffer);
};

}  // namespace io
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_IO_ZSTD_ZSTD_OUTPUTBUFFER_H_
