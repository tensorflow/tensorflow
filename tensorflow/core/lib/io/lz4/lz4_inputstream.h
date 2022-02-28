/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_INPUTSTREAM_H_
#define TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_INPUTSTREAM_H_

#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/lz4/lz4_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <lz4.h>

namespace tensorflow {
namespace io {

class Lz4InputStream : public InputStreamInterface {
 public:
  // Creates a Lz4InputStream for `input_stream`.
  //
  // Takes ownership  of `input_stream` iff `owns_input_stream` is true.
  Lz4InputStream(InputStreamInterface* input_stream, size_t input_buffer_bytes,
                  size_t output_buffer_bytes,
                  const Lz4CompressionOptions& lz4_options,
                  bool owns_input_stream);

  // Equivalent to the previous constructor with owns_input_stream=false.
  Lz4InputStream(InputStreamInterface* input_stream, size_t input_buffer_bytes,
                  size_t output_buffer_bytes,
                  const Lz4CompressionOptions& lz4_options);

  ~Lz4InputStream() override;

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // Return Status codes:
  // OK:           If successful.
  // OUT_OF_RANGE: If there are not enough bytes to read before
  //               the end of the stream.
  // ABORTED:      If inflate() fails, we return the error code with the
  //               error message in `z_stream_->msg`.
  // others:       If reading from stream failed.
  Status ReadNBytes(int64 bytes_to_read, tstring* result) override;

#if defined(TF_CORD_SUPPORT)
  Status ReadNBytes(int64 bytes_to_read, absl::Cord* result) override;
#endif

  int64 Tell() const override;

  Status Reset() override;

 private:
  // Decompress the next chunk of data and place the data into the cache.
  Status Inflate();

  Status ReadFromStream();

  // There may be bytes leftover from last read. We read them so that we don't
  // lose them, and we optimize resources.
  size_t ReadBytesFromCache(size_t bytes_to_read, tstring* result);

  void InitLz4Buffer();

  const bool owns_input_stream_;
  InputStreamInterface* input_stream_;
  std::unique_ptr<char[]> input_buffer_;
  size_t input_buffer_capacity_;  // Size of input_buffer_
  char* next_in_byte_;            // Next unread byte to decompress
  size_t avail_in_;  // Number of bytes available to be decompressed
  LZ4_inBuffer lz4_input_buffer_;

  std::unique_ptr<char[]> output_buffer_;  // Inflated buffer
  size_t output_buffer_capacity_;          // Size of output_buffer_
  char* next_unread_byte_;                  // Next unread byte in output_buffer_
  // bytes left in the output_buffer_ not yet read.
  size_t unread_bytes_;

  // Specifies the number of decompressed bytes currently read.
  size_t bytes_read_;

  size_t last_return_;

  const Lz4CompressionOptions lz4_options_;

  TF_DISALLOW_COPY_AND_ASSIGN(Lz4InputStream);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_INPUTSTREAM_H_
