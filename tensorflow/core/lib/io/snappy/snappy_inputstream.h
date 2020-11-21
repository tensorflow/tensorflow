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

#ifndef TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_INPUTSTREAM_H_
#define TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_INPUTSTREAM_H_

#include "tensorflow/core/lib/io/inputstream_interface.h"

namespace tensorflow {
namespace io {

class SnappyInputStream : public InputStreamInterface {
 public:
  // Creates a SnappyInputStream for `input_stream`.
  //
  // Takes ownership  of `input_stream` iff `owns_input_stream` is true.
  SnappyInputStream(InputStreamInterface* input_stream,
                    size_t output_buffer_bytes, bool owns_input_stream);

  // Equivalent to the previous constructor with owns_input_stream = false.
  explicit SnappyInputStream(InputStreamInterface* input_stream,
                             size_t output_buffer_bytes);

  ~SnappyInputStream() override;

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

  // Attempt to read `bytes_to_read` from the decompressed data cache. Returns
  // the actual number of bytes read.
  size_t ReadBytesFromCache(size_t bytes_to_read, char* result);

  InputStreamInterface* input_stream_;
  const size_t output_buffer_bytes_;
  const bool owns_input_stream_;

  // Specifies the number of decompressed bytes currently read.
  int64 bytes_read_;

  // output_buffer_ contains decompressed data not yet read by the client.
  std::unique_ptr<char[]> output_buffer_;

  // next_out_ points to the position in the `output_buffer_` that contains the
  // next unread byte.
  char* next_out_;

  // avail_out_ specifies the number of bytes left in the output_buffers_ that
  // is not yet read.
  size_t avail_out_;

  TF_DISALLOW_COPY_AND_ASSIGN(SnappyInputStream);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_SNAPPY_SNAPPY_INPUTSTREAM_H_
