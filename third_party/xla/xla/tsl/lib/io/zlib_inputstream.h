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

#ifndef XLA_TSL_LIB_IO_ZLIB_INPUTSTREAM_H_
#define XLA_TSL_LIB_IO_ZLIB_INPUTSTREAM_H_

#include <string>

#include "xla/tsl/lib/io/inputstream_interface.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace io {

// Forward declare some members of zlib.h, which is only included in the
// .cc file.
struct ZStreamDef;

// An ZlibInputStream provides support for reading from a stream compressed
// using zlib (http://www.zlib.net/). Buffers the contents of the file.
//
// A given instance of an ZlibInputStream is NOT safe for concurrent use
// by multiple threads
class ZlibInputStream : public InputStreamInterface {
 public:
  // Create a ZlibInputStream for `input_stream` with a buffer of size
  // `input_buffer_bytes` bytes for reading contents from `input_stream` and
  // another buffer with size `output_buffer_bytes` for caching decompressed
  // contents.
  //
  // Takes ownership of `input_stream` iff `owns_input_stream` is true.
  ZlibInputStream(InputStreamInterface* input_stream, size_t input_buffer_bytes,
                  size_t output_buffer_bytes,
                  const ZlibCompressionOptions& zlib_options,
                  bool owns_input_stream);

  // Equivalent to the previous constructor with owns_input_stream=false.
  ZlibInputStream(InputStreamInterface* input_stream, size_t input_buffer_bytes,
                  size_t output_buffer_bytes,
                  const ZlibCompressionOptions& zlib_options);

  ~ZlibInputStream() override;

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // Return Status codes:
  // OK:           If successful.
  // OUT_OF_RANGE: If there are not enough bytes to read before
  //               the end of the stream.
  // ABORTED:      If inflate() fails, we return the error code with the
  //               error message in `z_stream_->msg`.
  // others:       If reading from stream failed.
  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override;

#if defined(TF_CORD_SUPPORT)
  absl::Status ReadNBytes(int64_t bytes_to_read, absl::Cord* result) override;
#endif

  int64_t Tell() const override;

  absl::Status Reset() override;

 private:
  void InitZlibBuffer();

  const bool owns_input_stream_;
  InputStreamInterface* input_stream_;
  size_t input_buffer_capacity_;   // Size of z_stream_input_
  size_t output_buffer_capacity_;  // Size of z_stream_output_
  char* next_unread_byte_;         // Next unread byte in z_stream_output_
  bool init_error_ = false;        // Whether we encountered an error in init.

  ZlibCompressionOptions const zlib_options_;

  std::unique_ptr<ZStreamDef> z_stream_def_;

  // Reads data from `input_stream_` and tries to fill up `z_stream_input_` if
  // enough unread data is left in `input_stream_`.
  //
  // Looks up z_stream_->next_in to check how much data in z_stream_input_
  // has already been read. The used data is removed and new data is added to
  // after any unread data in z_stream_input_.
  // After this call z_stream_->next_in points to the start of z_stream_input_
  // and z_stream_->avail_in stores the number of readable bytes in
  // z_stream_input_.
  //
  // Returns OutOfRange error if NO data could be read from stream. Note that
  // this won't return an OutOfRange if there wasn't sufficient data in stream
  // to completely fill up z_stream_input_.
  absl::Status ReadFromStream();

  // Calls `inflate()` and returns DataLoss Status if it failed.
  absl::Status Inflate();

  // Starts reading bytes at `next_unread_byte_` till either `bytes_to_read`
  // bytes have been read or `z_stream_->next_out` is reached.
  // Returns the number of bytes read and advances the `next_unread_byte_`
  // pointer to the next location to read from.
  size_t ReadBytesFromCache(size_t bytes_to_read, tstring* result);

  // The number of unread bytes in z_stream_output_.
  //
  // z_stream_output_  -->
  //
  // [RRRRRRRRRRRRRRRRRRUUUUUUUUUUUUUU000000000000000000]
  //                    ^             ^
  //           next_unread_byte_    z_stream_->next_out
  //
  // R: Read bytes
  // U: Unread bytes
  // 0: garbage bytes where new output will be written
  //
  // Returns the size of [next_unread_byte_, z_stream_->next_out)
  size_t NumUnreadBytes() const;

  // Number of *uncompressed* bytes that have been read from this stream.
  int64_t bytes_read_;

  ZlibInputStream(const ZlibInputStream&) = delete;
  void operator=(const ZlibInputStream&) = delete;
};

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_ZLIB_INPUTSTREAM_H_
