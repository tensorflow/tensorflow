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

#include <zlib.h>

#include "tensorflow/core/lib/io/zlib_inputstream.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace io {

struct ZStreamDef {
  ZStreamDef(size_t input_buffer_capacity, size_t output_buffer_capacity)
      : input(new Bytef[input_buffer_capacity]),
        output(new Bytef[output_buffer_capacity]),
        stream(new z_stream) {}

  // Buffer for storing contents read from compressed stream.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<Bytef[]> input;

  // Buffer for storing inflated contents of `input_stream_`.
  std::unique_ptr<Bytef[]> output;

  // Configuration passed to `inflate`.
  //
  // z_stream_def_->stream->next_in:
  //   Next byte to de-compress. Points to some byte in
  //   z_stream_def_->streamdef_.input buffer.
  // z_stream_def_->stream->avail_in:
  //   Number of bytes available to be decompressed at this time.
  // z_stream_def_->stream->next_out:
  //   Next byte to write de-compressed data to. Points to some byte in
  //   z_stream_def_->streamdef_.output buffer.
  // z_stream_def_->stream->avail_out:
  //   Number of free bytes available at write location.
  std::unique_ptr<z_stream> stream;
};

ZlibInputStream::ZlibInputStream(
    InputStreamInterface* input_stream,
    size_t input_buffer_bytes,   // size of z_stream.next_in buffer
    size_t output_buffer_bytes,  // size of z_stream.next_out buffer
    const ZlibCompressionOptions& zlib_options, bool owns_input_stream)
    : owns_input_stream_(owns_input_stream),
      input_stream_(input_stream),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      zlib_options_(zlib_options),
      z_stream_def_(
          new ZStreamDef(input_buffer_capacity_, output_buffer_capacity_)),
      bytes_read_(0) {
  InitZlibBuffer();
}

ZlibInputStream::ZlibInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZlibCompressionOptions& zlib_options)
    : ZlibInputStream(input_stream, input_buffer_bytes, output_buffer_bytes,
                      zlib_options, false) {}

ZlibInputStream::~ZlibInputStream() {
  if (z_stream_def_->stream) {
    inflateEnd(z_stream_def_->stream.get());
  }
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

Status ZlibInputStream::Reset() {
  TF_RETURN_IF_ERROR(input_stream_->Reset());
  inflateEnd(z_stream_def_->stream.get());
  InitZlibBuffer();
  bytes_read_ = 0;
  return Status::OK();
}

void ZlibInputStream::InitZlibBuffer() {
  memset(z_stream_def_->stream.get(), 0, sizeof(z_stream));

  z_stream_def_->stream->zalloc = Z_NULL;
  z_stream_def_->stream->zfree = Z_NULL;
  z_stream_def_->stream->opaque = Z_NULL;
  z_stream_def_->stream->next_in = Z_NULL;
  z_stream_def_->stream->avail_in = 0;

  int status =
      inflateInit2(z_stream_def_->stream.get(), zlib_options_.window_bits);

  CHECK_EQ(status, Z_OK) << "inflateInit failed with status " << status;

  z_stream_def_->stream->next_in = z_stream_def_->input.get();
  z_stream_def_->stream->next_out = z_stream_def_->output.get();
  next_unread_byte_ = reinterpret_cast<char*>(z_stream_def_->output.get());
  z_stream_def_->stream->avail_in = 0;
  z_stream_def_->stream->avail_out = output_buffer_capacity_;
}

Status ZlibInputStream::ReadFromStream() {
  int bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(z_stream_def_->input.get());

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  if (z_stream_def_->stream->avail_in > 0) {
    uLong read_bytes =
        z_stream_def_->stream->next_in - z_stream_def_->input.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(z_stream_def_->input.get(), z_stream_def_->stream->next_in,
              z_stream_def_->stream->avail_in);
    }

    bytes_to_read -= z_stream_def_->stream->avail_in;
    read_location += z_stream_def_->stream->avail_in;
  }
  string data;
  // Try to read enough data to fill up z_stream_def_->input.
  // TODO(rohanj): Add a char* version of ReadNBytes to InputStreamInterface
  // and use that instead to make this more efficient.
  Status s = input_stream_->ReadNBytes(bytes_to_read, &data);
  memcpy(read_location, data.data(), data.size());

  // Since we moved unread data to the head of the input stream we can point
  // next_in to the head of the input stream.
  z_stream_def_->stream->next_in = z_stream_def_->input.get();

  // Note: data.size() could be different from bytes_to_read.
  z_stream_def_->stream->avail_in += data.size();

  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  // We throw OutOfRange error iff no new data has been read from stream.
  // Since we never check how much data is remaining in the stream, it is
  // possible that on the last read there isn't enough data in the stream to
  // fill up the buffer in which case input_stream_->ReadNBytes would return an
  // OutOfRange error.
  if (data.empty()) {
    return errors::OutOfRange("EOF reached");
  }
  if (errors::IsOutOfRange(s)) {
    return Status::OK();
  }

  return s;
}

size_t ZlibInputStream::ReadBytesFromCache(size_t bytes_to_read,
                                           string* result) {
  size_t unread_bytes =
      reinterpret_cast<char*>(z_stream_def_->stream->next_out) -
      next_unread_byte_;
  size_t can_read_bytes = std::min(bytes_to_read, unread_bytes);
  if (can_read_bytes > 0) {
    result->append(next_unread_byte_, can_read_bytes);
    next_unread_byte_ += can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

size_t ZlibInputStream::NumUnreadBytes() const {
  size_t read_bytes =
      next_unread_byte_ - reinterpret_cast<char*>(z_stream_def_->output.get());
  return output_buffer_capacity_ - z_stream_def_->stream->avail_out -
         read_bytes;
}

Status ZlibInputStream::ReadNBytes(int64 bytes_to_read, string* result) {
  result->clear();
  // Read as many bytes as possible from cache.
  bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);

  while (bytes_to_read > 0) {
    // At this point we can be sure that cache has been emptied.
    DCHECK_EQ(NumUnreadBytes(), 0);

    // Now that the cache is empty we need to inflate more data.

    // Step 1. Fill up input buffer.
    // We read from stream only after the previously read contents have been
    // completely consumed. This is an optimization and can be removed if
    // it causes problems. `ReadFromStream` is capable of handling partially
    // filled up buffers.
    if (z_stream_def_->stream->avail_in == 0) {
      TF_RETURN_IF_ERROR(ReadFromStream());
    }

    // Step 2. Setup output stream.
    z_stream_def_->stream->next_out = z_stream_def_->output.get();
    next_unread_byte_ = reinterpret_cast<char*>(z_stream_def_->output.get());
    z_stream_def_->stream->avail_out = output_buffer_capacity_;

    // Step 3. Inflate Inflate Inflate!
    TF_RETURN_IF_ERROR(Inflate());

    bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);
  }

  return Status::OK();
}

int64 ZlibInputStream::Tell() const { return bytes_read_; }

Status ZlibInputStream::Inflate() {
  int error = inflate(z_stream_def_->stream.get(), zlib_options_.flush_mode);
  if (error != Z_OK && error != Z_STREAM_END) {
    string error_string =
        strings::StrCat("inflate() failed with error ", error);
    if (z_stream_def_->stream->msg != nullptr) {
      strings::StrAppend(&error_string, ": ", z_stream_def_->stream->msg);
    }
    return errors::DataLoss(error_string);
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
