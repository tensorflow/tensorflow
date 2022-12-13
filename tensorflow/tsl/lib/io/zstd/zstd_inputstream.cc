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

#include "tensorflow/tsl/lib/io/zstd/zstd_inputstream.h"

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/strcat.h"

namespace tsl {
namespace io {

ZstdInputStream::ZstdInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZstdCompressionOptions& zstd_options,
                                 bool owns_input_stream)
    : owns_input_stream_(owns_input_stream),
      input_stream_(input_stream),
      input_buffer_(new char[input_buffer_bytes]),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_(new char[output_buffer_bytes]),
      output_buffer_capacity_(output_buffer_bytes),
      bytes_read_(0),
      zstd_options_(zstd_options) {
  InitZstdBuffer();
}

ZstdInputStream::ZstdInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZstdCompressionOptions& zstd_options)
    : ZstdInputStream(input_stream, input_buffer_bytes, output_buffer_bytes,
                      zstd_options, false) {}

ZstdInputStream::~ZstdInputStream() {
  ZSTD_freeDCtx(context_);
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

void ZstdInputStream::InitZstdBuffer() {
  context_ = ZSTD_createDCtx();
  if (context_ == nullptr) {
    LOG(FATAL) << "Creation of context failed.";
  }
  next_in_byte_ = input_buffer_.get();
  zstd_input_buffer_ = {next_in_byte_, 0, 0};
  next_unread_byte_ = output_buffer_.get();
  unread_bytes_ = 0;
  avail_in_ = 0;
}

Status ZstdInputStream::Reset() {
  TF_RETURN_IF_ERROR(input_stream_->Reset());
  ZSTD_DCtx_reset(context_, ZSTD_reset_session_only);
  InitZstdBuffer();
  bytes_read_ = 0;
  return OkStatus();
}

size_t ZstdInputStream::ReadBytesFromCache(size_t bytes_to_read,
                                           tstring* result) {
  size_t can_read_bytes = std::min(bytes_to_read, unread_bytes_);
  if (can_read_bytes > 0) {
    tstring cached_result;
    cached_result.append(next_unread_byte_, can_read_bytes);
    result->append(next_unread_byte_, can_read_bytes);
  }
  next_unread_byte_ += can_read_bytes;
  unread_bytes_ -= can_read_bytes;
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

Status ZstdInputStream::ReadNBytes(int64 bytes_to_read, tstring* result) {
  result->clear();
  bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);

  while (bytes_to_read > 0) {
    // No bytes should be left in the cache.
    CHECK_EQ(unread_bytes_, 0);

    // Now that the cache is empty we need to inflate more data.
    next_unread_byte_ = output_buffer_.get();

    TF_RETURN_IF_ERROR(Inflate());

    // If no progress was made by inflate, read more compressed data from the
    // input stream.
    if (unread_bytes_ == 0) {
      TF_RETURN_IF_ERROR(ReadFromStream());
      if (avail_in_ == 0) {
        bytes_to_read = 0;
      }
    } else {
      bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);
    }
  }

  return OkStatus();
}

#if defined(TF_CORD_SUPPORT)
Status ZstdInputStream::ReadNBytes(int64 bytes_to_read, absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return OkStatus();
}
#endif

Status ZstdInputStream::Inflate() {
  ZSTD_outBuffer output = {next_unread_byte_, output_buffer_capacity_, 0};
  last_return_ = ZSTD_decompressStream(context_, &output, &zstd_input_buffer_);

  if (ZSTD_isError(last_return_)) {
    string error_name = ZSTD_getErrorName(last_return_);
    string error_string =
        strings::StrCat("ZSTD_decompressStream: ", error_name);
    return errors::DataLoss(error_string);
  }

  avail_in_ = 0;
  unread_bytes_ = output.pos;

  return OkStatus();
}

Status ZstdInputStream::ReadFromStream() {
  size_t bytes_to_read = input_buffer_capacity_;
  char* read_location = input_buffer_.get();

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  if (avail_in_ > 0) {
    size_t read_bytes = next_in_byte_ - input_buffer_.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(input_buffer_.get(), next_in_byte_, avail_in_);
    }

    bytes_to_read -= avail_in_;
    read_location += avail_in_;
  }

  tstring data;
  Status s = input_stream_->ReadNBytes(bytes_to_read, &data);
  memcpy(read_location, data.data(), data.size());

  // Note: data.size() could be different from bytes_to_read.
  avail_in_ += data.size();
  zstd_input_buffer_.pos = 0;
  zstd_input_buffer_.size = data.size();

  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  if (errors::IsOutOfRange(s)) {
    return OkStatus();
  }

  return s;
}

int64 ZstdInputStream::Tell() const { return bytes_read_; }

}  // namespace io
}  // namespace tsl
