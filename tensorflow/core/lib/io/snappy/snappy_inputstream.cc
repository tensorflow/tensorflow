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

#include "tensorflow/core/lib/io/snappy/snappy_inputstream.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace io {

SnappyInputStream::SnappyInputStream(InputStreamInterface* input_stream,
                                     size_t output_buffer_bytes,
                                     bool owns_input_stream)
    : input_stream_(input_stream),
      output_buffer_bytes_(output_buffer_bytes),
      owns_input_stream_(owns_input_stream),
      bytes_read_(0),
      output_buffer_(new char[output_buffer_bytes]),
      next_out_(nullptr),
      avail_out_(0) {}

SnappyInputStream::SnappyInputStream(InputStreamInterface* input_stream,
                                     size_t output_buffer_bytes)
    : SnappyInputStream(input_stream, output_buffer_bytes, false) {}

SnappyInputStream::~SnappyInputStream() {
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

Status SnappyInputStream::ReadNBytes(int64 bytes_to_read, tstring* result) {
  result->clear();
  result->resize_uninitialized(bytes_to_read);

  char* result_ptr = result->mdata();

  // Read as many bytes as possible from the cache.
  size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
  bytes_to_read -= bytes_read;
  result_ptr += bytes_read;

  while (bytes_to_read > 0) {
    DCHECK_EQ(avail_out_, 0);

    // Fill the cache with more data.
    TF_RETURN_IF_ERROR(Inflate());

    size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
    bytes_to_read -= bytes_read;
    result_ptr += bytes_read;
  }

  return Status::OK();
}

#if defined(TF_CORD_SUPPORT)
Status SnappyInputStream::ReadNBytes(int64 bytes_to_read, absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return Status::OK();
}
#endif

Status SnappyInputStream::Inflate() {
  tstring compressed_block_length_ts;
  uint32 compressed_block_length;

  TF_RETURN_IF_ERROR(
      input_stream_->ReadNBytes(sizeof(uint32), &compressed_block_length_ts));
  for (int i = 0; i < sizeof(uint32); ++i) {
    compressed_block_length =
        (compressed_block_length << 8) |
        static_cast<unsigned char>(compressed_block_length_ts.data()[i]);
  }

  tstring compressed_block;
  compressed_block.resize_uninitialized(compressed_block_length);

  Status s =
      input_stream_->ReadNBytes(compressed_block_length, &compressed_block);
  if (errors::IsOutOfRange(s)) {
    return errors::DataLoss("Failed to read ", compressed_block_length,
                            " bytes from file. Possible data corruption.");
  }
  TF_RETURN_IF_ERROR(s);

  size_t uncompressed_length;
  if (!port::Snappy_GetUncompressedLength(compressed_block.data(),
                                          compressed_block_length,
                                          &uncompressed_length)) {
    return errors::DataLoss("Parsing error in Snappy_GetUncompressedLength");
  }

  DCHECK_EQ(avail_out_, 0);
  if (output_buffer_bytes_ < uncompressed_length) {
    return errors::ResourceExhausted(
        "Output buffer(size: ", output_buffer_bytes_,
        " bytes"
        ") too small. Should be larger than ",
        uncompressed_length, " bytes.");
  }

  next_out_ = output_buffer_.get();
  if (!port::Snappy_Uncompress(compressed_block.data(), compressed_block_length,
                               output_buffer_.get())) {
    return errors::DataLoss("Snappy_Uncompress failed.");
  }
  avail_out_ += uncompressed_length;

  return Status::OK();
}

size_t SnappyInputStream::ReadBytesFromCache(size_t bytes_to_read,
                                             char* result) {
  size_t can_read_bytes = std::min(bytes_to_read, avail_out_);
  if (can_read_bytes) {
    memcpy(result, next_out_, can_read_bytes);
    next_out_ += can_read_bytes;
    avail_out_ -= can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

int64 SnappyInputStream::Tell() const { return bytes_read_; }

Status SnappyInputStream::Reset() {
  TF_RETURN_IF_ERROR(input_stream_->Reset());
  avail_out_ = 0;
  bytes_read_ = 0;
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
