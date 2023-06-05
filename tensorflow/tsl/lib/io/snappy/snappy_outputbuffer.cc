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

#include "tensorflow/tsl/lib/io/snappy/snappy_outputbuffer.h"

#include <algorithm>

namespace tsl {
namespace io {

SnappyOutputBuffer::SnappyOutputBuffer(WritableFile* file,
                                       int32_t input_buffer_bytes,
                                       int32_t output_buffer_bytes)
    : file_(file),
      input_buffer_(new char[input_buffer_bytes]),
      input_buffer_capacity_(input_buffer_bytes),
      next_in_(input_buffer_.get()),
      output_buffer_(new char[output_buffer_bytes]),
      output_buffer_capacity_(output_buffer_bytes),
      next_out_(output_buffer_.get()),
      avail_out_(output_buffer_bytes) {}

SnappyOutputBuffer::~SnappyOutputBuffer() {
  size_t bytes_to_write = output_buffer_capacity_ - avail_out_;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
}

Status SnappyOutputBuffer::Append(StringPiece data) { return Write(data); }

#if defined(TF_CORD_SUPPORT)
Status SnappyOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return OkStatus();
}
#endif

Status SnappyOutputBuffer::Close() {
  // Given that we do not own `file`, we don't close it.
  return Flush();
}

Status SnappyOutputBuffer::Name(StringPiece* result) const {
  return file_->Name(result);
}

Status SnappyOutputBuffer::Sync() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status SnappyOutputBuffer::Tell(int64_t* position) {
  return file_->Tell(position);
}

Status SnappyOutputBuffer::Write(StringPiece data) {
  //
  // The deflated output is accumulated in output_buffer_ and gets written to
  // file as and when needed.

  size_t bytes_to_write = data.size();

  // If there is sufficient free space in input_buffer_ to fit data we
  // add it there and return.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return OkStatus();
  }

  // If there isn't enough available space in the input_buffer_ we empty it
  // by uncompressing its contents. If data now fits in input_buffer_
  // we add it there else we directly deflate it.
  TF_RETURN_IF_ERROR(DeflateBuffered());

  // input_buffer_ should be empty at this point.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return OkStatus();
  }

  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  next_in_ = const_cast<char*>(data.data());
  avail_in_ = bytes_to_write;

  TF_RETURN_IF_ERROR(Deflate());

  DCHECK_EQ(avail_in_, 0);  // All input will be used up.

  next_in_ = input_buffer_.get();

  return OkStatus();
}

Status SnappyOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(DeflateBuffered());
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return OkStatus();
}

int32 SnappyOutputBuffer::AvailableInputSpace() const {
  return input_buffer_capacity_ - avail_in_;
}

void SnappyOutputBuffer::AddToInputBuffer(StringPiece data) {
  size_t bytes_to_write = data.size();
  DCHECK_LE(bytes_to_write, AvailableInputSpace());

  // Input stream ->
  // [....................input_buffer_capacity_...............]
  // [<...read_bytes...><...avail_in...>......empty space......]
  //  ^                 ^
  //  |                 |
  //  input_buffer_   next_in
  //
  // Data in the input stream is sharded as shown above. next_in_ could
  // be pointing to some byte in the buffer with avail_in number of bytes
  // available to be read.
  //
  // In order to avoid shifting the avail_in bytes at next_in to the head of
  // the buffer we try to fit `data` in the empty space at the tail of the
  // input stream.
  // TODO(srbs): This could be avoided if we had a circular buffer.
  // If it doesn't fit we free the space at the head of the stream and then
  // append `data` at the end of existing data.

  const int32_t read_bytes = next_in_ - input_buffer_.get();
  const int32_t unread_bytes = avail_in_;
  const int32_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (static_cast<int32>(bytes_to_write) > free_tail_bytes) {
    memmove(input_buffer_.get(), next_in_, avail_in_);
    next_in_ = input_buffer_.get();
  }
  memcpy(next_in_ + avail_in_, data.data(), bytes_to_write);
  avail_in_ += bytes_to_write;
}

Status SnappyOutputBuffer::AddToOutputBuffer(const char* data, size_t length) {
  while (length > 0) {
    size_t bytes_to_copy = std::min(length, avail_out_);
    memcpy(next_out_, data, bytes_to_copy);
    data += bytes_to_copy;
    next_out_ += bytes_to_copy;
    avail_out_ -= bytes_to_copy;
    length -= bytes_to_copy;
    if (avail_out_ == 0) {
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
  }
  return OkStatus();
}

Status SnappyOutputBuffer::DeflateBuffered() {
  TF_RETURN_IF_ERROR(Deflate());
  DCHECK_EQ(avail_in_, 0);
  next_in_ = input_buffer_.get();
  return OkStatus();
}

Status SnappyOutputBuffer::FlushOutputBufferToFile() {
  size_t bytes_to_write = output_buffer_capacity_ - avail_out_;
  if (bytes_to_write > 0) {
    Status s = file_->Append(StringPiece(
        reinterpret_cast<char*>(output_buffer_.get()), bytes_to_write));
    if (s.ok()) {
      next_out_ = output_buffer_.get();
      avail_out_ = output_buffer_capacity_;
    }
    return s;
  }
  return OkStatus();
}

Status SnappyOutputBuffer::Deflate() {
  if (avail_in_ == 0) {
    return OkStatus();
  }
  string output;
  if (!port::Snappy_Compress(next_in_, avail_in_, &output)) {
    return errors::DataLoss("Snappy_Compress failed");
  }

  // Write length of compressed block to output buffer.
  char compressed_length_array[4];
  std::fill(compressed_length_array, compressed_length_array + 4, 0);
  for (int i = 0; i < 4; i++) {
    // Little endian.
    compressed_length_array[i] = output.size() >> (8 * (3 - i));
  }
  TF_RETURN_IF_ERROR(AddToOutputBuffer(compressed_length_array, 4));

  // Write compressed output to buffer.
  TF_RETURN_IF_ERROR(AddToOutputBuffer(output.data(), output.size()));
  next_in_ += avail_in_;
  avail_in_ = 0;

  return OkStatus();
}

}  // namespace io
}  // namespace tsl
