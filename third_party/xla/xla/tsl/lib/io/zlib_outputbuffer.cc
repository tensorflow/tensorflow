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

#include "xla/tsl/lib/io/zlib_outputbuffer.h"

#include "xla/tsl/platform/errors.h"

namespace tsl {
namespace io {

ZlibOutputBuffer::ZlibOutputBuffer(
    WritableFile* file,
    int32_t input_buffer_bytes,  // size of z_stream.next_in buffer
    int32_t output_buffer_bytes,
    const ZlibCompressionOptions&
        zlib_options)  // size of z_stream.next_out buffer
    : file_(file),
      init_status_(),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      z_stream_input_(new Bytef[input_buffer_bytes]),
      z_stream_output_(new Bytef[output_buffer_bytes]),
      zlib_options_(zlib_options),
      z_stream_(new z_stream) {}

ZlibOutputBuffer::~ZlibOutputBuffer() {
  if (z_stream_) {
    LOG(WARNING) << "ZlibOutputBuffer::Close() not called. Possible data loss";
  }
}

absl::Status ZlibOutputBuffer::Init() {
  // Output buffer size should be greater than 1 because deflation needs at
  // least one byte for book keeping etc.
  if (output_buffer_capacity_ <= 1) {
    return errors::InvalidArgument(
        "output_buffer_bytes should be greater than "
        "1");
  }
  memset(z_stream_.get(), 0, sizeof(z_stream));
  z_stream_->zalloc = Z_NULL;
  z_stream_->zfree = Z_NULL;
  z_stream_->opaque = Z_NULL;
  int status =
      deflateInit2(z_stream_.get(), zlib_options_.compression_level,
                   zlib_options_.compression_method, zlib_options_.window_bits,
                   zlib_options_.mem_level, zlib_options_.compression_strategy);
  if (status != Z_OK) {
    z_stream_.reset(nullptr);
    return errors::InvalidArgument("deflateInit failed with status", status);
  }
  z_stream_->next_in = z_stream_input_.get();
  z_stream_->next_out = z_stream_output_.get();
  z_stream_->avail_in = 0;
  z_stream_->avail_out = output_buffer_capacity_;
  return absl::OkStatus();
}

int32 ZlibOutputBuffer::AvailableInputSpace() const {
  return input_buffer_capacity_ - z_stream_->avail_in;
}

void ZlibOutputBuffer::AddToInputBuffer(absl::string_view data) {
  size_t bytes_to_write = data.size();
  CHECK_LE(bytes_to_write, AvailableInputSpace());

  // Input stream ->
  // [....................input_buffer_capacity_...............]
  // [<...read_bytes...><...avail_in...>......empty space......]
  //  ^                 ^
  //  |                 |
  //  z_stream_input_   next_in
  //
  // Data in the input stream is sharded as show above. z_stream_->next_in could
  // be pointing to some byte in the buffer with avail_in number of bytes
  // available to be read.
  //
  // In order to avoid shifting the avail_in bytes at next_in to the head of
  // the buffer we try to fit `data` in the empty space at the tail of the
  // input stream.
  // TODO(srbs): This could be avoided if we had a circular buffer.
  // If it doesn't fit we free the space at the head of the stream and then
  // append `data` at the end of existing data.

  int32_t read_bytes = z_stream_->next_in - z_stream_input_.get();
  int32_t unread_bytes = z_stream_->avail_in;
  int32_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (static_cast<int32>(bytes_to_write) > free_tail_bytes) {
    memmove(z_stream_input_.get(), z_stream_->next_in, z_stream_->avail_in);
    z_stream_->next_in = z_stream_input_.get();
  }
  memcpy(z_stream_->next_in + z_stream_->avail_in, data.data(), bytes_to_write);
  z_stream_->avail_in += bytes_to_write;
}

absl::Status ZlibOutputBuffer::DeflateBuffered(int flush_mode) {
  do {
    // From zlib manual (http://www.zlib.net/manual.html):
    //
    // "In the case of a Z_FULL_FLUSH or Z_SYNC_FLUSH, make sure that
    // avail_out is greater than six to avoid repeated flush markers due
    // to avail_out == 0 on return."
    //
    // If above condition is met or if output buffer is full we flush contents
    // to file.
    if (z_stream_->avail_out == 0 ||
        (IsSyncOrFullFlush(flush_mode) && z_stream_->avail_out < 6)) {
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
    TF_RETURN_IF_ERROR(Deflate(flush_mode));
  } while (z_stream_->avail_out == 0);

  DCHECK(z_stream_->avail_in == 0);
  z_stream_->next_in = z_stream_input_.get();
  return absl::OkStatus();
}

absl::Status ZlibOutputBuffer::FlushOutputBufferToFile() {
  uint32 bytes_to_write = output_buffer_capacity_ - z_stream_->avail_out;
  if (bytes_to_write > 0) {
    absl::Status s = file_->Append(absl::string_view(
        reinterpret_cast<char*>(z_stream_output_.get()), bytes_to_write));
    if (s.ok()) {
      z_stream_->next_out = z_stream_output_.get();
      z_stream_->avail_out = output_buffer_capacity_;
    }
    return s;
  }
  return absl::OkStatus();
}

absl::Status ZlibOutputBuffer::Append(absl::string_view data) {
  // If there is sufficient free space in z_stream_input_ to fit data we
  // add it there and return.
  // If there isn't enough space we deflate the existing contents of
  // z_input_stream_. If data now fits in z_input_stream_ we add it there
  // else we directly deflate it.
  //
  // The deflated output is accumulated in z_stream_output_ and gets written to
  // file as and when needed.

  size_t bytes_to_write = data.size();

  if (bytes_to_write <= static_cast<size_t>(AvailableInputSpace())) {
    AddToInputBuffer(data);
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(DeflateBuffered(zlib_options_.flush_mode));

  // At this point input stream should be empty.
  if (bytes_to_write <= static_cast<size_t>(AvailableInputSpace())) {
    AddToInputBuffer(data);
    return absl::OkStatus();
  }

  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  z_stream_->next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
  z_stream_->avail_in = bytes_to_write;

  do {
    if (z_stream_->avail_out == 0) {
      // No available output space.
      // Write output buffer to file.
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
    TF_RETURN_IF_ERROR(Deflate(zlib_options_.flush_mode));
  } while (z_stream_->avail_out == 0);

  DCHECK(z_stream_->avail_in == 0);  // All input will be used up.

  // Restore z_stream input pointers.
  z_stream_->next_in = z_stream_input_.get();

  return absl::OkStatus();
}

#if defined(TF_CORD_SUPPORT)
absl::Status ZlibOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return absl::OkStatus();
}
#endif

absl::Status ZlibOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(DeflateBuffered(Z_PARTIAL_FLUSH));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return file_->Flush();
}

absl::Status ZlibOutputBuffer::Name(absl::string_view* result) const {
  return file_->Name(result);
}

absl::Status ZlibOutputBuffer::Sync() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

absl::Status ZlibOutputBuffer::Close() {
  if (z_stream_) {
    TF_RETURN_IF_ERROR(DeflateBuffered(Z_FINISH));
    TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    deflateEnd(z_stream_.get());
    z_stream_.reset(nullptr);
  }
  return absl::OkStatus();
}

absl::Status ZlibOutputBuffer::Deflate(int flush) {
  int error = deflate(z_stream_.get(), flush);
  if (error == Z_OK || error == Z_BUF_ERROR ||
      (error == Z_STREAM_END && flush == Z_FINISH)) {
    return absl::OkStatus();
  }
  string error_string = strings::StrCat("deflate() failed with error ", error);
  if (z_stream_->msg != nullptr) {
    strings::StrAppend(&error_string, ": ", z_stream_->msg);
  }
  return errors::DataLoss(error_string);
}

absl::Status ZlibOutputBuffer::Tell(int64_t* position) {
  return file_->Tell(position);
}

}  // namespace io
}  // namespace tsl
