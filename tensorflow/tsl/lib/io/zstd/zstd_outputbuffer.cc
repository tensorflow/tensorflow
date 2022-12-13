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

#include "tensorflow/tsl/lib/io/zstd/zstd_outputbuffer.h"

namespace tsl {
namespace io {

struct ZstdStreamDef {
  ZstdStreamDef(const size_t input_buffer_bytes,
                const size_t output_buffer_bytes)
      : input_buffer_(new char[input_buffer_bytes]),
        output_buffer_(new char[output_buffer_bytes]),
        input_({input_buffer_.get(), 0, 0}),
        output_({output_buffer_.get(), output_buffer_bytes, 0}) {}
  ZSTD_inBuffer input_;
  ZSTD_outBuffer output_;

  char* next_in_;
  char* next_out_;

  std::unique_ptr<char[]> input_buffer_;
  std::unique_ptr<char[]> output_buffer_;
};

ZstdOutputBuffer::ZstdOutputBuffer(WritableFile* file, int32 input_buffer_bytes,
                                   int32 output_buffer_bytes,
                                   const ZstdCompressionOptions& zstd_options)
    : file_(file),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      zstd_stream_(new ZstdStreamDef(input_buffer_bytes, output_buffer_bytes)),
      zstd_options_(zstd_options) {
  InitZstdBuffer();
}

ZstdOutputBuffer::~ZstdOutputBuffer() {
  size_t bytes_to_write = 0;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
}

void ZstdOutputBuffer::InitZstdBuffer() {
  context_ = ZSTD_createCCtx();
  if (context_ == nullptr) {
    LOG(FATAL) << "Creation of context failed.";
  }
  ZSTD_CCtx_setParameter(context_, ZSTD_c_compressionLevel,
                         zstd_options_.compression_level);
  ZSTD_CCtx_setParameter(context_, ZSTD_c_strategy,
                         zstd_options_.compression_strategy);
  ZSTD_CCtx_setParameter(context_, ZSTD_c_checksumFlag, 1);
  ZSTD_CCtx_setParameter(context_, ZSTD_c_nbWorkers, zstd_options_.nb_workers);

  zstd_stream_->next_in_ = zstd_stream_->input_buffer_.get();
  zstd_stream_->next_out_ = zstd_stream_->output_buffer_.get();
}

Status ZstdOutputBuffer::Append(StringPiece data) {
  if (output_buffer_capacity_ == 0) {
    return errors::InvalidArgument(
        "Can't compress data with output_buffer_bytes = 0");
  }

  // The deflated output is accumulated in output_buffer_ and gets written to
  // file as and when needed.
  size_t bytes_to_write = data.size();

  // If there is sufficient free space in input_buffer_ to fit data we
  // add it there and return.

  if (bytes_to_write <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return OkStatus();
  }

  // If there isn't enough available space in the input_buffer_ we empty it
  // by compressing its contents. If data now fits in input_buffer_
  // we add it there else we directly deflate it.
  TF_RETURN_IF_ERROR(Deflate(zstd_options_.flush_mode));

  // At this point input stream should be empty.
  if (bytes_to_write <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return OkStatus();
  }

  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  zstd_stream_->next_in_ = const_cast<char*>(data.data());
  zstd_stream_->input_.size = bytes_to_write;

  TF_RETURN_IF_ERROR(Deflate(zstd_options_.flush_mode));

  DCHECK_EQ(zstd_stream_->input_.size, 0);  // All input used up.

  zstd_stream_->next_in_ = zstd_stream_->input_buffer_.get();

  return OkStatus();
}

void ZstdOutputBuffer::AddToInputBuffer(StringPiece data) {
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

  const size_t read_bytes =
      zstd_stream_->next_in_ - zstd_stream_->input_buffer_.get();
  const size_t unread_bytes = zstd_stream_->input_.size;
  const size_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (bytes_to_write > free_tail_bytes) {
    memmove(zstd_stream_->input_buffer_.get(), zstd_stream_->next_in_,
            zstd_stream_->input_.size);
    zstd_stream_->next_in_ = zstd_stream_->input_buffer_.get();
  }
  memcpy(zstd_stream_->next_in_ + zstd_stream_->input_.size,
         reinterpret_cast<const void*>(data.data()), bytes_to_write);
  zstd_stream_->input_.size += bytes_to_write;
}

#if defined(TF_CORD_SUPPORT)
Status ZstdOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return OkStatus();
}
#endif

Status ZstdOutputBuffer::Close() {
  // Given that we do not own `file`, we don't close it.
  TF_RETURN_IF_ERROR(Deflate(ZSTD_e_end));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  ZSTD_freeCCtx(context_);
  return OkStatus();
}

Status ZstdOutputBuffer::Name(StringPiece* result) const {
  return file_->Name(result);
}

Status ZstdOutputBuffer::Sync() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status ZstdOutputBuffer::Tell(int64* position) { return file_->Tell(position); }

Status ZstdOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(Deflate(ZSTD_e_flush));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return file_->Flush();
}

int32 ZstdOutputBuffer::AvailableInputSpace() const {
  return input_buffer_capacity_ - zstd_stream_->input_.size;
}

Status ZstdOutputBuffer::FlushOutputBufferToFile() {
  size_t bytes_to_write = output_buffer_capacity_ - zstd_stream_->output_.size;
  if (bytes_to_write > 0) {
    Status s = file_->Append(
        StringPiece(reinterpret_cast<char*>(zstd_stream_->output_buffer_.get()),
                    bytes_to_write));
    if (s.ok()) {
      zstd_stream_->next_out_ = zstd_stream_->output_buffer_.get();
      zstd_stream_->output_.size = output_buffer_capacity_;
    }
    return s;
  }
  return OkStatus();
}

Status ZstdOutputBuffer::DeflateBuffered(ZSTD_EndDirective end_directive) {}

Status ZstdOutputBuffer::Deflate(ZSTD_EndDirective end_directive) {
  zstd_stream_->input_.src = zstd_stream_->next_in_;
  zstd_stream_->input_.pos = 0;

  bool finished;
  do {
    zstd_stream_->output_.dst = zstd_stream_->next_out_;
    zstd_stream_->output_.pos = 0;

    const size_t remaining = ZSTD_compressStream2(
        context_, &zstd_stream_->output_, &zstd_stream_->input_, end_directive);
    if (ZSTD_isError(remaining)) {
      return errors::DataLoss(ZSTD_getErrorName(remaining));
    }

    zstd_stream_->output_.size =
        output_buffer_capacity_ - zstd_stream_->output_.pos;

    TF_RETURN_IF_ERROR(FlushOutputBufferToFile());

    if (end_directive == ZSTD_e_end || end_directive == ZSTD_e_flush) {
      finished = remaining == 0;
    } else {
      finished = zstd_stream_->input_.pos == zstd_stream_->input_.size;
    }
  } while (!finished);

  zstd_stream_->input_.size = 0;
  return OkStatus();
}

}  // namespace io
}  // namespace tsl
