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

#include "tensorflow/core/lib/io/lz4/lz4_outputbuffer.h"

namespace tensorflow {
namespace io {

struct Lz4OutputFrameDef {
  Lz4OutputFrameDef(const size_t input_buffer_bytes,
                    const size_t output_buffer_bytes)
      : input_buffer_(new char[input_buffer_bytes]),
        output_buffer_(new char[output_buffer_bytes]) {
    LZ4F_createCompressionContext(&lz4f_cctx, LZ4F_VERSION);
  }

  char* next_in_;
  char* next_out_;

  std::unique_ptr<char[]> input_buffer_;
  std::unique_ptr<char[]> output_buffer_;

  LZ4F_compressionContext_t lz4f_cctx;
};

Lz4OutputBuffer::Lz4OutputBuffer(WritableFile* file, int32 input_buffer_bytes,
                                 int32 output_buffer_bytes,
                                 const Lz4CompressionOptions& lz4_options)
    : file_(file),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      lz4_options_(lz4_options),
      lz4_frame_(
          new Lz4OutputFrameDef(input_buffer_bytes, output_buffer_bytes)) {
  InitLz4Buffer();
}

Lz4OutputBuffer::~Lz4OutputBuffer() {
  size_t bytes_to_write = 0;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
  LZ4F_freeCompressionContext(lz4_frame_->lz4f_cctx);
}

void Lz4OutputBuffer::InitLz4Buffer() {}

Status Lz4OutputBuffer::Append(StringPiece data) { return Status::OK(); }

void Lz4OutputBuffer::AddToInputBuffer(StringPiece data) {}

#if defined(TF_CORD_SUPPORT)
Status Lz4OutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return Status::OK();
}
#endif

Status Lz4OutputBuffer::Close() {
  // Given that we do not own `file`, we don't close it.
  TF_RETURN_IF_ERROR(Deflate());
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return Status::OK();
}

Status Lz4OutputBuffer::Name(StringPiece* result) const {
  return file_->Name(result);
}

Status Lz4OutputBuffer::Sync() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status Lz4OutputBuffer::Tell(int64* position) { return file_->Tell(position); }

Status Lz4OutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(Deflate());
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return file_->Flush();
}

int32 Lz4OutputBuffer::AvailableInputSpace() const { return 0; }

Status Lz4OutputBuffer::FlushOutputBufferToFile() { return Status::OK(); }

Status Lz4OutputBuffer::DeflateBuffered() {}

Status Lz4OutputBuffer::Deflate() { return Status::OK(); }

}  // namespace io
}  // namespace tensorflow
