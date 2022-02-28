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

#include "tensorflow/core/lib/io/lz4/lz4_inputstream.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace io {

Lz4InputStream::Lz4InputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const Lz4CompressionOptions& lz4_options,
                                 bool owns_input_stream)
    : owns_input_stream_(owns_input_stream),
      input_stream_(input_stream),
      input_buffer_(new char[input_buffer_bytes]),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_(new char[output_buffer_bytes]),
      output_buffer_capacity_(output_buffer_bytes),
      bytes_read_(0),
      lz4_options_(lz4_options) {
  InitLz4Buffer();
}

Lz4InputStream::Lz4InputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const Lz4CompressionOptions& lz4_options)
    : Lz4InputStream(input_stream, input_buffer_bytes, output_buffer_bytes,
                      lz4_options, false) {}

Lz4InputStream::~Lz4InputStream() {
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

void Lz4InputStream::InitLz4Buffer() {
}

Status Lz4InputStream::Reset() {
  return Status::OK();
}

size_t Lz4InputStream::ReadBytesFromCache(size_t bytes_to_read,
                                           tstring* result) {
  return 0;
}

Status Lz4InputStream::ReadNBytes(int64 bytes_to_read, tstring* result) {
  return Status::OK();
}

#if defined(TF_CORD_SUPPORT)
Status Lz4InputStream::ReadNBytes(int64 bytes_to_read, absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return Status::OK();
}
#endif

Status Lz4InputStream::Inflate() {
  return Status::OK();
}

Status Lz4InputStream::ReadFromStream() {
  return Status::OK();
}

int64 Lz4InputStream::Tell() const { return bytes_read_; }

}  // namespace io
}  // namespace tensorflow
