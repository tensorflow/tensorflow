/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/record_writer.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {
namespace {
bool IsZlibCompressed(RecordWriterOptions options) {
  return options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION;
}
}  // namespace

RecordWriterOptions RecordWriterOptions::CreateRecordWriterOptions(
    const string& compression_type) {
  RecordWriterOptions options;
  if (compression_type == compression::kZlib) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
#endif  // IS_SLIM_BUILD
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
#endif  // IS_SLIM_BUILD
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
  return options;
}

RecordWriter::RecordWriter(WritableFile* dest,
                           const RecordWriterOptions& options)
    : dest_(dest), options_(options) {
  if (IsZlibCompressed(options)) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    ZlibOutputBuffer* zlib_output_buffer = new ZlibOutputBuffer(
        dest, options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options);
    Status s = zlib_output_buffer->Init();
    if (!s.ok()) {
      LOG(FATAL) << "Failed to initialize Zlib inputbuffer. Error: "
                 << s.ToString();
    }
    dest_ = zlib_output_buffer;
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordWriterOptions::NONE) {
    // Nothing to do
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordWriter::~RecordWriter() {
  if (dest_ != nullptr) {
    Status s = Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

Status RecordWriter::WriteRecord(StringPiece data) {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data.data(), data.size());
  PopulateFooter(footer, data.data(), data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  TF_RETURN_IF_ERROR(dest_->Append(data));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}

#if defined(PLATFORM_GOOGLE)
Status RecordWriter::WriteRecord(const absl::Cord& data) {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data);
  PopulateFooter(footer, data);
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  TF_RETURN_IF_ERROR(dest_->Append(data));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}
#endif

Status RecordWriter::Close() {
  if (dest_ == nullptr) return Status::OK();
#if !defined(IS_SLIM_BUILD)
  if (IsZlibCompressed(options_)) {
    Status s = dest_->Close();
    delete dest_;
    dest_ = nullptr;
    return s;
  }
#endif  // IS_SLIM_BUILD
  return Status::OK();
}

Status RecordWriter::Flush() {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  return dest_->Flush();
}

}  // namespace io
}  // namespace tensorflow
