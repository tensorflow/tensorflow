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

#include "tensorflow/core/lib/io/record_reader.h"

#include <limits.h>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

RecordReaderOptions RecordReaderOptions::CreateRecordReaderOptions(
    const string& compression_type) {
  RecordReaderOptions options;
  if (compression_type == "ZLIB") {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
#endif  // IS_SLIM_BUILD
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
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

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : options_(options),
      input_stream_(new RandomAccessInputStream(file)),
      last_read_failed_(false) {
  if (options.buffer_size > 0) {
    input_stream_.reset(new BufferedInputStream(input_stream_.release(),
                                                options.buffer_size, true));
  }
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    input_stream_.reset(new ZlibInputStream(
        input_stream_.release(), options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options, true));
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unrecognized compression type :" << options.compression_type;
  }
}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
//
// offset corresponds to the user-provided value to ReadRecord()
// and is used only in error messages.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n, string* result) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(expected, result));

  if (result->size() != expected) {
    if (result->empty()) {
      return errors::OutOfRange("eof");
    } else {
      return errors::DataLoss("truncated record at ", offset);
    }
  }

  const uint32 masked_crc = core::DecodeFixed32(result->data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(result->data(), n)) {
    return errors::DataLoss("corrupted record at ", offset);
  }
  result->resize(n);
  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, string* record) {
  static const size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static const size_t kFooterSize = sizeof(uint32);

  // Position the input stream.
  int64 curr_pos = input_stream_->Tell();
  int64 desired_pos = static_cast<int64>(*offset);
  if (curr_pos > desired_pos || curr_pos < 0 /* EOF */ ||
      (curr_pos == desired_pos && last_read_failed_)) {
    last_read_failed_ = false;
    TF_RETURN_IF_ERROR(input_stream_->Reset());
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos));
  } else if (curr_pos < desired_pos) {
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos - curr_pos));
  }
  DCHECK_EQ(desired_pos, input_stream_->Tell());

  // Read header data.
  Status s = ReadChecksummed(*offset, sizeof(uint64), record);
  if (!s.ok()) {
    last_read_failed_ = true;
    return s;
  }
  const uint64 length = core::DecodeFixed64(record->data());

  // Read data
  s = ReadChecksummed(*offset + kHeaderSize, length, record);
  if (!s.ok()) {
    last_read_failed_ = true;
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record at ", *offset);
    }
    return s;
  }

  *offset += kHeaderSize + length + kFooterSize;
  DCHECK_EQ(*offset, input_stream_->Tell());
  return Status::OK();
}

SequentialRecordReader::SequentialRecordReader(
    RandomAccessFile* file, const RecordReaderOptions& options)
    : underlying_(file, options), offset_(0) {}

}  // namespace io
}  // namespace tensorflow
