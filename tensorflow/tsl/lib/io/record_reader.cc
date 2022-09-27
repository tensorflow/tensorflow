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

#include "tensorflow/tsl/lib/io/record_reader.h"

#include <limits.h>

#include "tensorflow/tsl/lib/hash/crc32c.h"
#include "tensorflow/tsl/lib/io/buffered_inputstream.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/lib/io/random_inputstream.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/raw_coding.h"

namespace tsl {
namespace io {

RecordReaderOptions RecordReaderOptions::CreateRecordReaderOptions(
    const string& compression_type) {
  RecordReaderOptions options;

#if defined(IS_SLIM_BUILD)
  if (compression_type != compression::kNone) {
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
  }
#else
  if (compression_type == compression::kZlib) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
  } else if (compression_type == compression::kSnappy) {
    options.compression_type = io::RecordReaderOptions::SNAPPY_COMPRESSION;
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
#endif
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
#if defined(IS_SLIM_BUILD)
  if (options.compression_type != RecordReaderOptions::NONE) {
    LOG(FATAL) << "Compression is unsupported on mobile platforms.";
  }
#else
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
    input_stream_.reset(new ZlibInputStream(
        input_stream_.release(), options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options, true));
  } else if (options.compression_type ==
             RecordReaderOptions::SNAPPY_COMPRESSION) {
    input_stream_.reset(
        new SnappyInputStream(input_stream_.release(),
                              options.snappy_options.output_buffer_size, true));
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unrecognized compression type :" << options.compression_type;
  }
#endif
}

namespace {
inline const char* GetChecksumErrorSuffix(uint64 offset) {
  if (offset == 0) {
    return " (Is this even a TFRecord file?)";
  }
  return "";
}
}  // namespace

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
//
// offset corresponds to the user-provided value to ReadRecord()
// and is used only in error messages. For failures at offset 0,
// a reminder about the file format is added, because TFRecord files
// contain no explicit format marker.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n, tstring* result) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large",
                            GetChecksumErrorSuffix(offset));
  }

  const size_t expected = n + sizeof(uint32);
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(expected, result));

  if (result->size() != expected) {
    if (result->empty()) {
      return errors::OutOfRange("eof", GetChecksumErrorSuffix(offset));
    } else {
      return errors::DataLoss("truncated record at ", offset,
                              GetChecksumErrorSuffix(offset));
    }
  }

  const uint32 masked_crc = core::DecodeFixed32(result->data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(result->data(), n)) {
    return errors::DataLoss("corrupted record at ", offset,
                            GetChecksumErrorSuffix(offset));
  }
  result->resize(n);
  return OkStatus();
}

Status RecordReader::GetMetadata(Metadata* md) {
  if (!md) {
    return errors::InvalidArgument(
        "Metadata object call to GetMetadata() was null");
  }

  // Compute the metadata of the TFRecord file if not cached.
  if (!cached_metadata_) {
    TF_RETURN_IF_ERROR(input_stream_->Reset());

    int64_t data_size = 0;
    int64_t entries = 0;

    // Within the loop, we always increment offset positively, so this
    // loop should be guaranteed to either return after reaching EOF
    // or encountering an error.
    uint64 offset = 0;
    tstring record;
    while (true) {
      // Read header, containing size of data.
      Status s = ReadChecksummed(offset, sizeof(uint64), &record);
      if (!s.ok()) {
        if (errors::IsOutOfRange(s)) {
          // We should reach out of range when the record file is complete.
          break;
        }
        return s;
      }

      // Read the length of the data.
      const uint64 length = core::DecodeFixed64(record.data());

      // Skip reading the actual data since we just want the number
      // of records and the size of the data.
      TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(length + kFooterSize));
      offset += kHeaderSize + length + kFooterSize;

      // Increment running stats.
      data_size += length;
      ++entries;
    }

    cached_metadata_.reset(new Metadata());
    cached_metadata_->stats.entries = entries;
    cached_metadata_->stats.data_size = data_size;
    cached_metadata_->stats.file_size =
        data_size + (kHeaderSize + kFooterSize) * entries;
  }

  md->stats = cached_metadata_->stats;
  return OkStatus();
}

Status RecordReader::PositionInputStream(uint64 offset) {
  int64_t curr_pos = input_stream_->Tell();
  int64_t desired_pos = static_cast<int64_t>(offset);
  if (curr_pos > desired_pos || curr_pos < 0 /* EOF */ ||
      (curr_pos == desired_pos && last_read_failed_)) {
    last_read_failed_ = false;
    TF_RETURN_IF_ERROR(input_stream_->Reset());
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos));
  } else if (curr_pos < desired_pos) {
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos - curr_pos));
  }
  DCHECK_EQ(desired_pos, input_stream_->Tell());
  return OkStatus();
}

Status RecordReader::ReadRecord(uint64* offset, tstring* record) {
  TF_RETURN_IF_ERROR(PositionInputStream(*offset));

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
      s = errors::DataLoss("truncated record at ", *offset, "' failed with ",
                           s.error_message());
    }
    return s;
  }

  *offset += kHeaderSize + length + kFooterSize;
  DCHECK_EQ(*offset, input_stream_->Tell());
  return OkStatus();
}

Status RecordReader::SkipRecords(uint64* offset, int num_to_skip,
                                 int* num_skipped) {
  TF_RETURN_IF_ERROR(PositionInputStream(*offset));

  Status s;
  tstring record;
  *num_skipped = 0;
  for (int i = 0; i < num_to_skip; ++i) {
    s = ReadChecksummed(*offset, sizeof(uint64), &record);
    if (!s.ok()) {
      last_read_failed_ = true;
      return s;
    }
    const uint64 length = core::DecodeFixed64(record.data());

    // Skip data
    s = input_stream_->SkipNBytes(length + kFooterSize);
    if (!s.ok()) {
      last_read_failed_ = true;
      if (errors::IsOutOfRange(s)) {
        s = errors::DataLoss("truncated record at ", *offset, "' failed with ",
                             s.error_message());
      }
      return s;
    }
    *offset += kHeaderSize + length + kFooterSize;
    DCHECK_EQ(*offset, input_stream_->Tell());
    (*num_skipped)++;
  }
  return OkStatus();
}

SequentialRecordReader::SequentialRecordReader(
    RandomAccessFile* file, const RecordReaderOptions& options)
    : underlying_(file, options), offset_(0) {}

}  // namespace io
}  // namespace tsl
