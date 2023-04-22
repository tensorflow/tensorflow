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

#ifndef TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/snappy/snappy_compression_options.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

struct RecordReaderOptions {
  enum CompressionType {
    NONE = 0,
    ZLIB_COMPRESSION = 1,
    SNAPPY_COMPRESSION = 2
  };
  CompressionType compression_type = NONE;

  // If buffer_size is non-zero, then all reads must be sequential, and no
  // skipping around is permitted. (Note: this is the same behavior as reading
  // compressed files.) Consider using SequentialRecordReader.
  int64 buffer_size = 0;

  static RecordReaderOptions CreateRecordReaderOptions(
      const string& compression_type);

#if !defined(IS_SLIM_BUILD)
  // Options specific to compression.
  ZlibCompressionOptions zlib_options;
  SnappyCompressionOptions snappy_options;
#endif  // IS_SLIM_BUILD
};

// Low-level interface to read TFRecord files.
//
// If using compression or buffering, consider using SequentialRecordReader.
//
// Note: this class is not thread safe; external synchronization required.
class RecordReader {
 public:
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  static constexpr size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static constexpr size_t kFooterSize = sizeof(uint32);

  // Statistics (sizes are in units of bytes)
  struct Stats {
    int64 file_size = -1;
    int64 data_size = -1;
    int64 entries = -1;  // Number of values
  };

  // Metadata for the TFRecord file.
  struct Metadata {
    Stats stats;
  };

  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit RecordReader(
      RandomAccessFile* file,
      const RecordReaderOptions& options = RecordReaderOptions());

  virtual ~RecordReader() = default;

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, tstring* record);

  // Skip num_to_skip record starting at "*offset" and update *offset
  // to point to the offset of the next num_to_skip + 1 record.
  // Return OK on success, OUT_OF_RANGE for end of file, or something
  // else for an error. "*num_skipped" records the number of records that
  // are actually skipped. It should be equal to num_to_skip on success.
  Status SkipRecords(uint64* offset, int num_to_skip, int* num_skipped);

  // Return the metadata of the Record file.
  //
  // The current implementation scans the file to completion,
  // skipping over the data regions, to extract the metadata once
  // on the first call to GetStats().  An improved implementation
  // would change RecordWriter to write the metadata into TFRecord
  // so that GetMetadata() could be a const method.
  //
  // 'metadata' must not be nullptr.
  Status GetMetadata(Metadata* md);

 private:
  Status ReadChecksummed(uint64 offset, size_t n, tstring* result);
  Status PositionInputStream(uint64 offset);

  RecordReaderOptions options_;
  std::unique_ptr<InputStreamInterface> input_stream_;
  bool last_read_failed_;

  std::unique_ptr<Metadata> cached_metadata_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecordReader);
};

// High-level interface to read TFRecord files.
//
// Note: this class is not thread safe; external synchronization required.
class SequentialRecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit SequentialRecordReader(
      RandomAccessFile* file,
      const RecordReaderOptions& options = RecordReaderOptions());

  virtual ~SequentialRecordReader() = default;

  // Read the next record in the file into *record. Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(tstring* record) {
    return underlying_.ReadRecord(&offset_, record);
  }

  // Skip the next num_to_skip record in the file. Return OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  // "*num_skipped" records the number of records that are actually skipped.
  // It should be equal to num_to_skip on success.
  Status SkipRecords(int num_to_skip, int* num_skipped) {
    return underlying_.SkipRecords(&offset_, num_to_skip, num_skipped);
  }

  // Return the current offset in the file.
  uint64 TellOffset() { return offset_; }

  // Seek to this offset within the file and set this offset as the current
  // offset. Trying to seek backward will throw error.
  Status SeekOffset(uint64 offset) {
    if (offset < offset_)
      return errors::InvalidArgument(
          "Trying to seek offset: ", offset,
          " which is less than the current offset: ", offset_);
    offset_ = offset;
    return Status::OK();
  }

 private:
  RecordReader underlying_;
  uint64 offset_ = 0;
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
