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

#ifndef XLA_TSL_LIB_IO_RECORD_WRITER_H_
#define XLA_TSL_LIB_IO_RECORD_WRITER_H_

#include "xla/tsl/lib/hash/crc32c.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/coding.h"
#include "tsl/platform/stringpiece.h"
#if !defined(IS_SLIM_BUILD)
#include "xla/tsl/lib/io/snappy/snappy_compression_options.h"
#include "xla/tsl/lib/io/snappy/snappy_outputbuffer.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#endif  // IS_SLIM_BUILD
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/cord.h"

namespace tsl {

class WritableFile;

namespace io {

struct RecordWriterOptions {
 public:
  enum CompressionType {
    NONE = 0,
    ZLIB_COMPRESSION = 1,
    SNAPPY_COMPRESSION = 2
  };
  CompressionType compression_type = NONE;

  static RecordWriterOptions CreateRecordWriterOptions(
      const string& compression_type);

#if !defined(IS_SLIM_BUILD)
  // Options specific to compression.
  io::ZlibCompressionOptions zlib_options;
  io::SnappyCompressionOptions snappy_options;
#endif  // IS_SLIM_BUILD
};

class RecordWriter {
 public:
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  static constexpr size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static constexpr size_t kFooterSize = sizeof(uint32);

  // Create a writer that will append data to "*dest".
  // "*dest" must be initially empty.
  // "*dest" must remain live while this Writer is in use.
  explicit RecordWriter(WritableFile* dest, const RecordWriterOptions& options =
                                                RecordWriterOptions());

  // Calls Close() and logs if an error occurs.
  //
  // TODO(jhseu): Require that callers explicitly call Close() and remove the
  // implicit Close() call in the destructor.
  ~RecordWriter();

  absl::Status WriteRecord(absl::string_view data);

#if defined(TF_CORD_SUPPORT)
  absl::Status WriteRecord(const absl::Cord& data);
#endif

  // Flushes any buffered data held by underlying containers of the
  // RecordWriter to the WritableFile. Does *not* flush the
  // WritableFile.
  absl::Status Flush();

  // Writes all output to the file. Does *not* close the WritableFile.
  //
  // After calling Close(), any further calls to `WriteRecord()` or `Flush()`
  // are invalid.
  absl::Status Close();

  // Utility method to populate TFRecord headers.  Populates record-header in
  // "header[0,kHeaderSize-1]".  The record-header is based on data[0, n-1].
  inline static void PopulateHeader(char* header, const char* data, size_t n);

#if defined(TF_CORD_SUPPORT)
  inline static void PopulateHeader(char* header, const absl::Cord& data);
#endif

  // Utility method to populate TFRecord footers.  Populates record-footer in
  // "footer[0,kFooterSize-1]".  The record-footer is based on data[0, n-1].
  inline static void PopulateFooter(char* footer, const char* data, size_t n);

#if defined(TF_CORD_SUPPORT)
  inline static void PopulateFooter(char* footer, const absl::Cord& data);
#endif

 private:
  WritableFile* dest_;
  RecordWriterOptions options_;

  inline static uint32 MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

#if defined(TF_CORD_SUPPORT)
  inline static uint32 MaskedCrc(const absl::Cord& data) {
    return crc32c::Mask(crc32c::Value(data));
  }
#endif

  RecordWriter(const RecordWriter&) = delete;
  void operator=(const RecordWriter&) = delete;
};

void RecordWriter::PopulateHeader(char* header, const char* data, size_t n) {
  core::EncodeFixed64(header + 0, n);
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
}

void RecordWriter::PopulateFooter(char* footer, const char* data, size_t n) {
  core::EncodeFixed32(footer, MaskedCrc(data, n));
}

#if defined(TF_CORD_SUPPORT)
void RecordWriter::PopulateHeader(char* header, const absl::Cord& data) {
  core::EncodeFixed64(header + 0, data.size());
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
}

void RecordWriter::PopulateFooter(char* footer, const absl::Cord& data) {
  core::EncodeFixed32(footer, MaskedCrc(data));
}
#endif

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_RECORD_WRITER_H_
