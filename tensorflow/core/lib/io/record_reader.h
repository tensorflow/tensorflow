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

#ifndef TENSORFLOW_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_LIB_IO_RECORD_READER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

class RecordReaderOptions {
 public:
  enum CompressionType { NONE = 0, ZLIB_COMPRESSION = 1 };
  CompressionType compression_type = NONE;

  static RecordReaderOptions CreateRecordReaderOptions(
      const string& compression_type);

#if !defined(IS_SLIM_BUILD)
  // Options specific to zlib compression.
  ZlibCompressionOptions zlib_options;
#endif  // IS_SLIM_BUILD
};

class RecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  RecordReader(RandomAccessFile* file,
               const RecordReaderOptions& options = RecordReaderOptions());

  virtual ~RecordReader();

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, string* record);

 private:
  Status ReadChecksummed(uint64 offset, size_t n, StringPiece* result,
                         string* storage);

  RandomAccessFile* src_;
  RecordReaderOptions options_;
#if !defined(IS_SLIM_BUILD)
  std::unique_ptr<RandomAccessInputStream> random_input_stream_;
  std::unique_ptr<ZlibInputStream> zlib_input_stream_;
#endif  // IS_SLIM_BUILD

  TF_DISALLOW_COPY_AND_ASSIGN(RecordReader);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RECORD_READER_H_
