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

#ifndef TENSORFLOW_CORE_KERNELS_HADOOP_SEQUENCE_FILE_READER_H_
#define TENSORFLOW_CORE_KERNELS_HADOOP_SEQUENCE_FILE_READER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

struct SequenceFileReaderOptions {
  int64 buffer_size_bytes = 0;

  static const SequenceFileReaderOptions& Defaults();
};

// Simple hadoop sequence file reader.
//
// The implementation is according to
// https://github.com/colinmarc/sequencefile
//
// This is not intended to be a general purpose sequence file implementation.
// The objective is merely to enable reading serialized tf.Examples directly
// from hadoop mapreduces, bypassing the recordio conversion.
//
// Limitations:
//
// - Requires sequence file ver 5 and above.
//
// - Does not handle compression (block or record).
//
// - Keys are ignored.
//
// - Values are assumed to be encoded using org.apache.hadoop.io.BytesWritable
//
// NOT safe for concurrent use.
class SequenceFileReader {
 public:
  // Creates a sequence file reader. Dies on error. Will NOT own the input file.
  SequenceFileReader(RandomAccessFile* file,
                     const SequenceFileReaderOptions& options);
  ~SequenceFileReader() = default;
  // Reads a record. The output string will contain the raw bytes.
  //
  // Returns OK on success, OUT_OF_RANGE for the end of file, or something else
  // for an error.
  Status ReadRecord(string* value);

 private:
  int64 ReadHadoopVarIntOrDie();
  Status ReadBigEndianUint32(uint32* out);
  Status ReadBigEndianInt32(int32* out);
  void ReadHadoopVarLenStringOrDie();
  void ConsumeMetadataOrDie();

  const SequenceFileReaderOptions& options_;
  std::unique_ptr<InputStreamInterface> stream_;
  string sync_marker_;
  string buf_;

  friend class SequenceFileReaderTest;
  explicit SequenceFileReader(const SequenceFileReaderOptions& options)
    : options_(options) {}
  SequenceFileReader() = delete;
  TF_DISALLOW_COPY_AND_ASSIGN(SequenceFileReader);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HADOOP_SEQUENCE_FILE_READER_H_
