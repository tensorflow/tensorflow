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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SNAPSHOT_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SNAPSHOT_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {
namespace experimental {

class SnapshotTensorMetadata;

class SnapshotWriter {
 public:
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotWriter";
  static constexpr const char* const kWriteStringPiece = "WriteStringPiece";
  static constexpr const char* const kWriteCord = "WriteCord";
  static constexpr const char* const kSeparator = "::";

  explicit SnapshotWriter(WritableFile* dest, const string& compression_type,
                          int version, const DataTypeVector& dtypes);

  Status WriteTensors(const std::vector<Tensor>& tensors);

  Status Sync();

  Status Close();

  ~SnapshotWriter();

 private:
  Status WriteRecord(const StringPiece& data);

#if defined(PLATFORM_GOOGLE)
  Status WriteRecord(const absl::Cord& data);
#endif  // PLATFORM_GOOGLE

  WritableFile* dest_;
  bool dest_is_owned_ = false;
  const string compression_type_;
  const int version_;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
  int num_simple_ = 0;
  int num_complex_ = 0;
};

class SnapshotReader {
 public:
  // The reader input buffer size is deliberately large because the input reader
  // will throw an error if the compressed block length cannot fit in the input
  // buffer.
  static constexpr const int64 kSnappyReaderInputBufferSizeBytes =
      1 << 30;  // 1 GiB
  // TODO(b/148804377): Set this in a smarter fashion.
  static constexpr const int64 kSnappyReaderOutputBufferSizeBytes =
      32 << 20;  // 32 MiB
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotReader";
  static constexpr const char* const kReadString = "ReadString";
  static constexpr const char* const kReadCord = "ReadCord";
  static constexpr const char* const kSeparator = "::";

  explicit SnapshotReader(RandomAccessFile* file,
                          const string& compression_type, int version,
                          const DataTypeVector& dtypes);

  Status ReadTensors(std::vector<Tensor>* read_tensors);

 private:
  Status ReadTensorsV0(std::vector<Tensor>* read_tensors);

  Status SnappyUncompress(
      const SnapshotTensorMetadata* metadata,
      std::vector<Tensor>* simple_tensors,
      std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
          tensor_proto_strs);

  Status ReadRecord(tstring* record);

#if defined(PLATFORM_GOOGLE)
  Status ReadRecord(absl::Cord* record);
#endif

  RandomAccessFile* file_;
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
  const int version_;
  const DataTypeVector dtypes_;
  int num_simple_ = 0;
  int num_complex_ = 0;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SNAPSHOT_UTIL_H_
