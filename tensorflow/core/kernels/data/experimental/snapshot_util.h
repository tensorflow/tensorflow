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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class GraphDef;

namespace data {

namespace experimental {

class SnapshotMetadataRecord;
class SnapshotTensorMetadata;

}  // namespace experimental

namespace snapshot_util {

constexpr char kMetadataFilename[] = "snapshot.metadata";

constexpr char kModeAuto[] = "auto";
constexpr char kModeWrite[] = "write";
constexpr char kModeRead[] = "read";
constexpr char kModePassthrough[] = "passthrough";

enum Mode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

class Writer {
 public:
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotWriter";
  static constexpr const char* const kWriteStringPiece = "WriteStringPiece";
  static constexpr const char* const kWriteCord = "WriteCord";
  static constexpr const char* const kSeparator = "::";

  static Status Create(Env* env, const std::string& filename,
                       const std::string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Writer>* out_writer);

  Status WriteTensors(const std::vector<Tensor>& tensors);

  Status Sync();

  Status Close();

  ~Writer();

 private:
  explicit Writer(const std::string& filename,
                  const std::string& compression_type, int version,
                  const DataTypeVector& dtypes);

  Status Initialize(tensorflow::Env* env);

  Status WriteRecord(const StringPiece& data);

#if defined(PLATFORM_GOOGLE)
  Status WriteRecord(const absl::Cord& data);
#endif  // PLATFORM_GOOGLE

  std::unique_ptr<WritableFile> dest_;
  const std::string filename_;
  const std::string compression_type_;
  const int version_;
  const DataTypeVector dtypes_;
  // We hold zlib_dest_ because we may create a ZlibOutputBuffer and put that
  // in dest_ if we want compression. ZlibOutputBuffer doesn't own the original
  // dest_ and so we need somewhere to store the original one.
  std::unique_ptr<WritableFile> zlib_underlying_dest_;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
  int num_simple_ = 0;
  int num_complex_ = 0;
};

class Reader {
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

  static Status Create(Env* env, const std::string& filename,
                       const string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Reader>* out_reader);

  // Returns a nested dataset for a set of given snapshot file names.
  //
  // This function takes a vector of snapshot files, and returns a nested
  // dataset. Each element within the nested dataset is itself a dataset, and
  // contains all the elements written out to each individual snapshot file.
  static Status MakeNestedDataset(Env* env,
                                  const std::vector<std::string>& filenames,
                                  const string& compression_type, int version,
                                  const DataTypeVector& dtypes,
                                  const std::vector<PartialTensorShape>& shapes,
                                  DatasetBase** output);

  Status ReadTensors(std::vector<Tensor>* read_tensors);

 private:
  explicit Reader(const std::string& filename, const string& compression_type,
                  int version, const DataTypeVector& dtypes);

  Status Initialize(Env* env);

  Status ReadTensorsV0(std::vector<Tensor>* read_tensors);

  Status SnappyUncompress(
      const experimental::SnapshotTensorMetadata* metadata,
      std::vector<Tensor>* simple_tensors,
      std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
          tensor_proto_strs);

  Status ReadRecord(tstring* record);

#if defined(PLATFORM_GOOGLE)
  Status ReadRecord(absl::Cord* record);
#endif

  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
  const int version_;
  const DataTypeVector dtypes_;
  int num_simple_ = 0;
  int num_complex_ = 0;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.

  class Dataset;
  class NestedDataset;
};

Status WriteMetadataFile(const string& hash_dir,
                         const experimental::SnapshotMetadataRecord* metadata);

Status ReadMetadataFile(const string& hash_dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists);

Status DumpDatasetGraph(const std::string& path, uint64 hash,
                        const GraphDef* graph);

Status DetermineOpState(const std::string& mode_string, bool file_exists,
                        const experimental::SnapshotMetadataRecord* metadata,
                        const uint64 pending_snapshot_expiry_seconds,
                        Mode* mode);

}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SNAPSHOT_UTIL_H_
