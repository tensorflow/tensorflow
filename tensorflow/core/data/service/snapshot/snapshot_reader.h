/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_READER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_READER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

struct SnapshotReaderParams {
  // The directory path of the snapshot. See the comment on SnapshotManager for
  // how the directory is structured.
  std::string snapshot_path;

  // Distributed snapshot metadata.
  experimental::DistributedSnapshotMetadata metadata;

  // Data types of the snapshot data elements.
  DataTypeVector dtypes;

  // Data shape of the snapshot data elements.
  std::vector<PartialTensorShape> shapes;

  // The Tensorflow environment.
  Env* env = nullptr;

  std::string CommittedChunksDirectory() const {
    return tensorflow::data::CommittedChunksDirectory(snapshot_path);
  }

  std::string DebugString() const {
    return absl::Substitute(
        "SnapshotReaderParams { base_path: $0, metadata: $1 }", snapshot_path,
        metadata.DebugString());
  }
};

// Creates a dataset that reads tf.data distributed snapshots.
Status MakeSnapshotReaderDataset(
    const SnapshotReaderParams& params,
    InstantiatedCapturedFunction& instantiated_captured_func,
    IteratorContext* ctx, core::RefCountPtr<DatasetBase>* output);

// Reads a distributed tf.data snapshot written by `SnapshotManager` and
// `SnapshotStreamWriter`. See the comment on SnapshotManager for
// how the directory is structured.
// TODO(b/258691097): Deprecate this API.
class SnapshotReader {
 public:
  explicit SnapshotReader(const SnapshotReaderParams& params);
  virtual ~SnapshotReader() = default;
  SnapshotReader(const SnapshotReader&) = delete;
  SnapshotReader& operator=(const SnapshotReader&) = delete;

  // Gets the next element from the snapshot.
  StatusOr<GetNextResult> GetNext();

 private:
  // Initializes the reader if it's not already initialized. This is called when
  // `GetNext` is first called.
  Status EnsureInitialized();
  // Returns a list of the committed chunks.
  StatusOr<std::vector<std::string>> GetChunkFiles();
  // If a chunk file is exhausted, starts reading the next chunk file. If there
  // are no more files to read, `end_of_sequence_` will be set to true.
  Status InitializeNextRecordReader();

  const SnapshotReaderParams params_;

  // A list of the committed chunks to read.
  std::vector<std::string> chunk_files_;
  // The index of the next chunk to read.
  uint64_t next_chunk_index_ = 0;
  bool end_of_sequence_ = false;

  std::unique_ptr<snapshot_util::TFRecordReader> tfrecord_reader_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_READER_H_
