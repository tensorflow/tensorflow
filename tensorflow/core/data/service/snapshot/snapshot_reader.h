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

#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/refcount.h"
#include "tensorflow/tsl/platform/status.h"

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

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_READER_H_
