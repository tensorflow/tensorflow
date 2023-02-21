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
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64_t kTFRecordReaderOutputBufferSize = 256 << 20;  // 256MB

}

SnapshotReader::SnapshotReader(const SnapshotReaderParams& params)
    : params_(params) {}

StatusOr<GetNextResult> SnapshotReader::GetNext() {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  while (!end_of_sequence_) {
    GetNextResult result;
    Status status = tfrecord_reader_->ReadTensors(&result.tensors);
    if (status.ok()) {
      return result;
    }
    if (!errors::IsOutOfRange(status)) {
      return status;
    }
    TF_RETURN_IF_ERROR(InitializeNextRecordReader());
  }
  return GetNextResult::EndOfSequence();
}

Status SnapshotReader::EnsureInitialized() {
  if (!chunk_files_.empty()) {
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(chunk_files_, GetChunkFiles());
  TF_RETURN_IF_ERROR(InitializeNextRecordReader());
  if (end_of_sequence_) {
    return errors::NotFound("Failed to read distributed tf.data snapshot ",
                            params_.DebugString(), ": No snapshot is written.");
  }
  return OkStatus();
}

StatusOr<std::vector<std::string>> SnapshotReader::GetChunkFiles() {
  std::string chunks_directory = params_.CommittedChunksDirectory();
  std::vector<string> chunk_files;
  TF_RETURN_IF_ERROR(params_.env->GetChildren(chunks_directory, &chunk_files));
  for (std::string& chunk_file : chunk_files) {
    chunk_file = tsl::io::JoinPath(chunks_directory, chunk_file);
  }
  return chunk_files;
}

Status SnapshotReader::InitializeNextRecordReader() {
  if (next_chunk_index_ >= chunk_files_.size()) {
    end_of_sequence_ = true;
    tfrecord_reader_ = nullptr;
    return OkStatus();
  }
  tfrecord_reader_ = std::make_unique<snapshot_util::TFRecordReader>(
      chunk_files_[next_chunk_index_], params_.metadata.compression(),
      params_.output_types, kTFRecordReaderOutputBufferSize);
  TF_RETURN_IF_ERROR(tfrecord_reader_->Initialize(params_.env));
  LOG(INFO) << "Starting to read distributed tf.data snapshot "
            << params_.DebugString() << ", chunk " << next_chunk_index_;
  ++next_chunk_index_;
  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
