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
#include "tensorflow/core/data/service/snapshot/file_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/random.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kTempFileSuffix[] = ".tmp";

absl::Status AtomicallyWrite(
    absl::string_view filename, tsl::Env* env,
    absl::FunctionRef<absl::Status(const std::string&)> nonatomically_write) {
  std::string uncommitted_filename = absl::StrCat(filename, "__");
  if (!env->CreateUniqueFileName(&uncommitted_filename, kTempFileSuffix)) {
    return tsl::errors::Internal("Failed to write file ", filename,
                                 ": Unable to create temporary files.");
  }
  TF_RETURN_IF_ERROR(nonatomically_write(uncommitted_filename));
  absl::Status status =
      env->RenameFile(uncommitted_filename, std::string(filename));
  if (!status.ok()) {
    return tsl::errors::Internal("Failed to rename file: ", status.ToString(),
                                 ". Source: ", uncommitted_filename,
                                 ", destination: ", filename);
  }
  return status;
}
}  // namespace

absl::Status AtomicallyWriteStringToFile(absl::string_view filename,
                                         absl::string_view str, tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncommitted_filename) {
    TF_RETURN_IF_ERROR(WriteStringToFile(env, uncommitted_filename, str));
    return absl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write string: ", str);
  return absl::OkStatus();
}

absl::Status AtomicallyWriteBinaryProto(absl::string_view filename,
                                        const tsl::protobuf::Message& proto,
                                        tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncommitted_filename) {
    TF_RETURN_IF_ERROR(WriteBinaryProto(env, uncommitted_filename, proto));
    return absl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write proto in binary format: ", proto.DebugString());
  return absl::OkStatus();
}

absl::Status AtomicallyWriteTextProto(absl::string_view filename,
                                      const tsl::protobuf::Message& proto,
                                      tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncommitted_filename) {
    TF_RETURN_IF_ERROR(WriteTextProto(env, uncommitted_filename, proto));
    return absl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write proto in text format: ", proto.DebugString());
  return absl::OkStatus();
}

absl::Status AtomicallyWriteTFRecords(absl::string_view filename,
                                      const std::vector<Tensor>& tensors,
                                      absl::string_view compression,
                                      tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncommitted_filename) {
    snapshot_util::TFRecordWriter writer(uncommitted_filename,
                                         std::string(compression));
    TF_RETURN_IF_ERROR(writer.Initialize(env));
    TF_RETURN_IF_ERROR(writer.WriteTensors(tensors));
    return writer.Close();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      " Requested to atomically write TF record file: ", filename);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> GetChildren(
    absl::string_view directory, tsl::Env* env) {
  std::vector<std::string> files, result;
  TF_RETURN_IF_ERROR(env->FileExists(std::string(directory)));
  absl::Status status = env->GetChildren(std::string(directory), &files);
  if (absl::IsNotFound(status)) {
    return result;
  }

  for (std::string& file : files) {
    if (!IsTemporaryFile(file)) {
      result.push_back(std::move(file));
    }
  }
  return result;
}

bool IsTemporaryFile(absl::string_view filename) {
  return absl::EndsWith(filename, kTempFileSuffix);
}

int64_t SnapshotChunksCardinality(absl::string_view snapshot_path,
                                  tsl::Env* env) {
  if (!env->FileExists(SnapshotDoneFilePath(snapshot_path)).ok()) {
    return kUnknownCardinality;
  }
  absl::StatusOr<std::vector<std::string>> chunks =
      GetChildren(CommittedChunksDirectory(snapshot_path), env);
  if (!chunks.ok()) {
    return kUnknownCardinality;
  }
  return chunks->size();
}

}  // namespace data
}  // namespace tensorflow
