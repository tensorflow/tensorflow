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

#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/random.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kTempFileSuffix[] = ".tmp";

tsl::Status AtomicallyWrite(
    absl::string_view filename, tsl::Env* env,
    absl::FunctionRef<tsl::Status(const std::string&)> nonatomically_write) {
  std::string uncommitted_filename(filename);
  if (!env->CreateUniqueFileName(&uncommitted_filename, kTempFileSuffix)) {
    return tsl::errors::Internal("Failed to write file ", filename,
                                 ": Unable to create temporary files.");
  }
  TF_RETURN_IF_ERROR(nonatomically_write(uncommitted_filename));
  Status status = env->RenameFile(uncommitted_filename, std::string(filename));
  if (!status.ok()) {
    return tsl::errors::Internal("Failed to rename file: ", status.ToString(),
                                 ". Source: ", uncommitted_filename,
                                 ", destination: ", filename);
  }
  return status;
}
}  // namespace

tsl::Status AtomicallyWriteStringToFile(absl::string_view filename,
                                        absl::string_view str, tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncomitted_filename) {
    TF_RETURN_IF_ERROR(WriteStringToFile(env, uncomitted_filename, str));
    return tsl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write string: ", str);
  return tsl::OkStatus();
}

tsl::Status AtomicallyWriteBinaryProto(absl::string_view filename,
                                       const tsl::protobuf::Message& proto,
                                       tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncomitted_filename) {
    TF_RETURN_IF_ERROR(WriteBinaryProto(env, uncomitted_filename, proto));
    return tsl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write proto in binary format: ", proto.DebugString());
  return tsl::OkStatus();
}

tsl::Status AtomicallyWriteTextProto(absl::string_view filename,
                                     const tsl::protobuf::Message& proto,
                                     tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncomitted_filename) {
    TF_RETURN_IF_ERROR(WriteTextProto(env, uncomitted_filename, proto));
    return tsl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write proto in text format: ", proto.DebugString());
  return tsl::OkStatus();
}

tsl::Status AtomicallyWriteTFRecord(absl::string_view filename,
                                    const Tensor& tensor,
                                    absl::string_view compression,
                                    tsl::Env* env) {
  auto nonatomically_write = [&](const std::string& uncomitted_filename) {
    snapshot_util::TFRecordWriter writer(uncomitted_filename,
                                         std::string(compression));
    TF_RETURN_IF_ERROR(writer.Initialize(env));
    TF_RETURN_IF_ERROR(writer.WriteTensors({tensor}));
    return tsl::OkStatus();
  };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      AtomicallyWrite(filename, env, nonatomically_write),
      "Requested to write tensor: ", tensor.DebugString());
  return tsl::OkStatus();
}

tsl::StatusOr<std::vector<std::string>> GetChildren(absl::string_view directory,
                                                    tsl::Env* env) {
  std::vector<std::string> files, result;
  TF_RETURN_IF_ERROR(env->FileExists(std::string(directory)));
  Status status = env->GetChildren(std::string(directory), &files);
  if (errors::IsNotFound(status)) {
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

}  // namespace data
}  // namespace tensorflow
