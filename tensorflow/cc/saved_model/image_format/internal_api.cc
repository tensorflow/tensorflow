/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/image_format/internal_api.h"

#include <string>
#include <tuple>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
// TODO(b/291933687), TODO(b/291001524)
#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
#include "tensorflow/tools/proto_splitter/cc/saved_model_splitter.h"
#include "tensorflow/tools/proto_splitter/merge.h"
#endif
#define IS_OSS false
namespace tensorflow {
namespace image_format {

absl::Status ReadSavedModel(const std::string& file_prefix,
                            SavedModel* saved_model_proto) {
  LOG(INFO) << "Reading SavedModel from: " << file_prefix;

#if defined(PLATFORM_WINDOWS) || defined(__APPLE__)
  const std::string saved_model_pb_path = absl::StrCat(file_prefix, ".pb");
  TF_ASSIGN_OR_RETURN(
      bool saved_model_pb_exists,
      internal::FileExists(Env::Default(), saved_model_pb_path));
  if (saved_model_pb_exists) {
    absl::Status result =
        ReadBinaryProto(Env::Default(), saved_model_pb_path, saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelReadCount(
          saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }
#endif

  // TODO(b/295208714): add pbtxt support to Merger::Read
  const std::string saved_model_pbtxt_path =
      absl::StrCat(file_prefix, ".pbtxt");
  auto saved_model_pbtxt_exists =
      internal::FileExists(Env::Default(), saved_model_pbtxt_path);
  if (saved_model_pbtxt_exists.value_or(false)) {
    absl::Status result = ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                        saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelReadCount(
          saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }

#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
  absl::Status result =
      tools::proto_splitter::Merger::Read(file_prefix, saved_model_proto);
  if (result.ok()) {
    metrics::SavedModelReadCount(
        saved_model::GetWriteVersion(*saved_model_proto))
        .IncrementBy(1);
  }
  return result;
#endif

  return absl::Status(
      absl::StatusCode::kNotFound,
      absl::StrCat("Could not find SavedModel .pb or .pbtxt at supplied "
                   "file prefix: ",
                   file_prefix,
                   ". Check that "
                   "the directory exists and that you have the right "
                   "permissions for accessing it."));
}

absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix) {
#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
  tools::proto_splitter::SavedModelSplitter splitter(saved_model_proto);
  return splitter.Write(file_prefix);
#else
  return absl::UnimplementedError(
      "WriteSavedModel not implemented for Windows or MacOS.");
#endif
}

absl::StatusOr<std::tuple<std::string, bool>> WriteSavedModelToString(
    SavedModel* saved_model_proto) {
#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
  tools::proto_splitter::SavedModelSplitter splitter(saved_model_proto);
  return splitter.WriteToString();
#else
  return absl::UnimplementedError(
      "WriteSavedModelToString not implemented for Windows or MacOS.");
#endif
}

#if !IS_OSS
// TODO(b/311769337): Define the function unconditionally after tf oss
// dependency is updated to protobuf v22.x.
absl::StatusOr<std::tuple<absl::Cord, bool>> WriteSavedModelToCord(
    SavedModel* saved_model_proto) {
  tools::proto_splitter::SavedModelSplitter splitter(saved_model_proto);
  return splitter.WriteToCord();
}
#endif

absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix,
                             int debug_max_size) {
#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
  tools::proto_splitter::DebugSetMaxSize(debug_max_size);
  return WriteSavedModel(saved_model_proto, file_prefix);
#else
  return absl::UnimplementedError(
      "WriteSavedModel not implemented for Windows or MacOS.");
#endif
}

}  // namespace image_format
}  // namespace tensorflow
