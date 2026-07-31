/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/singleprint.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::saved_model::fingerprinting {

namespace {

absl::StatusOr<FingerprintDef> ReadSavedModelFingerprint(
    absl::string_view export_dir) {
  const std::string fingerprint_pb_path =
      io::JoinPath(export_dir, kFingerprintFilenamePb);
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(fingerprint_pb_path));

  FingerprintDef fingerprint_proto;
  absl::Status result =
      ReadBinaryProto(Env::Default(), fingerprint_pb_path, &fingerprint_proto);
  if (!result.ok()) return result;

  return fingerprint_proto;
}

}  // namespace

std::string Singleprint(uint64_t graph_def_program_hash,
                        uint64_t signature_def_hash,
                        uint64_t saved_object_graph_hash,
                        uint64_t checkpoint_hash) {
  return std::to_string(graph_def_program_hash) + "/" +
         std::to_string(signature_def_hash) + "/" +
         std::to_string(saved_object_graph_hash) + "/" +
         std::to_string(checkpoint_hash);
}

std::string Singleprint(const FingerprintDef& fingerprint) {
  return Singleprint(
      fingerprint.graph_def_program_hash(), fingerprint.signature_def_hash(),
      fingerprint.saved_object_graph_hash(), fingerprint.checkpoint_hash());
}

absl::StatusOr<std::string> Singleprint(absl::string_view export_dir) {
  TF_ASSIGN_OR_RETURN(FingerprintDef fingerprint_def,
                      ReadSavedModelFingerprint(export_dir));
  return Singleprint(fingerprint_def);
}

}  // namespace tensorflow::saved_model::fingerprinting
