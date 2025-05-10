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

#include "tensorflow/cc/saved_model/fingerprinting.h"

#include <cstdint>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/fingerprinting_utils.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/regularization/simple_delete.h"
#include "tensorflow/core/graph/regularization/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tsl/platform/random.h"
// b/291933687, b/291001524
#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
#include "tensorflow/tools/proto_splitter/cc/util.h"
#endif
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
// IWYU pragma: no_include "third_party/protobuf/io/coded_stream.h"
// IWYU pragma: no_include "third_party/protobuf/io/zero_copy_stream_impl_lite.h"

namespace tensorflow::saved_model::fingerprinting {

namespace {

using ::tensorflow::protobuf::Map;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::tensorflow::protobuf::io::CodedOutputStream;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::tensorflow::protobuf::io::StringOutputStream;

// TODO(b/290063184): remove when USM is GA
uint64_t HashCheckpointIndexFile(absl::string_view model_dir) {
  std::string meta_filename = MetaFilename(io::JoinPath(
      model_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename));
  std::string data;
  absl::Status read_status =
      ReadFileToString(Env::Default(), meta_filename, &data);
  if (read_status.ok()) {
    return tensorflow::Fingerprint64(data);
  } else {
    LOG(WARNING) << "Failed to read checkpoint file: " << read_status;
    return 0;
  }
}

uint64_t HashSavedModel(const SavedModel& saved_model) {
  std::string saved_model_serialized;
  {
    // Local scope guarantees coded stream will be trimmed (ensures
    // serialization determinism).
    // Unfortunately the saving process itself isn't deterministic, so the
    // checksum may still change since the saved_model proto may be different.
    StringOutputStream stream(&saved_model_serialized);
    CodedOutputStream output(&stream);
    output.SetSerializationDeterministic(true);
    saved_model.SerializeToCodedStream(&output);
  }
  return tensorflow::Fingerprint64(saved_model_serialized);
}

uint64_t RegularizeAndHashSignatureDefs(
    const Map<std::string, SignatureDef>& signature_def_map) {
  // Sort `signature_def_map`, which is an unordered map from string keys to
  // SignatureDefs.
  absl::btree_map<std::string, SignatureDef> sorted_signature_defs;
  sorted_signature_defs.insert(signature_def_map.begin(),
                               signature_def_map.end());
  uint64_t result_hash = 0;
  for (const auto& item : sorted_signature_defs) {
    result_hash =
        FingerprintCat64(result_hash, tensorflow::Fingerprint64(item.first));
    std::string signature_def_serialized;
    {
      StringOutputStream stream(&signature_def_serialized);
      CodedOutputStream output(&stream);
      output.SetSerializationDeterministic(true);
      item.second.SerializeToCodedStream(&output);
    }
    result_hash = FingerprintCat64(
        result_hash, tensorflow::Fingerprint64(signature_def_serialized));
  }
  return result_hash;
}

// The SavedObjectGraph contains two parts: the list of nodes and the map of
// concrete functions. Regularization treats these two parts separately.
absl::StatusOr<uint64_t> RegularizeAndHashSavedObjectGraph(
    const SavedObjectGraph& object_graph_def) {
  // Sort `concrete_functions`, which is an unordered map from function names to
  // SavedConcreteFunction, using the suffix UID of the function name. Assumes
  // that the trackable children are listed in a deterministic order during
  // serialization.
  absl::btree_map<int64_t, std::string> uid_to_function_names;
  for (const auto& [name, concrete_function] :
       object_graph_def.concrete_functions()) {
    // All valid function names should end in an UID.
    TF_ASSIGN_OR_RETURN(int64_t uid, graph_regularization::GetSuffixUID(name));
    uid_to_function_names.insert({uid, name});
  }
  uint64_t result_hash = 0;
  for (const auto& [uid, function_name] : uid_to_function_names) {
    // Hash the function name (with the UID stripped).
    result_hash = FingerprintCat64(result_hash,
                                   tensorflow::Fingerprint64(absl::StripSuffix(
                                       function_name, std::to_string(uid))));
    // Hash the serialized concrete function.
    std::string concrete_function_serialized;
    {
      StringOutputStream stream(&concrete_function_serialized);
      CodedOutputStream output(&stream);
      output.SetSerializationDeterministic(true);
      object_graph_def.concrete_functions()
          .at(function_name)
          .SerializeToCodedStream(&output);
    }
    result_hash = FingerprintCat64(
        result_hash, tensorflow::Fingerprint64(concrete_function_serialized));
  }
  // TODO(b/241294832): Complete canonicalization of `object_graph_def.nodes`.
  return result_hash;
}

// Creates a FingerprintDef proto from a SavedModel and the checkpoint meta file
// (.index) in `export_dir`.
absl::StatusOr<FingerprintDef> CreateFingerprintDefPb(
    absl::string_view export_dir, std::string pb_file) {
  // Version of the code that produced the fingerprint.
  const int kFingerprintProducer = 1;

  SavedModel saved_model;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), pb_file, &saved_model));

  // Create a copy of `metagraph` which will be used and mutated for fingerprint
  // computation.
  FingerprintDef fingerprint_def;
  MetaGraphDef* metagraph = saved_model.mutable_meta_graphs(0);
  // Set fingerprint field #1.
  fingerprint_def.set_saved_model_checksum(HashSavedModel(saved_model));
  // Set fingerprint field #2.
  graph_regularization::SimpleDelete(*metagraph->mutable_graph_def());
  fingerprint_def.set_graph_def_program_hash(
      graph_regularization::ComputeHash(metagraph->graph_def()));
  // Set fingerprint field #3.
  fingerprint_def.set_signature_def_hash(
      RegularizeAndHashSignatureDefs(metagraph->signature_def()));
  // Set fingerprint field #4.
  TF_ASSIGN_OR_RETURN(
      uint64_t object_graph_hash,
      RegularizeAndHashSavedObjectGraph(metagraph->object_graph_def()));
  fingerprint_def.set_saved_object_graph_hash(object_graph_hash);
  // Set fingerprint field #5.
  fingerprint_def.set_checkpoint_hash(HashCheckpointIndexFile(export_dir));
  // Assign a random UUID to the fingerprint.
  fingerprint_def.set_uuid(fingerprinting::CreateRandomUUID());
  // Set version of the fingerprint.
  VersionDef* version = fingerprint_def.mutable_version();
  version->set_producer(kFingerprintProducer);

  return fingerprint_def;
}

}  // namespace

absl::StatusOr<FingerprintDef> CreateFingerprintDef(
    absl::string_view export_dir) {
  std::string prefix = io::JoinPath(export_dir, kSavedModelFilenamePrefix);

#if !defined(PLATFORM_WINDOWS) && !defined(__APPLE__)
  TF_ASSIGN_OR_RETURN(bool only_contains_pb,
                      tools::proto_splitter::OnlyContainsPb(prefix));
  if (only_contains_pb) {
    return CreateFingerprintDefPb(export_dir, absl::StrCat(prefix, ".pb"));
  }

  return CreateFingerprintDefCpb(export_dir, absl::StrCat(prefix, ".cpb"));
#else
  return CreateFingerprintDefPb(export_dir, absl::StrCat(prefix, ".pb"));
#endif
}

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
  TF_ASSIGN_OR_RETURN(const FingerprintDef fingerprint_def,
                      ReadSavedModelFingerprint(export_dir));
  return Singleprint(fingerprint_def);
}

}  // namespace tensorflow::saved_model::fingerprinting
