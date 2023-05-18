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

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "absl/container/btree_map.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/regularization/simple_delete.h"
#include "tensorflow/core/graph/regularization/util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/tsl/platform/types.h"

namespace tensorflow::saved_model::fingerprinting {

// Version of the code that produced the fingerprint.
const int kFingerprintProducer = 1;
namespace {

uint64 HashSavedModel(const SavedModel& saved_model) {
  std::string saved_model_string;
  SerializeToStringDeterministic(saved_model, &saved_model_string);
  return tensorflow::Fingerprint64(saved_model_string);
}

uint64 RegularizeAndHashSignatureDefs(
    const google::protobuf::Map<std::string, SignatureDef>& signature_def_map) {
  // Sort `signature_def_map`, which is an unordered map from string keys to
  // SignatureDefs.
  absl::btree_map<std::string, SignatureDef> sorted_signature_defs;
  sorted_signature_defs.insert(signature_def_map.begin(),
                               signature_def_map.end());
  uint64 result_hash = 0;
  for (const auto& item : sorted_signature_defs) {
    std::string signature_def_string;
    SerializeToStringDeterministic(item.second, &signature_def_string);
    result_hash = FingerprintCat64(
        result_hash, tensorflow::Fingerprint64(signature_def_string));
  }
  return result_hash;
}

// The SavedObjectGraph contains two parts: the list of nodes and the map of
// concrete functions. Regularization treats these two parts separately.
StatusOr<uint64> RegularizeAndHashSavedObjectGraph(
    const SavedObjectGraph& object_graph_def) {
  // Sort `concrete_functions`, which is an unordered map from function names to
  // SavedConcreteFunction, using the suffix UID of the function name. Assumes
  // that the trackable children are listed in a deterministic order during
  // serialization.
  absl::btree_map<int, std::string> uid_to_function_names;
  for (const auto& [name, concrete_function] :
       object_graph_def.concrete_functions()) {
    // All valid function names should end in an UID.
    TF_ASSIGN_OR_RETURN(int uid, graph_regularization::GetSuffixUID(name));
    uid_to_function_names.insert({uid, name});
  }
  uint64 result_hash = 0;
  for (const auto& [uid, function_name] : uid_to_function_names) {
    // Hash the function name (with the UID stripped).
    result_hash = FingerprintCat64(result_hash,
                                   tensorflow::Fingerprint64(absl::StripSuffix(
                                       function_name, std::to_string(uid))));
    // Hash the serialized concrete function.
    std::string concrete_function_string;
    SerializeToStringDeterministic(
        object_graph_def.concrete_functions().at(function_name),
        &concrete_function_string);
    result_hash = FingerprintCat64(
        result_hash, tensorflow::Fingerprint64(concrete_function_string));
  }
  // TODO(b/241294832): Complete canonicalization of `object_graph_def.nodes`.
  return result_hash;
}

// Returns the hash of the checkpoint .index file, 0 if there is none.
uint64 HashCheckpointIndexFile(absl::string_view model_dir) {
  std::string meta_filename = MetaFilename(io::JoinPath(
      model_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename));
  std::string data;
  Status read_status = ReadFileToString(Env::Default(), meta_filename, &data);
  if (read_status.ok()) {
    return tensorflow::Fingerprint64(data);
  } else {
    return 0;
  }
}

}  // namespace

StatusOr<FingerprintDef> CreateFingerprintDef(const SavedModel& saved_model,
                                              absl::string_view export_dir) {
  // Create a copy of `metagraph` which will be used and mutated for fingerprint
  // computation.
  MetaGraphDef metagraph_copy = saved_model.meta_graphs(0);
  FingerprintDef fingerprint_def;
  // Set fingerprint field #1.
  fingerprint_def.set_saved_model_checksum(HashSavedModel(saved_model));
  // Set fingerprint field #2.
  graph_regularization::SimpleDelete(*metagraph_copy.mutable_graph_def());
  fingerprint_def.set_graph_def_program_hash(
      graph_regularization::ComputeHash(metagraph_copy.graph_def()));
  // Set fingerprint field #3.
  fingerprint_def.set_signature_def_hash(
      RegularizeAndHashSignatureDefs(metagraph_copy.signature_def()));
  // Set fingerprint field #4.
  TF_ASSIGN_OR_RETURN(
      StatusOr<uint64> object_graph_hash,
      RegularizeAndHashSavedObjectGraph(metagraph_copy.object_graph_def()));
  fingerprint_def.set_saved_object_graph_hash(object_graph_hash.value());
  // Set fingerprint field #5.
  fingerprint_def.set_checkpoint_hash(HashCheckpointIndexFile(export_dir));
  // Set version of the fingerprint.
  VersionDef* version = fingerprint_def.mutable_version();
  version->set_producer(kFingerprintProducer);

  return fingerprint_def;
}

StatusOr<FingerprintDef> ReadSavedModelFingerprint(
    absl::string_view export_dir) {
  const string fingerprint_pb_path =
      io::JoinPath(export_dir, kFingerprintFilenamePb);
  Status found_pb = Env::Default()->FileExists(fingerprint_pb_path);
  if (!found_pb.ok()) return found_pb;

  FingerprintDef fingerprint_proto;
  Status result =
      ReadBinaryProto(Env::Default(), fingerprint_pb_path, &fingerprint_proto);
  if (!result.ok()) return result;

  return fingerprint_proto;
}

std::string Singleprint(uint64 graph_def_program_hash,
                        uint64 signature_def_hash,
                        uint64 saved_object_graph_hash,
                        uint64 checkpoint_hash) {
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

std::string Singleprint(absl::string_view export_dir) {
  FingerprintDef fingerprint = ReadSavedModelFingerprint(export_dir).value();
  return Singleprint(fingerprint);
}

}  // namespace tensorflow::saved_model::fingerprinting
