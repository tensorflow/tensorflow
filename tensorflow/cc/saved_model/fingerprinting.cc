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
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow::fingerprinting {

namespace {

// This function mutates the GraphDef, changing the names and config_proto's
// of the Function nodes.
void CanonicalizeNodes(GraphDef* orig_graph_def) {
  for (NodeDef& node : *orig_graph_def->mutable_node()) {
    // Check if this is a function call.
    if (grappler::IsPartitionedCall(node) ||
        grappler::IsStatefulPartitionedCall(node)) {
      // TODO(b/240174577): Strip UID from the end of function names.
      // Regularize "f" attribute, the function name for PartitionedCall and
      // and StatefulPartitionedCall ops.
      node.mutable_attr()->find("f")->second.mutable_func()->set_name(
          "FINGERPRINT_PASS");
      // Erase the "config_proto" attribute which contains device-specific
      // information.
      node.mutable_attr()->find("config_proto")->second.mutable_s()->erase();
    }
    // Erase the value of string constants, which can vary based on platform.
    if (grappler::IsConstant(node)) {
      if (node.attr().at("dtype").type() == DT_STRING) {
        node.mutable_attr()->find("value")->second.clear_value();
      }
    }
  }
}

// Returns the suffix UID of `function_name`.
StatusOr<int> GetSuffixUID(absl::string_view function_name) {
  std::vector<std::string> v = absl::StrSplit(function_name, '_');
  int uid;
  if (!strings::safe_strto32(v.back(), &uid)) {
    return errors::InvalidArgument(absl::StrCat(
        "Function name: `", function_name, "` does not end in an integer."));
  }
  return uid;
}

}  // namespace

uint64 ComputeHash(const GraphDef& graph_def) {
  std::string graph_def_string;
  SerializeToStringDeterministic(graph_def, &graph_def_string);
  return tensorflow::Fingerprint64(graph_def_string);
}

FingerprintDef CreateFingerprintDef(const MetaGraphDef& metagraph) {
  // Create a copy of `metagraph` which will be used and mutated for fingerprint
  // computation.
  MetaGraphDef metagraph_copy = metagraph;
  FingerprintDef fingerprint_def;
  // Set fingerprint field #1.
  fingerprint_def.set_graph_def_checksum(
      ComputeHash(metagraph_copy.graph_def()));
  // Set fingerprint field #2.
  CanonicalizeGraphDef(*metagraph_copy.mutable_graph_def());
  fingerprint_def.set_graph_def_program_hash(
      ComputeHash(metagraph_copy.graph_def()));
  // Set fingerprint field #3.
  fingerprint_def.set_signature_def_hash(
      RegularizeAndHashSignatureDefs(metagraph_copy.signature_def()));
  // Set fingerprint field #4.
  StatusOr<uint64> object_graph_hash =
      RegularizeAndHashSavedObjectGraph(metagraph_copy.object_graph_def());
  fingerprint_def.set_saved_object_graph_hash(
      RegularizeAndHashSavedObjectGraph(metagraph_copy.object_graph_def()));
  return fingerprint_def;
}

// The GraphDef contains two main sections: a list of nodes and the
// FunctionDefLibrary. Canonicalization treats these two sections separately.
void CanonicalizeGraphDef(GraphDef& graph_def) {
  CanonicalizeNodes(&graph_def);
  // TODO(b/240173815): Complete canonicalization of the FunctionDefLibrary.
  // For now, we just clear the FunctionDefLibrary.
  graph_def.mutable_library()->Clear();
  graph_def.mutable_versions()->Clear();
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
uint64 RegularizeAndHashSavedObjectGraph(
    const SavedObjectGraph& object_graph_def) {
  // Sort `concrete_functions`, which is an unordered map from function names to
  // SavedConcreteFunction, using the suffix UID of the function name. Assumes
  // that the trackable children are listed in a deterministic order during
  // serialization.
  absl::btree_map<int, std::string> uid_to_function_names;
  for (const auto& [name, concrete_function] :
       object_graph_def.concrete_functions()) {
    StatusOr<int> uid = GetSuffixUID(name);
    // All valid function names should end in an UID.
    if (uid.ok()) {
      uid_to_function_names.insert({*uid, name});
    } else {
      LOG(ERROR) << uid.status().error_message();
    }
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
}  // namespace tensorflow::fingerprinting
