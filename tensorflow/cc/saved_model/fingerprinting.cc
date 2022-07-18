/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow::fingerprinting {

uint64 ComputeHash(const GraphDef& graph_def) {
  std::string graph_def_string;
  SerializeToStringDeterministic(graph_def, &graph_def_string);
  return tensorflow::Fingerprint64(graph_def_string);
}

FingerprintDef CreateFingerprintDef(const MetaGraphDef& metagraph) {
  FingerprintDef fingerprint_def;
  fingerprint_def.set_graph_def_hash(ComputeHash(metagraph.graph_def()));
  return fingerprint_def;
}

}  // namespace tensorflow::fingerprinting
