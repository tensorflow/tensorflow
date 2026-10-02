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

#include "tensorflow/core/protobuf/fingerprint.pb.h"

namespace tensorflow::saved_model::fingerprinting {

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

}  // namespace tensorflow::saved_model::fingerprinting
