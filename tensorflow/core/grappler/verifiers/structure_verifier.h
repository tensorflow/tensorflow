/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_VERIFIERS_STRUCTURE_VERIFIER_H_
#define TENSORFLOW_CORE_GRAPPLER_VERIFIERS_STRUCTURE_VERIFIER_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/verifiers/graph_verifier.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// Verifies the structure of a graph to ensure it is valid.
class StructureVerifier : public GraphVerifier {
 public:
  StructureVerifier() {}
  ~StructureVerifier() override {}

  string name() const override { return "structure_verifier"; };

  absl::Status Verify(const GraphDef& graph) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_VERIFIERS_STRUCTURE_VERIFIER_H_
