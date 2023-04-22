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

#ifndef TENSORFLOW_CORE_GRAPPLER_VERIFIERS_GRAPH_VERIFIER_H_
#define TENSORFLOW_CORE_GRAPPLER_VERIFIERS_GRAPH_VERIFIER_H_

#include <string>
#include <vector>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// An abstract interface for verifying a graph.
// This will be used to implement specific verifiers to verify that a grappler
// transformed graph is valid.
// Some examples of specific verifiers are:
// 1. A general structural verifier that verifies that the specified graph has
//    a valid structure that meets the specification of what it means to be
//      a valid TensorFlow graph.
// 2. A backend specific verifier that verifies that the specified graph,
//     generated after a grappler transformation to convert the input TensorFlow
//     graph to a corresponding backend graph, is a valid graph in the
//     specification of the backend.
class GraphVerifier {
 public:
  GraphVerifier() {}
  virtual ~GraphVerifier() {}

  // A name for the verifier.
  virtual string name() const = 0;

  // Implement an algorithm to verify the specified graph.
  // The return value is a Status that represents a concatenation of Status of
  // each verification step.
  virtual Status Verify(const GraphDef& graph) = 0;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_VERIFIERS_GRAPH_VERIFIER_H_
