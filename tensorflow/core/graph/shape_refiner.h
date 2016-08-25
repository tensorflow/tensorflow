/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_GRAPH_SHAPE_REFINER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_GRAPH_SHAPE_REFINER_H_

#include <vector>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// ShapeRefiner performs shape inference for TensorFlow Graphs.  It is
// responsible for instantiating InferenceContext objects for each
// Node in the Graph, and providing/storing the 'input_tensor' Tensors
// used by Shape Inference functions, when available at graph
// construction time.
class ShapeRefiner {
 public:
  ShapeRefiner();
  ~ShapeRefiner();

  // Performs validation of 'node' and runs 'node's shape function,
  // storing its shape outputs.
  //
  // All inputs of 'node' must be added to ShapeRefiner prior to
  // adding 'node'.
  //
  // Returns an error if:
  //  - the shape function for 'node' was not registered.
  //  - 'node' was added before its inputs.
  //  - The shape inference function returns an error.
  Status AddNode(const Node* node);

  // Returns the InferenceContext for 'node', if present.
  shape_inference::InferenceContext* GetContext(const Node* node) const {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  // Extracts the 'constant_value' of 'input_node' if possible.  Uses
  // 'tensor_storage' for storage and sets '*input_tensor' to
  // 'tensor_storage' if a constant value could be extracted.
  Status ConstantValue(const Node* node, Tensor* tensor_storage,
                       const Tensor** input_tensor) const;

  // Helper functions to extract the Tensor associated with 'node'.
  Status Constant(const Node* node, Tensor* tensor_storage,
                  const Tensor** input_tensor) const;
  Status Shape(const Node* node, Tensor* tensor_storage,
               const Tensor** input_tensor) const;
  Status Size(const Node* node, Tensor* tensor_storage,
              const Tensor** input_tensor) const;
  Status Rank(const Node* node, Tensor* tensor_storage,
              const Tensor** input_tensor) const;
  // Stores a map from a node to its InferenceContext.
  //
  // Owns values.
  std::unordered_map<const Node*, shape_inference::InferenceContext*>
      node_to_context_;

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPH_SHAPE_REFINER_H_
