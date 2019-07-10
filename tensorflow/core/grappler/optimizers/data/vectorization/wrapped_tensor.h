/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_WRAPPED_TENSOR_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_WRAPPED_TENSOR_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace grappler {

// Represents a tensor that has been vectorized.
struct WrappedTensor {
  Node* const node;
  const int output_index;

  // Whether the tensor is stacked, i.e. represents the results of applying
  // the operation on all slices of the input, where each row i of the
  // tensor corresponds to the op's output on slice i of the input. False
  // if the tensor is not stacked, i.e. represents the result of the op on
  // a single slice of the input, where the result does not vary between
  // slices.
  bool stacked;

  WrappedTensor(Node* node, int output_index, bool stacked)
      : node(node), output_index(output_index), stacked(stacked) {}
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_WRAPPED_TENSOR_H_
