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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_H_

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/wrapped_tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// Interface for vectorization of TensorFlow operations. See `CastVectorizer`
// for an example.
class Vectorizer {
 public:
  virtual ~Vectorizer() {}

  // Vectorizes an operation, `node`, by adding Node(s) to `outer_scope`
  // that produce the same vector output(s) as executing `node`'s op
  // on elements of `inputs`. The new Node(s) collectively have the
  // same number of input and output ports as the node being converted.
  // Adds edges between the newly created nodes and nodes in `inputs`, and adds
  // mappings to the new nodes' output ports to `outputs`, where the i'th
  // value in `outputs` corresponds to the i'th output port of the node
  // to be converted.
  virtual Status Vectorize(const Node& node, Graph* outer_scope,
                           std::vector<WrappedTensor>&& inputs,
                           std::vector<WrappedTensor>* outputs) = 0;
};

}  // namespace grappler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_H_
