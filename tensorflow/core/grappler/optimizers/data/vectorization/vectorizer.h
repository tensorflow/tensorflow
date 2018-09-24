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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

// Interface for vectorization of TensorFlow operations. See `CastVectorizer`
// for an example.
class Vectorizer {
 public:
  virtual ~Vectorizer() {}

  // Vectorizes an operation, `node`, by adding operation(s) to `outer_scope`
  // that produce the same vector output(s) as executing `node`'s op
  // on elements of the vector inputs, and adding mappings to `conversion_map`
  // from old output tensor names to new (vectorized) output tensor names.
  // The new node(s) collectively have the same number of inputs and outputs as
  // the node being converted, and use the tensor names in `inputs` as their
  // inputs.
  virtual Status Vectorize(const NodeDef& node, gtl::ArraySlice<string> inputs,
                           FunctionDef* outer_scope,
                           std::map<string, string>* conversion_map) = 0;
};

}  // namespace vectorization_utils
}  // namespace grappler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_H_
