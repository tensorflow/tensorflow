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

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {

namespace {

const char* const kReshapePrefix = "vectorized/reshape";

// The vectorized shape should be the original shape with an additional leading
// dimension that is the same as the leading dimension of the stacked
// input tensor.
Output GetVectorizedShape(Scope* s, Output tensor, Output original_shape) {
  Output const_vec_1 = ops::Const(*s, {1});
  Output shape = ops::Shape(*s, tensor);

  // shape[:1]
  Output dim_0 =
      ops::StridedSlice(*s, shape, const_vec_1, const_vec_1, const_vec_1,
                        ops::StridedSlice::Attrs().BeginMask(1));

  // tf.concat([dim_0, original], 0)
  return ops::Concat(*s, {dim_0, original_shape}, ops::Const(*s, 0));
}

class ReshapeVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   std::vector<WrappedTensor>&& inputs,
                   std::vector<WrappedTensor>* outputs) override {
    if (!inputs[0].stacked || inputs[1].stacked) {
      return errors::InvalidArgument(
          "Expecting input 0 (`tensor`) to be stacked and input 1 (`shape`) to "
          "be unstacked.");
    }

    Status status;
    Scope parent = NewInternalScope(outer_scope, &status, nullptr);
    Scope s = parent.NewSubScope(kReshapePrefix);

    Output tensor = {inputs[0].node, inputs[0].output_index};
    Output vectorized_reshape =
        ops::Reshape(s, tensor,
                     GetVectorizedShape(
                         &s, tensor, {inputs[1].node, inputs[1].output_index}));

    TF_RETURN_IF_ERROR(status);

    // Add output mappings
    outputs->push_back({vectorized_reshape.node(), 0, true});
    return Status::OK();
  }
};

REGISTER_VECTORIZER("Reshape", ReshapeVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
