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

#include <initializer_list>
#include <memory>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/wrapped_tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kTransposePrefix[] = "vectorized/transpose";

class TransposeVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    Status status;
    Scope parent = NewInternalScope(outer_scope, &status, /*refiner=*/nullptr);
    Scope scope = parent.NewSubScope(kTransposePrefix);

    Output tensor, original_perm;
    TF_RETURN_IF_ERROR(inputs.stacked(0, &tensor));
    TF_RETURN_IF_ERROR(inputs.unstacked(1, &original_perm));
    if (original_perm.type() != DT_INT32) {
      original_perm = ops::Cast(scope, original_perm, DT_INT32);
    }

    // The vectorized permutation is the original permutation with an additional
    // leading 0 and all other values incremented by 1.
    // perm = tf.concat([[0], original_perm + 1], axis=0)
    Output perm =
        ops::Concat(scope,
                    std::initializer_list<Output>(
                        {ops::Const(scope, {0}),
                         ops::Add(scope, original_perm, ops::Const(scope, 1))}),
                    ops::Const(scope, 0));

    Output vectorized_transpose = ops::Transpose(scope, tensor, perm);

    TF_RETURN_IF_ERROR(status);

    // Add output mappings.
    outputs->push_back({vectorized_transpose.node(), 0, true});
    return Status::OK();
  }
};

REGISTER_VECTORIZER("Transpose", TransposeVectorizer);

}  // namespace

}  // namespace grappler
}  // namespace tensorflow
