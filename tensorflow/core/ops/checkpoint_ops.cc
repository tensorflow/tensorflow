/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("GenerateVocabRemapping")
    .Input("new_vocab_file: string")
    .Input("old_vocab_file: string")
    .Attr("new_vocab_offset: int >= 0")
    .Attr("num_new_vocab: int >= 0")
    .Attr("old_vocab_size: int >= -1 = -1")
    .Output("remapping: int64")
    .Output("num_present: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      int64_t new_vocab_offset;
      TF_RETURN_IF_ERROR(c->GetAttr("new_vocab_offset", &new_vocab_offset));
      int64_t num_new_vocab;
      TF_RETURN_IF_ERROR(c->GetAttr("num_new_vocab", &num_new_vocab));

      c->set_output(0, c->Vector(num_new_vocab));
      c->set_output(1, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("LoadAndRemapMatrix")
    .Input("ckpt_path: string")
    .Input("old_tensor_name: string")
    .Input("row_remapping: int64")
    .Input("col_remapping: int64")
    .Input("initializing_values: float")
    .Attr("num_rows: int >= 0")
    .Attr("num_cols: int >= 1")
    .Attr("max_rows_in_memory: int = -1")
    .Output("output_matrix: float")
    // TODO(b/30502450): Setting the op as being stateful prevents it from being
    // executed more often than expected (possibly due to stateful ops not being
    // subject to constant folding?). This op is usually slow and may require
    // multiple disk reads, so we want to minimize the number of times it's
    // executed redundantly.
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      int64_t num_rows;
      TF_RETURN_IF_ERROR(c->GetAttr("num_rows", &num_rows));
      int64_t num_cols;
      TF_RETURN_IF_ERROR(c->GetAttr("num_cols", &num_cols));

      c->set_output(0, c->Matrix(num_rows, num_cols));
      return OkStatus();
    });
}  // namespace tensorflow
