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
REGISTER_OP("LoadAndRemapMatrix")
    .Input("ckpt_path: string")
    .Input("old_tensor_name: string")
    .Input("row_remapping: int64")
    .Input("col_remapping: int64")
    .Input("initializing_values: float")
    .Attr("num_rows: int >= 0")
    .Attr("num_cols: int >= 1")
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

      int64 num_rows;
      TF_RETURN_IF_ERROR(c->GetAttr("num_rows", &num_rows));
      int64 num_cols;
      TF_RETURN_IF_ERROR(c->GetAttr("num_cols", &num_cols));

      c->set_output(0, c->Matrix(num_rows, num_cols));
      return Status::OK();
    })
    .Doc(R"doc(
Loads a 2-D (matrix) `Tensor` with name `old_tensor_name` from the checkpoint
at `ckpt_path` and potentially reorders its rows and columns using the
specified remappings.

Most users should use one of the wrapper initializers (such as
`tf.contrib.framework.load_and_remap_matrix_initializer`) instead of this
function directly.

The remappings are 1-D tensors with the following properties:

* `row_remapping` must have exactly `num_rows` entries. Row `i` of the output
  matrix will be initialized from the row corresponding to index
  `row_remapping[i]` in the old `Tensor` from the checkpoint.
* `col_remapping` must have either 0 entries (indicating that no column
  reordering is needed) or `num_cols` entries. If specified, column `j` of the
  output matrix will be initialized from the column corresponding to index
  `col_remapping[j]` in the old `Tensor` from the checkpoint.
* A value of -1 in either of the remappings signifies a "missing" entry. In that
  case, values from the `initializing_values` tensor will be used to fill that
  missing row or column. If `row_remapping` has `r` missing entries and
  `col_remapping` has `c` missing entries, then the following condition must be
  true:

`(r * num_cols) + (c * num_rows) - (r * c) == len(initializing_values)`

The remapping tensors can be generated using the GenerateVocabRemapping op.

As an example, with row_remapping = [1, 0, -1], col_remapping = [0, 2, -1],
initializing_values = [0.5, -0.5, 0.25, -0.25, 42], and w(i, j) representing
the value from row i, column j of the old tensor in the checkpoint, the output
matrix will look like the following:

[[w(1, 0),  w(1, 2),  0.5],
 [w(0, 0),  w(0, 2), -0.5],
 [0.25,    -0.25,      42]]

ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`) from
  which the old matrix `Tensor` will be loaded.
old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
row_remapping: An int `Tensor` of row remappings (generally created by
  `generate_vocab_remapping`).  Even if no row remapping is needed, this must
  still be an index-valued Tensor (e.g. [0, 1, 2, ...]), or a shifted
  index-valued `Tensor` (e.g. [8, 9, 10, ...], for partitioned `Variables`).
col_remapping: An int `Tensor` of column remappings (generally created by
  `generate_vocab_remapping`).  May be a size-0 `Tensor` if only row remapping
  is to be done (e.g. column ordering is the same).
initializing_values: A float `Tensor` containing  values to fill in for cells
  in the output matrix that are not loaded from the checkpoint. Length must be
  exactly the same as the number of missing / new cells.
num_rows: Number of rows (length of the 1st dimension) in the output matrix.
num_cols: Number of columns (length of the 2nd dimension) in the output matrix.
output_matrix: Output matrix containing existing values loaded from the
  checkpoint, and with any missing values filled in from initializing_values.
)doc");
}  // namespace tensorflow
