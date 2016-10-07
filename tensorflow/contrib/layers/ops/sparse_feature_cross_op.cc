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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("SparseFeatureCross")
    .Input("indices: N * int64")
    .Input("values: sparse_types")
    .Input("shapes: N * int64")
    .Input("dense: dense_types")
    .Output("output_indices: int64")
    .Output("output_values: out_type")
    .Output("output_shape: int64")
    .Attr("N: int >= 0")
    .Attr("hashed_output: bool")
    .Attr("num_buckets: int >= 0")
    .Attr("sparse_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .Attr("out_type: {int64, string}")
    .Attr("internal_type: {int64, string}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
Generates sparse cross form a list of sparse tensors.

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
representing features of one feature column. It outputs a 2D `SparseTensor` with
the batchwise crosses of these features.

For example, if the inputs are

    inputs[0]: SparseTensor with shape = [2, 2]
    [0, 0]: "a"
    [1, 0]: "b"
    [1, 1]: "c"

    inputs[1]: SparseTensor with shape = [2, 1]
    [0, 0]: "d"
    [1, 0]: "e"

    inputs[2]: Tensor [["f"], ["g"]]

then the output will be

    shape = [2, 2]
    [0, 0]: "a_X_d_X_f"
    [1, 0]: "b_X_e_X_g"
    [1, 1]: "c_X_e_X_g"

if hashed_output=true then the output will be

    shape = [2, 2]
    [0, 0]: Hash64("f", Hash64("d", Hash64("a")))
    [1, 0]: Hash64("g", Hash64("e", Hash64("b")))
    [1, 1]: Hash64("g", Hash64("e", Hash64("c")))

indices: 2-D.  Indices of each input `SparseTensor`.
values: 1-D.   values of each `SparseTensor`.
shapes: 1-D.   Shapes of each `SparseTensor`.
dense: 2-D.    Columns represented by dense `Tensor`.
output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
output_values: 1-D.  Non-empty values of the concatenated or hashed
  `SparseTensor`.
output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
)doc");
}  // namespace tensorflow
