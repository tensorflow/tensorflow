// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("ScatterAddNdim")
    .Input("input: Ref(float)")
    .Input("indices: int32")
    .Input("deltas: float")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Doc(R"doc(
  Add elements in deltas to mutable input according to indices.

  input: A N-dimensional float tensor to mutate.
  indices:= A 2-D int32 tensor. The size of dimension 0 is the number of
    deltas, the size of dimension 1 is the rank of the input.  `indices[i]`
    gives the coordinates of input that `deltas[i]` should add to.  If
    `indices[i]` does not fully specify a location (it has less indices than
    there are dimensions in `input`), it is assumed that they are start
    indices and that deltas contains enough values to fill in the remaining
    input dimensions.
  deltas: `deltas[i]` is the value to add to input at index indices[i][:]
)doc");

REGISTER_OP("ReinterpretStringToFloat")
    .Input("input_data: string")
    .Output("output_data: float")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
   Converts byte arrays represented by strings to 32-bit
   floating point numbers. The output numbers themselves are meaningless, and
   should only be used in == comparisons.

   input_data: A batch of string features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
   output_data: A tensor of the same shape as input_data but the values are
     float32.

)doc");

}  // namespace tensorflow
