/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// Note that the following operator is just a placeholder and has no
// associated kernel. The code in accumulate_n_optimizer.cc replaces
// this placeholder with a graph of operators that do have kernels.
REGISTER_OP("AccumulateNV2")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .Attr("shape: shape")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Returns the element-wise sum of a list of tensors.

`tf.accumulate_n` performs the same operation as `tf.add_n`, but does not wait
for all of its inputs to be ready before beginning to sum. This can save memory
if inputs are ready at different times, since minimum temporary storage is
proportional to the output size rather than the inputs size.

Returns a `Tensor` of same shape and type as the elements of `inputs`.

inputs: A list of `Tensor` objects, each with same shape and type.
shape: Shape of elements of `inputs`.
)doc");

}  // namespace tensorflow
