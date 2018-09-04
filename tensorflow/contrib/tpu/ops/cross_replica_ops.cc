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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("CrossReplicaSum")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
An Op to sum inputs across replicated TPU instances. Each
instance supplies its own input. If group_assignment is empty, the output of
each is the sum of all the inputs, otherwise the output of each is the sum of
the inputs belonging to the same group.

For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
and `B, D, F, H` as group 1. Thus we get the outputs:
`[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.

input: The local input to the sum.
group_assignment: An int32 tensor with shape
  [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
  replica ids in the ith subgroup.
output: The sum of all the distributed inputs.
T: The type of elements to be summed.
)doc");

}  // namespace tensorflow
