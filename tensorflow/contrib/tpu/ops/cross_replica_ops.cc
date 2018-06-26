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
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("group_assignment: list(int) = []")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
An Op to sum inputs across replicated TPU instances. Each
instance supplies its own input. If group_assignment is empty, the output of
each is the sum of all the inputs, otherwise the output of each is the sum of
the inputs belonging to the same group.

For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
group_assignment=`[0,1,0,1]` sets `A, C` as group 0, and `B, D` as group 1.
Thus we get the outputs: `[A+C, B+D, A+C, B+D]`.

input: The local input to the sum.
output: The sum of all the distributed inputs.
T: The type of elements to be summed.
group_assignment: The list of group ids. `group_assignment[i]` represents the
  group id of replica i.
)doc");

}  // namespace tensorflow
