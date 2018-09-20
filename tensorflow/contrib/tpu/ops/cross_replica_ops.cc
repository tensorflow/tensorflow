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
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("AllToAll")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("concat_dimension: int")
    .Attr("split_dimension: int")
    .Attr("split_count: int")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input = c->input(0);
      int64 rank;
      if (c->RankKnown(input)) {
        rank = c->Rank(input);
      } else {
        return errors::InvalidArgument("input's rank is unknown.");
      }
      int concat_dimension;
      int split_dimension;

      TF_RETURN_IF_ERROR(c->GetAttr("concat_dimension", &concat_dimension));

      if (concat_dimension < 0 || concat_dimension >= rank) {
        return errors::InvalidArgument("concat_dimension ", concat_dimension,
                                       " is out of range of input rank ", rank);
      }

      TF_RETURN_IF_ERROR(c->GetAttr("split_dimension", &split_dimension));
      if (split_dimension < 0 || split_dimension >= rank) {
        return errors::InvalidArgument("split_dimension ", split_dimension,
                                       " is out of range of input rank ", rank);
      }

      std::vector<DimensionHandle> dims;
      dims.resize(rank);

      for (int32 i = 0; i < rank; ++i) {
        int64 in_idx = i;
        if (i == concat_dimension) {
          in_idx = split_dimension;
        } else if (i == split_dimension) {
          in_idx = concat_dimension;
        }

        dims[i] = c->Dim(input, in_idx);
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
An Op to exchange data across TPU replicas. On each replica, the input is
split into `split_count` blocks along `split_dimension` and send to the other
replicas given group_assignment. After receiving `split_count` - 1 blocks from
other replicas, we concatenate the blocks along `concat_dimension` as the
output.

For example, suppose there are 2 TPU replicas:
replica 0 receives input: `[[A, B]]`
replica 1 receives input: `[[C, D]]`

group_assignment=`[[0, 1]]`
concat_dimension=0
split_dimension=1
split_count=2

replica 0's output: `[[A], [C]]`
replica 1's output: `[[B], [D]]`

input: The local input to the sum.
group_assignment: An int32 tensor with shape
  [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
  replica ids in the ith subgroup.
concat_dimension: The dimension number to concatenate.
split_dimension: The dimension number to split.
split_count: The number of splits, this number must equal to the sub-group
  size(group_assignment.get_shape()[1])
output: The exchanged result.
T: The type of elements to be exchanged.
)doc");

REGISTER_OP("CrossReplicaSum")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
An Op to sum inputs across replicated TPU instances. Each instance supplies its
own input.

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
