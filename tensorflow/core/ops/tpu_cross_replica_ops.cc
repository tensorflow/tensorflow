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
    .Attr("T: {numbertype, bool}")
    .Attr("concat_dimension: int")
    .Attr("split_dimension: int")
    .Attr("split_count: int")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input = c->input(0);
      ShapeHandle group_assignment = c->input(1);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64_t rank = c->Rank(input);
      int concat_dimension;
      int split_dimension;
      int split_count;
      TF_RETURN_IF_ERROR(c->GetAttr("split_count", &split_count));
      if (split_count < 1) {
        return errors::InvalidArgument("split_count ", split_count,
                                       " must at least be one.");
      }
      if (c->RankKnown(group_assignment) && c->Rank(group_assignment) != 2) {
        return errors::InvalidArgument("group_assignment must have rank 2.");
      }
      DimensionHandle num_replicas_per_group = c->Dim(group_assignment, 1);
      if (c->ValueKnown(num_replicas_per_group) &&
          (c->Value(num_replicas_per_group) != split_count)) {
        return errors::InvalidArgument(
            "split_count ", split_count,
            " must equal the size of the second dimension of group_assignment ",
            c->Value(num_replicas_per_group));
      }

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

      if (!c->ValueKnown(c->Dim(input, concat_dimension)) ||
          !c->ValueKnown(c->Dim(input, split_dimension))) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      std::vector<DimensionHandle> dims;
      dims.resize(rank);

      for (int32_t i = 0; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
        if (i == concat_dimension) {
          dims[i] = c->MakeDim(c->Value(dims[i]) * split_count);
        }
        if (i == split_dimension) {
          if (c->ValueKnown(dims[i]) &&
              (c->Value(dims[i]) % split_count != 0)) {
            return errors::InvalidArgument(
                "input dimension ", c->Value(dims[i]),
                " not divisible by split_count ", split_count);
          }
          dims[i] = c->MakeDim(c->Value(dims[i]) / split_count);
        }
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    });

REGISTER_OP("CrossReplicaSum")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, float64, int32, uint32}")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CollectivePermute")
    .Input("input: T")
    .Input("source_target_pairs: int32")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);
}  // namespace tensorflow
