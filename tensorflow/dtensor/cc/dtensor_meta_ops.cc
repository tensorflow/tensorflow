/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

REGISTER_OP("DTensorAllReduce")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr(
        "T: {half, bfloat16, float, float64, int8, uint8, int32, uint32, "
        "int64, uint64, bool}")
    .Attr("reduce_op: {'Min', 'Max', 'Mul', 'Add', 'Mean', 'Any', 'All'}")
    .Attr("device_type: string")  // e.g. "/device:TPU"
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DTensorReduceScatter")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Input("scatter_dimension: int32")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, int32, uint32, int64, bool}")
    .Attr("reduce_op: {'Min', 'Max', 'Mul', 'Add', 'Mean', 'Any', 'All'}")
    .Attr("device_type: string")  // e.g. "/device:TPU"
    .SetShapeFn(shape_inference::ReduceScatterShape);

REGISTER_OP("DTensorAllScatter")
    .Input("input: T")
    .Output("output: T")
    .Attr(
        "T: {half, bfloat16, float, float64, int8, uint8, int32, uint32, "
        "int64, uint64, bool, string}")
    .Attr("input_layout: string")
    .Attr("output_layout: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      shape_inference::ShapeHandle in = c->input(0);
      if (!c->RankKnown(in)) {
        // Input shape unknown, so set unknown output shape.
        c->set_output(0, in);
        return absl::OkStatus();
      }

      std::string input_layout_string;
      std::string output_layout_string;
      TF_RETURN_IF_ERROR(c->GetAttr("input_layout", &input_layout_string));
      TF_RETURN_IF_ERROR(c->GetAttr("output_layout", &output_layout_string));
      TF_ASSIGN_OR_RETURN(Layout input_layout,
                          Layout::FromString(input_layout_string));
      TF_ASSIGN_OR_RETURN(Layout output_layout,
                          Layout::FromString(output_layout_string));
      if (c->Rank(in) != input_layout.rank() ||
          c->Rank(in) != output_layout.rank()) {
        return errors::InvalidArgument(
            "Input tensor rank and layout ranks do not agree: input rank ",
            c->Rank(in), " input layout rank ", input_layout.rank(),
            " output "
            "layout rank ",
            output_layout.rank());
      }
      const std::vector<int32> output_sharding = output_layout.num_shards();
      std::vector<shape_inference::DimensionHandle> out_dims;
      out_dims.reserve(c->Rank(in));
      for (int i = 0; i < c->Rank(in); ++i) {
        shape_inference::DimensionHandle dim = c->Dim(in, i);
        if (!c->ValueKnown(dim) ||
            input_layout.sharding_spec(i) == output_layout.sharding_spec(i)) {
          out_dims.emplace_back(dim);
        } else if (Layout::IsUnshardedDimension(
                       input_layout.sharding_spec(i))) {
          shape_inference::DimensionHandle out_dim;
          TF_RETURN_IF_ERROR(c->Divide(dim, output_sharding[i],
                                       /*evenly_divisible=*/true, &out_dim));
          out_dims.push_back(out_dim);
        } else {
          return errors::InvalidArgument(
              "DTensorAllScatter only supports output layouts which are more "
              "sharded than input layouts. Received input sharding spec ",
              input_layout.sharding_spec(i), " and output sharding spec ",
              output_layout.sharding_spec(i), " for dimension ", i, ".");
        }
      }
      c->set_output(0, c->MakeShape(out_dims));
      return absl::OkStatus();
    });

REGISTER_OP("DTensorAllGather")
    .Input("input: T")
    .Output("output: T")
    .Attr(
        "T: {half, bfloat16, float, float64, int8, uint8, int32, uint32, "
        "int64, uint64, "
        "bool}")
    .Attr("input_layout: string")
    .Attr("output_layout: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      shape_inference::ShapeHandle in = c->input(0);
      if (!c->RankKnown(in)) {
        // Input shape unknown, so set unknown output shape.
        c->set_output(0, in);
        return absl::OkStatus();
      }

      std::string input_layout_string;
      std::string output_layout_string;
      TF_RETURN_IF_ERROR(c->GetAttr("input_layout", &input_layout_string));
      TF_RETURN_IF_ERROR(c->GetAttr("output_layout", &output_layout_string));
      TF_ASSIGN_OR_RETURN(Layout input_layout,
                          Layout::FromString(input_layout_string));
      TF_ASSIGN_OR_RETURN(Layout output_layout,
                          Layout::FromString(output_layout_string));
      if (c->Rank(in) != input_layout.rank() ||
          c->Rank(in) != output_layout.rank()) {
        return errors::InvalidArgument(
            "Input tensor rank and layout ranks do not agree: input rank ",
            c->Rank(in), " input layout rank ", input_layout.rank(),
            " output "
            "layout rank ",
            output_layout.rank());
      }
      const std::vector<int32> input_sharding = input_layout.num_shards();
      std::vector<shape_inference::DimensionHandle> out_dims;
      out_dims.reserve(c->Rank(in));
      for (int32 i = 0; i < c->Rank(in); ++i) {
        shape_inference::DimensionHandle dim = c->Dim(in, i);
        if (!c->ValueKnown(dim) ||
            input_layout.sharding_spec(i) == output_layout.sharding_spec(i)) {
          out_dims.emplace_back(dim);
        } else if (Layout::IsUnshardedDimension(
                       output_layout.sharding_spec(i))) {
          shape_inference::DimensionHandle out_dim;
          TF_RETURN_IF_ERROR(c->Multiply(dim, input_sharding[i], &out_dim));
          out_dims.push_back(out_dim);
        } else {
          return errors::InvalidArgument(
              "DTensorAllGatherr only supports input layouts which are more "
              "sharded than output layouts. Received input sharding spec ",
              input_layout.sharding_spec(i), " and output sharding spec ",
              output_layout.sharding_spec(i), " for dimension ", i, ".");
        }
      }
      c->set_output(0, c->MakeShape(out_dims));
      return absl::OkStatus();
    });

REGISTER_OP("DTensorAllToAll")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, float64, int32, uint32, int64, bool}")
    .Attr("input_layout: string")
    .Attr("output_layout: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      shape_inference::ShapeHandle in = c->input(0);
      if (!c->RankKnown(in)) {
        // Input shape unknown, so set unknown output shape.
        c->set_output(0, in);
        return absl::OkStatus();
      }

      std::string input_layout_string;
      std::string output_layout_string;
      TF_RETURN_IF_ERROR(c->GetAttr("input_layout", &input_layout_string));
      TF_RETURN_IF_ERROR(c->GetAttr("output_layout", &output_layout_string));
      TF_ASSIGN_OR_RETURN(Layout input_layout,
                          Layout::FromString(input_layout_string));
      TF_ASSIGN_OR_RETURN(Layout output_layout,
                          Layout::FromString(output_layout_string));
      if (c->Rank(in) != input_layout.rank() ||
          c->Rank(in) != output_layout.rank()) {
        return errors::InvalidArgument(
            "Input tensor rank and layout ranks do not agree: input rank ",
            c->Rank(in), " input layout rank ", input_layout.rank(),
            " output "
            "layout rank ",
            output_layout.rank());
      }
      const std::vector<int32> input_sharding = input_layout.num_shards();
      const std::vector<int32> output_sharding = output_layout.num_shards();
      std::vector<shape_inference::DimensionHandle> out_dims;
      out_dims.reserve(c->Rank(in));
      for (int i = 0; i < c->Rank(in); ++i) {
        shape_inference::DimensionHandle dim = c->Dim(in, i);
        if (!c->ValueKnown(dim) ||
            input_layout.sharding_spec(i) == output_layout.sharding_spec(i)) {
          out_dims.emplace_back(dim);
        } else if (Layout::IsUnshardedDimension(
                       input_layout.sharding_spec(i)) &&
                   Layout::IsShardedDimension(output_layout.sharding_spec(i))) {
          shape_inference::DimensionHandle out_dim;
          TF_RETURN_IF_ERROR(c->Divide(dim, output_sharding[i],
                                       /*evenly_divisible=*/true, &out_dim));
          out_dims.push_back(out_dim);
        } else if (Layout::IsShardedDimension(input_layout.sharding_spec(i)) &&
                   Layout::IsUnshardedDimension(
                       output_layout.sharding_spec(i))) {
          shape_inference::DimensionHandle out_dim;
          TF_RETURN_IF_ERROR(c->Multiply(dim, input_sharding[i], &out_dim));
          out_dims.push_back(out_dim);
        }
      }
      c->set_output(0, c->MakeShape(out_dims));
      return absl::OkStatus();
    });

}  // namespace dtensor
}  // namespace tensorflow
