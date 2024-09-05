/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

ShapeHandle _UpdatePartitionDim(InferenceContext* c, const ShapeHandle handle,
                                const int partition_dim) {
  ShapeHandle newoutput0;
  shape_inference::DimensionHandle new_dim;
  TF_CHECK_OK(
      c->Multiply(c->Dim(handle, partition_dim), c->num_inputs(), &new_dim));
  TF_CHECK_OK(c->ReplaceDim(handle, partition_dim, new_dim, &newoutput0));
  return newoutput0;
}

absl::StatusOr<ShapeHandle> _ComputeOutputShape(
    InferenceContext* c, const ShapeHandle handle,
    const std::vector<int>& partition_dims) {
  int rank = InferenceContext::Rank(handle);
  if (partition_dims.empty()) {
    return handle;  // no partitioning; input and output shapes same
  } else if (rank > partition_dims.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Need at least ", rank, " partition dimensions."));
  }

  ShapeHandle previous = handle;
  for (int i = 0; i < rank; ++i) {
    shape_inference::DimensionHandle dim;
    TF_RETURN_IF_ERROR(
        c->Multiply(c->Dim(previous, i), partition_dims[i], &dim));
    TF_RETURN_IF_ERROR(c->ReplaceDim(previous, i, dim, &previous));
  }

  return previous;
}

REGISTER_OP("TPUPartitionedInput")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .Attr("partition_dim: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      int partition_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("partition_dim", &partition_dim));

      if (c->num_inputs() == 0) {
        return absl::InvalidArgumentError(
            "Expected at least one input to TPUPartitionedInput.");
      }

      ShapeHandle cur = c->input(c->num_inputs() - 1);
      int rank = InferenceContext::kUnknownRank;
      if (dtype == DT_RESOURCE) {
        auto* shapes_and_types =
            c->input_handle_shapes_and_types(c->num_inputs() - 1);
        if (shapes_and_types) {
          ShapeHandle shape_handle = shapes_and_types->at(0).shape;
          rank = InferenceContext::Rank(shape_handle);
        }
      } else {
        rank = InferenceContext::Rank(cur);
      }

      // limitation: can only validate rank when it is known
      if ((rank != InferenceContext::kUnknownRank && partition_dim >= rank) ||
          (partition_dim < -1))
        return absl::InvalidArgumentError(
            absl::StrCat("Cannot partition dim ", partition_dim, " of rank ",
                         rank, " tensor."));

      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      if (partition_dim == -1 || dtype == DT_RESOURCE) {
        c->set_output(0, cur);
      } else {
        ShapeHandle newoutput0 = _UpdatePartitionDim(c, cur, partition_dim);
        c->set_output(0, newoutput0);
      }

      // If this is a resource, unify the resource shapes.
      if (dtype == DT_RESOURCE) {
        ShapeHandle previous_shape_handle;
        const std::vector<shape_inference::ShapeAndType>* shapes_and_types =
            nullptr;
        for (int i = c->num_inputs() - 1; i >= 0; --i) {
          shapes_and_types = c->input_handle_shapes_and_types(i);
          if (shapes_and_types) {
            ShapeHandle shape_handle = shapes_and_types->at(0).shape;
            if (!c->FullyDefined(shape_handle)) {
              return absl::InvalidArgumentError(
                  absl::StrCat("Inputs must have static shape,", "input[", i,
                               "] has unknown dimension."));
            }
            if (i != c->num_inputs() - 1) {
              ShapeHandle tmp;
              if (!c->Merge(shape_handle, previous_shape_handle, &tmp).ok()) {
                return absl::InvalidArgumentError(
                    "Inputs must have the same shape.");
              }
            } else {
              previous_shape_handle = shape_handle;
            }
          }
        }
        if (shapes_and_types) {
          if (partition_dim == -1) {
            c->set_output_handle_shapes_and_types(0, *shapes_and_types);
          } else {
            ShapeHandle newoutput0 =
                _UpdatePartitionDim(c, previous_shape_handle, partition_dim);

            std::vector<shape_inference::ShapeAndType> output_shapes_and_types;
            output_shapes_and_types.push_back(shape_inference::ShapeAndType(
                newoutput0, shapes_and_types->at(0).dtype));
            c->set_output_handle_shapes_and_types(0, output_shapes_and_types);
          }
        }
      }

      return absl::OkStatus();
    });

REGISTER_OP("TPUPartitionedInputV2")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .Attr("partition_dims: list(int)")
    .Attr("is_packed: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      std::vector<int> partition_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("partition_dims", &partition_dims));
      bool is_packed;
      TF_RETURN_IF_ERROR(c->GetAttr("is_packed", &is_packed));

      int num_partitions = 1;
      for (const int& partition_dim : partition_dims) {
        num_partitions *= partition_dim;
      }

      bool replicated = partition_dims.empty();
      int num_inputs_expected = is_packed ? 1 : num_partitions;
      if (!((replicated && !is_packed) ||
            (c->num_inputs() == num_inputs_expected))) {
        // we cannot validate the number of inputs for replicated, unpacked ops
        // since we cannot infer the number of partitions from partition_dims
        return absl::InvalidArgumentError(
            absl::StrCat("Expected ", num_inputs_expected, " inputs, got ",
                         c->num_inputs(), "."));
      } else if (c->num_inputs() == 0) {
        return absl::InvalidArgumentError(
            "Expected at least one input to TPUPartitionedInputV2.");
      }

      ShapeHandle output_shape;
      if (dtype == DT_RESOURCE) {
        ShapeHandle previous_shape_handle;
        const std::vector<shape_inference::ShapeAndType>* shapes_and_types =
            nullptr;
        for (int i = c->num_inputs() - 1; i >= 0; --i) {
          shapes_and_types = c->input_handle_shapes_and_types(i);
          if (shapes_and_types) {
            ShapeHandle shape_handle = shapes_and_types->at(0).shape;
            if (!c->FullyDefined(shape_handle)) {
              return absl::InvalidArgumentError(
                  absl::StrCat("Inputs must have static shape,", "input[", i,
                               "] has unknown dimension."));
            }

            if (i != c->num_inputs() - 1) {
              ShapeHandle tmp;
              if (!c->Merge(shape_handle, previous_shape_handle, &tmp).ok()) {
                return absl::InvalidArgumentError(
                    "Inputs must have the same shape.");
              }
            } else {
              previous_shape_handle = shape_handle;
            }
          }
        }

        if (shapes_and_types) {
          TF_ASSIGN_OR_RETURN(
              output_shape,
              _ComputeOutputShape(c, previous_shape_handle, partition_dims));
          std::vector<shape_inference::ShapeAndType> output_shapes_and_types;
          output_shapes_and_types.push_back(shape_inference::ShapeAndType(
              output_shape, shapes_and_types->at(0).dtype));
          c->set_output_handle_shapes_and_types(0, output_shapes_and_types);
        }
      }

      if (!c->FullyDefined(output_shape)) {
        TF_ASSIGN_OR_RETURN(
            output_shape, _ComputeOutputShape(c, c->input(0), partition_dims));
      }

      c->set_output(0, output_shape);

      return absl::OkStatus();
    });

}  // namespace tensorflow
