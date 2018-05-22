/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

xla::StatusOr<xla::XlaOp> XlaScatter(
    const xla::XlaOp& buffer, const xla::XlaOp& updates,
    const xla::XlaOp& indices, bool indices_are_vectors,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner,
    xla::XlaBuilder* builder) {
  TF_ASSIGN_OR_RETURN(xla::Shape buffer_shape, builder->GetShape(buffer));
  TF_RETURN_IF_ERROR(builder->GetShape(updates).status());
  TF_ASSIGN_OR_RETURN(xla::Shape indices_shape, builder->GetShape(indices));
  gtl::ArraySlice<int64> indices_dims =
      xla::AsInt64Slice(indices_shape.dimensions());
  gtl::ArraySlice<int64> buffer_dims =
      xla::AsInt64Slice(buffer_shape.dimensions());

  // If the indices are N-dimensional, the minor dimension of indices contains
  // the indices to update. Otherwise the indices are all scalars.
  int64 num_index_dims = 1;
  if (indices_are_vectors) {
    TF_RET_CHECK(!indices_dims.empty());
    num_index_dims = indices_dims.back();
    if (num_index_dims > xla::ShapeUtil::Rank(buffer_shape)) {
      return errors::InvalidArgument(
          "The size of the minor dimension of the indices (shape: ",
          xla::ShapeUtil::HumanString(indices_shape),
          ") must be <= the rank of the buffer (shape: ",
          xla::ShapeUtil::HumanString(buffer_shape), ")");
    }
    indices_dims.pop_back();
  }

  int64 num_indices = 1;
  for (int64 dim : indices_dims) {
    num_indices *= dim;
  }

  // Degenerate case: nothing to update. Return the buffer unchanged.
  if (num_indices == 0) {
    return buffer;
  }

  // If any of the indexed dimensions are zero in the buffer, the update cannot
  // succeed since it updates a slice of size 1.
  for (int64 i = 0; i < num_index_dims; ++i) {
    if (xla::ShapeUtil::GetDimension(buffer_shape, i) == 0) {
      return errors::InvalidArgument("Scatter dimension ", i,
                                     " is of size zero in tensor with shape ",
                                     xla::ShapeUtil::HumanString(buffer_shape));
    }
  }

  // Shape of the non-indexed dimensions of the buffer.
  std::vector<int64> buffer_shape_post_axes(
      buffer_dims.begin() + num_index_dims, buffer_dims.end());

  // Flatten the major dimensions of indices and updates into a single dimension
  // for ease of iteration.
  std::vector<int64> flat_indices_shape({num_indices});
  if (indices_are_vectors) {
    flat_indices_shape.push_back(num_index_dims);
  }

  std::vector<int64> flat_updates_shape({num_indices});
  flat_updates_shape.insert(flat_updates_shape.end(),
                            buffer_shape_post_axes.begin(),
                            buffer_shape_post_axes.end());

  // Construct the initial values of the loop-carried Tensors.
  auto flat_indices = builder->Reshape(indices, flat_indices_shape);
  auto flat_updates = builder->Reshape(updates, flat_updates_shape);
  auto init = {flat_indices, flat_updates, buffer};

  // Constructs the loop body. The implementation of scatter is essentially:
  // for i in range(num_indices):
  //   index = dynamic-slice(indices, i)
  //   update = dynamic-slice(updates, i)
  //   buffer = dynamic-update-slice(buffer, update, index)
  auto body_fn = [&](xla::XlaOp i, gtl::ArraySlice<xla::XlaOp> loop_vars,
                     xla::XlaBuilder* body_builder) {
    auto indices = loop_vars[0];
    auto updates = loop_vars[1];
    auto buffer = loop_vars[2];

    auto zero_index = body_builder->ConstantLiteral(
        xla::Literal::Zero(indices_shape.element_type()));

    // Slice the i-th index from the indices array.
    xla::XlaOp index;
    auto indices_offset = body_builder->Reshape(i, {1});
    if (indices_are_vectors) {
      indices_offset = body_builder->Pad(indices_offset, zero_index,
                                         xla::MakeEdgePaddingConfig({{0, 1}}));

      index = body_builder->DynamicSlice(indices, indices_offset,
                                         {1, num_index_dims});
      index = body_builder->Collapse(index, {0, 1});
    } else {
      index = body_builder->DynamicSlice(indices, indices_offset, {1});
    }

    // Discard updates with negative indices, since some users expect this.
    auto index_in_range =
        body_builder->ReduceAll(body_builder->Le(zero_index, index),
                                body_builder->ConstantR0<bool>(true),
                                xla::CreateScalarAndComputation(body_builder));

    // Make the index in bounds to prevent implementation defined behavior.
    index = body_builder->Max(index, zero_index);
    index = body_builder->Pad(
        index, zero_index,
        xla::MakeEdgePaddingConfig({{0, buffer_shape_post_axes.size()}}));

    // Slice the i-th index from the updates array.
    auto updates_offset = body_builder->Reshape(i, {1});
    updates_offset = body_builder->Pad(
        updates_offset, zero_index,
        xla::MakeEdgePaddingConfig({{0, buffer_shape_post_axes.size()}}));
    std::vector<int64> flat_updates_slice_shape({1});
    flat_updates_slice_shape.insert(flat_updates_slice_shape.end(),
                                    buffer_shape_post_axes.begin(),
                                    buffer_shape_post_axes.end());
    auto update = body_builder->DynamicSlice(updates, updates_offset,
                                             flat_updates_slice_shape);

    // Unflatten the major (iteration) dimensions of the slice to their
    // original shape.
    std::vector<int64> updates_slice_shape(num_index_dims, 1);
    updates_slice_shape.insert(updates_slice_shape.end(),
                               buffer_shape_post_axes.begin(),
                               buffer_shape_post_axes.end());
    update = body_builder->Reshape(update, updates_slice_shape);

    // Apply the update to the buffer. If there is a combiner, use it to merge
    // the current values with the update.
    auto current_value =
        body_builder->DynamicSlice(buffer, index, updates_slice_shape);
    if (combiner) {
      update = combiner(current_value, update, body_builder);
    }
    // Use the current value instead of the update if the index is out of
    // bounds.
    update = body_builder->Select(index_in_range, update, current_value);
    // Apply the update.
    buffer = body_builder->DynamicUpdateSlice(buffer, update, index);

    return std::vector<xla::XlaOp>{indices, updates, buffer};
  };

  TF_ASSIGN_OR_RETURN(auto outputs,
                      XlaForEachIndex(num_indices, indices_shape.element_type(),
                                      body_fn, init, "scatter", builder));
  return outputs[2];
}

}  // namespace tensorflow
