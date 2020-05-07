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
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ConvertTrivialTileToConcat::Run(Model* model,
                                                     std::size_t op_index,
                                                     bool* modified) {
  *modified = false;
  auto tile_it = model->operators.begin() + op_index;
  if (tile_it->get()->type != OperatorType::kTile) {
    return ::tensorflow::Status::OK();
  }
  auto* tile_op = static_cast<TransposeOperator*>(tile_it->get());

  const auto& input_array = model->GetArray(tile_op->inputs[0]);
  const auto& multiples_array = model->GetArray(tile_op->inputs[1]);
  const auto& output_array = model->GetArray(tile_op->outputs[0]);
  if (!input_array.has_shape() || !multiples_array.has_shape() ||
      !output_array.has_shape()) {
    // Yield until PropagateFixedSizes has been run on this op.
    return ::tensorflow::Status::OK();
  }
  // Note: We can assume we have error checked inputs in PropagateFixedSizes.

  if (!multiples_array.buffer) {
    // Yield until the multiples is constant.
    return ::tensorflow::Status::OK();
  }
  std::vector<int32> const& multiples =
      multiples_array.GetBuffer<ArrayDataType::kInt32>().data;

  // We can simplify the tile if only a single dimension is being multiplied.
  // It then just becomes a concat along that dimension.
  int non_one_dims = 0;
  int concat_axis = 0;
  for (int i = 0; i < multiples.size(); ++i) {
    if (multiples[i] != 1) {
      ++non_one_dims;
      concat_axis = i;
    }
  }
  if (non_one_dims != 1) {
    // The tile is non-trivial. Good luck.
    AddMessageF("Tile %s is non-trivial (has more than one multiply dimension)",
                LogName(*tile_op));
    return ::tensorflow::Status::OK();
  }

  // The tile is like a concat.
  AddMessageF("Simplifying %s to a Concat along a single axis %d",
              LogName(*tile_op), concat_axis);

  auto* concat_op = new ConcatenationOperator;

  // Copy input and output.
  // Note that we multiply out the input by the number of times requested.
  for (int i = 0; i < multiples[concat_axis]; ++i) {
    concat_op->inputs.push_back(tile_op->inputs[0]);
  }
  concat_op->axis = concat_axis;
  concat_op->outputs = tile_op->outputs;

  // Delete multiples array if unused.
  if (IsDiscardableArray(*model, tile_op->inputs[1]) &&
      CountOpsWithInput(*model, tile_op->inputs[1]) == 1) {
    model->EraseArray(tile_op->inputs[1]);
  }

  // Replace the operator in the graph.
  model->operators.emplace(tile_it, concat_op);
  DeleteOpAndArrays(model, tile_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
