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
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status UnpartitionEmbeddingLookup::Run(Model* model,
                                                     std::size_t op_index,
                                                     bool* modified) {
  *modified = false;
  // Collapses a partitioned tf.nn.embedding_lookup back into a single Gather.
  // https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
  // This transform attempts to identify the len(params) > 1 case and collapse
  // it to the len(params) = 1 case by concatenating the original params and
  // reversing the partitioning.
  //
  // If len(params) to the tf.nn.embedding_lookup == 1, the whole op becomes
  // simply a gather:
  // https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/ops/embedding_ops.py#L150
  //
  // Notes on this implementation:
  // - only supports partition_strategy='mod'
  //
  // A rough graph of a partitioned embedding_lookup looks like:
  //   (ids)--+-->FloorDiv--+-->DynamicPartition-->[[Gather]]--\
  //          \-->FloorMod--/                                  |
  //                 V                                         |
  //   Range-->DynamicPartition-------->DynamicStitch<---------/
  //  (const)                                V
  //                                     (embeddings)

  // First look for the final DynamicStitch.
  auto op_it = model->operators.begin() + op_index;
  if (op_it->get()->type != OperatorType::kDynamicStitch) {
    return ::tensorflow::Status::OK();
  }
  auto* stitch_op = static_cast<DynamicStitchOperator*>(op_it->get());

  // Split up the DynamicStitch inputs into the indices and data.
  std::vector<std::string> stitch_indices_inputs;
  std::vector<std::string> stitch_data_inputs;
  for (int i = 0; i < stitch_op->num_partitions; ++i) {
    stitch_indices_inputs.push_back(stitch_op->inputs[i]);
  }
  for (int i = stitch_op->num_partitions; i < stitch_op->num_partitions * 2;
       ++i) {
    stitch_data_inputs.push_back(stitch_op->inputs[i]);
  }

  // Validate all indices come from the same DynamicPartition.
  DynamicPartitionOperator* indices_partition_op = nullptr;
  for (const std::string& indices_partition_output_name :
       stitch_indices_inputs) {
    auto* op = GetOpWithOutput(*model, indices_partition_output_name);
    CHECK(op) << "Source of " << indices_partition_output_name << " not found";
    if (op->type != OperatorType::kDynamicPartition) {
      AddMessageF(
          "Skipping because indices input %s into "
          "%s is unexpected",
          LogName(*op), LogName(*stitch_op));
      return ::tensorflow::Status::OK();
    }
    if (!indices_partition_op) {
      indices_partition_op = static_cast<DynamicPartitionOperator*>(op);
    } else {
      // Ensure this is the same op as previous ones.
      if (op != indices_partition_op) {
        AddMessageF(
            "Skipping because indices input %s into "
            "%s is from a different source op than others",
            LogName(*op), LogName(*stitch_op));
        return ::tensorflow::Status::OK();
      }
    }
  }
  CHECK(indices_partition_op) << "No indices inputs";

  // The data for the indices must be a constant range of the array shape.
  if (!IsConstantParameterArray(*model, indices_partition_op->inputs[0])) {
    AddMessageF("Skipping because indices partition data is non-constant");
    return ::tensorflow::Status::OK();
  }
  auto& indices_data_array = model->GetArray(indices_partition_op->inputs[0]);
  if (indices_data_array.data_type == ArrayDataType::kNone) {
    // Yield until data types are propagated.
    return ::tensorflow::Status::OK();
  }
  CHECK(indices_data_array.data_type == ArrayDataType::kInt32)
      << "Indices partition inputs must be int32";
  const auto& indices_data_buffer =
      indices_data_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (size_t i = 0; i < indices_data_buffer.size(); ++i) {
    CHECK_EQ(indices_data_buffer[i], i) << "Indices range must be identity";
  }

  // Find all of the gathers used for the data inputs.
  std::vector<GatherOperator*> gather_ops;
  for (const std::string& gather_output_name : stitch_data_inputs) {
    auto* op = GetOpWithOutput(*model, gather_output_name);
    CHECK(op) << "Source of " << gather_output_name << " not found";
    if (op->type != OperatorType::kGather) {
      AddMessageF(
          "Skipping because data input %s into %s "
          "is unexpected",
          LogName(*op), LogName(*stitch_op));
      return ::tensorflow::Status::OK();
    }
    gather_ops.push_back(static_cast<GatherOperator*>(op));
  }

  // Validate all gathers come from the same DynamicPartition.
  DynamicPartitionOperator* data_partition_op = nullptr;
  for (auto* gather_op : gather_ops) {
    auto* op = GetOpWithOutput(*model, gather_op->inputs[1]);
    CHECK(op) << "Source of " << gather_op->inputs[1] << " not found";
    if (op->type != OperatorType::kDynamicPartition) {
      AddMessageF(
          "Skipping because data input %s into "
          "%s is unexpected",
          LogName(*op), LogName(*gather_op));
      return ::tensorflow::Status::OK();
    }
    if (!data_partition_op) {
      data_partition_op = static_cast<DynamicPartitionOperator*>(op);
    } else {
      // Ensure this is the same op as previous ones.
      if (op != data_partition_op) {
        AddMessageF(
            "Skipping because data input %s into "
            "%s is from a different source op than others",
            LogName(*op), LogName(*gather_op));
        return ::tensorflow::Status::OK();
      }
    }
  }
  CHECK(data_partition_op) << "No data inputs";

  // Validate the partition ops have the same sizes.
  CHECK_EQ(indices_partition_op->num_partitions,
           data_partition_op->num_partitions)
      << "Indices and data partition ops have differing dimensions";
  int num_partitions = indices_partition_op->num_partitions;

  // Partition strategy of 'mod' gives us a FloorMod and FloorDiv.
  // The gather partition uses the FloorDiv as the data and FloorMod as the
  // partitions and the indices use the FloorMod as their partitions.
  Operator* div_op = GetOpWithOutput(*model, data_partition_op->inputs[0]);
  Operator* mod_op = GetOpWithOutput(*model, data_partition_op->inputs[1]);
  CHECK(div_op && div_op->type == OperatorType::kFloorDiv)
      << "Unsupported partition strategy";
  CHECK(mod_op && mod_op->type == OperatorType::kFloorMod)
      << "Unsupported partition strategy";
  CHECK_EQ(mod_op, GetOpWithOutput(*model, indices_partition_op->inputs[1]))
      << "Indices and data partition ops require the same partition strategy "
         "and inputs";

  // Glob together all of the gather data. This is not yet in the correct order.
  auto* gather_params_concat_op = new ConcatenationOperator;
  for (const auto& gather_op : gather_ops) {
    gather_params_concat_op->inputs.push_back(gather_op->inputs[0]);
  }
  gather_params_concat_op->outputs.push_back(
      AvailableArrayName(*model, gather_ops[0]->inputs[0] + "_unpartitioned"));
  op_it = model->operators.emplace(op_it, gather_params_concat_op) + 1;
  model->GetOrCreateArray(gather_params_concat_op->outputs[0]);

  // Permute the gather params to undo the partitioning that was originally
  // done.
  auto* gather_params_permute_op = new GatherOperator;
  gather_params_permute_op->inputs.push_back(
      gather_params_concat_op->outputs[0]);
  gather_params_permute_op->inputs.push_back(
      AvailableArrayName(*model, gather_ops[0]->inputs[0] + "_permuted/perm"));
  gather_params_permute_op->outputs.push_back(
      AvailableArrayName(*model, gather_ops[0]->inputs[0] + "_permuted"));
  gather_params_permute_op->axis = {0};
  op_it = model->operators.emplace(op_it, gather_params_permute_op) + 1;
  model->GetOrCreateArray(gather_params_permute_op->outputs[0]);
  const auto& partition_array = model->GetArray(gather_ops[0]->inputs[0]);
  const auto& partition_array_dims = partition_array.shape().dims();
  gather_params_permute_op->input_rank =
      partition_array.shape().dimensions_count();
  auto& perm_array =
      model->GetOrCreateArray(gather_params_permute_op->inputs[1]);
  perm_array.data_type = ArrayDataType::kInt32;
  perm_array.mutable_shape()->ReplaceDims(
      {num_partitions * partition_array_dims[0]});
  auto& perm_data = perm_array.GetMutableBuffer<ArrayDataType::kInt32>().data;
  perm_data.resize(RequiredBufferSizeForShape(perm_array.shape()));
  // NOTE: this is what relies on the partition_strategy.
  for (int i = 0; i < num_partitions * partition_array_dims[0]; ++i) {
    int p = i % num_partitions;
    perm_data[i] = p * partition_array_dims[0] + i / num_partitions;
  }

  // Insert the new unpartitioned gather op.
  auto* merged_gather_op = new GatherOperator;
  merged_gather_op->inputs = {gather_params_permute_op->outputs[0],
                              mod_op->inputs[0]};
  merged_gather_op->outputs = {stitch_op->outputs[0]};
  merged_gather_op->input_rank = partition_array.shape().dimensions_count();
  merged_gather_op->axis = {0};
  model->operators.emplace(op_it, merged_gather_op);

  AddMessageF(
      "Replacing suspected partitioned tf.nn.embedding_lookup (starting at %s "
      "+ %s and ending at %s) with a single unpartitioned gather %s",
      LogName(*div_op), LogName(*mod_op), LogName(*stitch_op),
      LogName(*merged_gather_op));

  // Ensure the stitch output array is dead, as we don't want whatever was in it
  // previously now that we've redefined it. It'll be recreated when needed.
  model->EraseArray(merged_gather_op->outputs[0]);
  model->GetOrCreateArray(merged_gather_op->outputs[0]);

  // Erase all the original ops.
  DeleteOpAndArrays(model, div_op);
  DeleteOpAndArrays(model, mod_op);
  for (auto* gather_op : gather_ops) {
    DeleteOpAndArrays(model, gather_op);
  }
  DeleteOpAndArrays(model, indices_partition_op);
  DeleteOpAndArrays(model, data_partition_op);
  DeleteOpAndArrays(model, stitch_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
