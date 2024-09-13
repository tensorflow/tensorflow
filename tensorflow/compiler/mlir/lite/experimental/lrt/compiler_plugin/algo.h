// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_ALGO_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_ALGO_H_

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/model/lite_rt_model.h"

// NOLINTBEGIN

namespace algo {

// TODO: b/365339578 - Put in graph tools.
inline void EraseUse(LrtTensor tensor, lrt_param_index_t use_ind) {
  if (use_ind < 0 || use_ind >= tensor->users.size()) {
    return;
  }
  tensor->users[use_ind] = tensor->users.back();
  tensor->users.pop_back();
  tensor->user_arg_inds[use_ind] = tensor->user_arg_inds.back();
  tensor->user_arg_inds.pop_back();
}

inline std::optional<lrt_param_index_t> FindUseInd(LrtTensor tensor,
                                                   LrtOp user) {
  for (lrt_param_index_t i = 0; i < tensor->users.size(); ++i) {
    if (tensor->users[i] == user) {
      return i;
    }
  }
  return std::nullopt;
}

//
// flatlist to partition(s)
//===----------------------------------------------------------------------===//

class DisjointSets {
 public:
  static std::vector<std::vector<LrtOp>> GetPartitionsFromFlatList(
      const std::vector<LrtOp>& flat_op_list);

 private:
  void Insert(LrtOp op, LrtOp parent);
  std::vector<std::vector<LrtOp>> GetBuckets();
  LrtOp GetBucket(LrtOp op);
  std::unordered_map<LrtOp, LrtOp> map_;
};

inline std::vector<std::vector<LrtOp>> DisjointSets::GetPartitionsFromFlatList(
    const std::vector<LrtOp>& flat_op_list) {
  DisjointSets disjoint_sets;
  for (auto* op : flat_op_list) {
    disjoint_sets.map_[op] = op;
  }

  for (auto* op : flat_op_list) {
    for (auto* output : op->outputs) {
      for (auto* user : output->users) {
        if (disjoint_sets.map_.count(user) == 0) {
          continue;
        }
        disjoint_sets.Insert(op, user);
      }
    }
  }

  return disjoint_sets.GetBuckets();
}

inline void DisjointSets::Insert(LrtOp op, LrtOp parent) {
  auto* parent_bucket = GetBucket(parent);
  auto* op_bucket = GetBucket(op);
  if (op_bucket == parent_bucket) {
    return;
  }
  map_[op_bucket] = parent_bucket;
}

// Get all disjoint sets.
inline std::vector<std::vector<LrtOp>> DisjointSets::GetBuckets() {
  std::unordered_map<LrtOp, std::vector<LrtOp>> invert_map;
  for (const auto& entry : map_) {
    auto* bucket = GetBucket(entry.first);
    if (!invert_map.contains(bucket)) {
      invert_map.insert_or_assign(bucket, std::vector<LrtOp>{});
    }
    invert_map[bucket].push_back(entry.first);
  }
  std::vector<std::vector<LrtOp>> res;
  res.reserve(invert_map.size());
  for (auto& entry : invert_map) {
    res.push_back(std::move(entry.second));
  }
  return res;
}

// Gets the pointer which serves as the key for given ops bucket. Collapses
// paths to amortize.
inline LrtOp DisjointSets::GetBucket(LrtOp op) {
  auto* parent = map_[op];
  if (op != parent) {
    parent = GetBucket(parent);
    map_[op] = parent;
  }
  return parent;
}

//
// slice partitions out of a subgraph (into new subgraphs)
//===----------------------------------------------------------------------===//

class GraphSlicer {
 public:
  static std::unique_ptr<LrtSubgraphT> SlicePartitionFromGraph(
      LrtSubgraphT& graph, std::vector<LrtOp>& partition);

 private:
  void CloneInto(LrtOpT& op);

  // maps tensor in old subgraph to tensor in new subgraph.
  std::unordered_map<LrtTensor, LrtTensor> tensor_map_;
  std::unique_ptr<LrtSubgraphT> subgraph_ = std::make_unique<LrtSubgraphT>();
  LrtOp hal_cal_op_;
};

inline std::unique_ptr<LrtSubgraphT> GraphSlicer::SlicePartitionFromGraph(
    LrtSubgraphT& graph, std::vector<LrtOp>& partition) {
  GraphSlicer slicer;

  // Append new op to storage.
  slicer.hal_cal_op_ = &graph.ops_storage.emplace_back();
  slicer.hal_cal_op_->op_code = kLrtOpCodeTflCustom;

  for (auto* op : partition) {
    slicer.CloneInto(*op);
  }

  // Now fix up new subgraph outputs. If a tensor is in the map, it is either
  // defined within the new subgraph or it is a subgraph input.
  for (auto& [old_tensor, new_tensor] : slicer.tensor_map_) {
    if (new_tensor->defining_op == nullptr) {
      // Subgraph input, cannot be an output.
      continue;
    }
    if (!old_tensor->users.empty()) {
      // Subgraph output. We already popped all the users that were moved
      // into the new subgraph.

      slicer.subgraph_->outputs.push_back(new_tensor);

      old_tensor->defining_op_out_ind = slicer.hal_cal_op_->outputs.size();
      old_tensor->defining_op = slicer.hal_cal_op_;
      slicer.hal_cal_op_->outputs.push_back(old_tensor);
    }
  }

  std::vector<LrtOp> new_op_refs;
  auto cur_partition_op = partition.begin();

  for (auto& op_ref : graph.ops) {
    if (cur_partition_op == partition.end() || op_ref != *cur_partition_op) {
      new_op_refs.push_back(op_ref);
      continue;
    }
    if (cur_partition_op == partition.end() - 1) {
      new_op_refs.push_back(slicer.hal_cal_op_);
    }
    cur_partition_op++;
  }

  graph.ops = new_op_refs;

  for (auto& new_subgraph_op : slicer.subgraph_->ops_storage) {
    slicer.subgraph_->ops.push_back(&new_subgraph_op);
  }

  return std::move(slicer.subgraph_);
}

inline void GraphSlicer::CloneInto(LrtOpT& old_op) {
  auto& new_op = subgraph_->ops_storage.emplace_back();
  new_op.op_code = old_op.op_code;

  for (int input_ind = 0; input_ind < old_op.inputs.size(); ++input_ind) {
    auto* old_input_tensor = old_op.inputs.at(input_ind);
    const auto old_input_tensor_use_ind =
        FindUseInd(old_input_tensor, &old_op).value();
    EraseUse(old_input_tensor, old_input_tensor_use_ind);

    if (!tensor_map_.contains(old_input_tensor)) {
      // Its a subgraph input.
      auto& new_input_tensor = subgraph_->tensors_storage.emplace_back();
      new_input_tensor.buffer = std::move(old_input_tensor->buffer);
      new_input_tensor.type_id = old_input_tensor->type_id;
      new_input_tensor.type_detail = old_input_tensor->type_detail;

      new_input_tensor.defining_op = nullptr;

      old_input_tensor->user_arg_inds.push_back(hal_cal_op_->inputs.size());
      old_input_tensor->users.push_back(hal_cal_op_);
      subgraph_->inputs.push_back(&new_input_tensor);
      hal_cal_op_->inputs.push_back(old_input_tensor);

      tensor_map_[old_input_tensor] = &new_input_tensor;
    }

    auto new_input_tensor = tensor_map_.at(old_input_tensor);

    new_input_tensor->users.push_back(&new_op);
    new_input_tensor->user_arg_inds.push_back(input_ind);
    new_op.inputs.push_back(new_input_tensor);
  }

  for (int output_ind = 0; output_ind < old_op.outputs.size(); ++output_ind) {
    auto* old_output_tensor = old_op.outputs.at(output_ind);

    auto& new_output_tensor = subgraph_->tensors_storage.emplace_back();
    new_output_tensor.buffer = std::move(old_output_tensor->buffer);
    new_output_tensor.type_id = old_output_tensor->type_id;
    new_output_tensor.type_detail = old_output_tensor->type_detail;

    new_op.outputs.push_back(&new_output_tensor);
    new_output_tensor.defining_op = &new_op;
    new_output_tensor.defining_op_out_ind = output_ind;

    tensor_map_[old_output_tensor] = &new_output_tensor;
  }
}

}  // namespace algo

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_ALGO_H_

// NOLINTEND
