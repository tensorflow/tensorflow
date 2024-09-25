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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_ALGO_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_ALGO_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/MapVector.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

// NOLINTBEGIN

namespace algo {

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
  llvm::MapVector<LrtOp, LrtOp> map_;
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

    if (invert_map.find(bucket) == invert_map.end()) {
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

// TODO: b/365339578 - Move helpers from algo.h to the internal model library.

inline void CloneOpData(const LrtOpT& old_op, LrtOpT& new_op) {
  // TODO: b/365339578 - Support options in op clone.
  new_op.op_code = old_op.op_code;
}

inline void CloneTensorData(const LrtTensorT& old_tensor,
                            LrtTensorT& new_tensor) {
  new_tensor.type_id = old_tensor.type_id;
  new_tensor.type_detail = old_tensor.type_detail;
  new_tensor.buffer.fb_buffer = std::make_unique<tflite::BufferT>();
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

inline void EraseUse(LrtTensor tensor, lrt_param_index_t use_ind) {
  if (use_ind < 0 || use_ind >= tensor->users.size()) {
    return;
  }
  tensor->users[use_ind] = tensor->users.back();
  tensor->users.pop_back();
  tensor->user_arg_inds[use_ind] = tensor->user_arg_inds.back();
  tensor->user_arg_inds.pop_back();
}

inline void EraseUse(LrtTensor tensor, LrtOp user) {
  auto use_ind = FindUseInd(tensor, user);
  if (!use_ind.has_value()) {
    _LRT_D_MSG("Trying to erase from tensor that doesn't use.")
    return;
  }
  EraseUse(tensor, use_ind.value());
}

// Push tensor to the end of ops arguments.
inline void AddUse(LrtTensorT& tensor, LrtOpT& op) {
  op.inputs.push_back(&tensor);
  tensor.users.push_back(&op);
  tensor.user_arg_inds.push_back(op.inputs.size() - 1);
}

inline void AddOutput(LrtOpT& op, LrtTensorT& tensor) {
  DCHECK(tensor.defining_op == nullptr);
  op.outputs.push_back(&tensor);
  tensor.defining_op = &op;
  tensor.defining_op_out_ind = op.outputs.size() - 1;
}

inline LrtTensor RequestNewTensor(LrtSubgraph subgraph,
                                  const LrtTensorT& like) {
  auto& new_tensor = subgraph->tensors_storage.emplace_back();
  CloneTensorData(like, new_tensor);
  return &new_tensor;
}

inline LrtTensor RequestNewInput(LrtSubgraph subgraph, const LrtTensorT& like) {
  auto new_tensor = RequestNewTensor(subgraph, like);
  subgraph->inputs.push_back(new_tensor);
  return new_tensor;
}

inline LrtOp RequestNewOp(LrtSubgraph subgraph, const LrtOpT& like) {
  auto& new_op = subgraph->ops_storage.emplace_back();
  CloneOpData(like, new_op);
  return &new_op;
}

inline void AddOutput(LrtSubgraph subgraph, LrtTensor tensor) {
  subgraph->outputs.push_back(tensor);
}

inline bool IsOutput(const LrtSubgraphT& subgraph, LrtTensor tensor) {
  return std::count(subgraph.outputs.begin(), subgraph.outputs.end(), tensor) >
         0;
}

inline void UpdateReferences(LrtSubgraphT& subgraph) {
  subgraph.tensors.clear();
  subgraph.ops.clear();
  for (auto& tensor : subgraph.tensors_storage) {
    subgraph.tensors.push_back(&tensor);
  }
  for (auto& op : subgraph.ops_storage) {
    subgraph.ops.push_back(&op);
  }
}

inline void Drop(LrtOpT& op) {
  for (auto tensor : op.inputs) {
    EraseUse(tensor, &op);
  }
  op.inputs.clear();
  for (auto tensor : op.outputs) {
    tensor->defining_op = nullptr;
  }
  op.outputs.clear();
}

// TODO expand dead code elimination to work recursively. This is a very simple.
inline void DCE(LrtSubgraphT& subgraph) {
  auto& ops = subgraph.ops_storage;
  for (auto it = ops.begin(); it != ops.end();) {
    if (it->inputs.empty() && it->outputs.empty()) {
      it = ops.erase(it);
    } else {
      ++it;
    }
  }

  std::set<LrtTensor> inputs(subgraph.inputs.begin(), subgraph.inputs.end());
  std::set<LrtTensor> outputs(subgraph.outputs.begin(), subgraph.outputs.end());

  auto& tensors = subgraph.tensors_storage;
  for (auto it = tensors.begin(); it != tensors.end();) {
    auto* tensor = &*it;

    const bool not_in = inputs.find(tensor) == inputs.end();
    const bool not_out = outputs.find(tensor) == outputs.end();
    const bool dead = tensor->defining_op == nullptr && tensor->users.empty();

    if (not_in && not_out && dead) {
      it = tensors.erase(it);
    } else {
      ++it;
    }
  }

  UpdateReferences(subgraph);
}

class GraphSlicer {
 public:
  // Slices "partitions" from "root" into the empty subgraph "slice". Assumes
  // the partition is a valid sub-DAG, and replaces it witha single
  // tfl.custom_op in "root". A reference to that op is returned.
  static LrtOp SlicePartitionFromGraph(LrtSubgraphT& root, LrtSubgraph slice,
                                       std::vector<LrtOp>& partition);

 private:
  explicit GraphSlicer(LrtSubgraph slice) : slice_(slice) {}

  void CloneInto(const LrtOpT& op);

  void RerouteTensorsThroughCustomOp(const LrtSubgraphT& root);

  LrtSubgraph slice_;
  // maps tensor in old subgraph to tensor in new subgraph.
  llvm::MapVector<LrtTensor, LrtTensor> tensor_map_;
  LrtOp hal_cal_op_ = nullptr;
};

inline LrtOp GraphSlicer::SlicePartitionFromGraph(
    LrtSubgraphT& root, LrtSubgraph slice, std::vector<LrtOp>& partition) {
  GraphSlicer slicer(slice);

  for (auto* op : partition) {
    slicer.CloneInto(*op);
  }

  for (auto* op : partition) {
    Drop(*op);
  }

  // Reuse the storage from the last op in partition to maintain
  // toplogical order.
  slicer.hal_cal_op_ = partition.back();
  slicer.hal_cal_op_->op_code = kLrtOpCodeTflCustom;

  UpdateReferences(*slicer.slice_);
  slicer.RerouteTensorsThroughCustomOp(root);
  DCE(root);

  return slicer.hal_cal_op_;
}

// TODO replace this with iteration order sensitve one and fix the reversered
// arg order issue
inline void GraphSlicer::RerouteTensorsThroughCustomOp(
    const LrtSubgraphT& root) {
  for (auto& [old_tensor, new_tensor] : tensor_map_) {
    // Reroute tensors which need to be passed into the scope of the new
    // subgraph to inputs of the custom op.
    if (new_tensor->defining_op == nullptr) {
      AddUse(*old_tensor, *hal_cal_op_);
      continue;
    }

    // Reroute custom op as the definer of tensors within the removed partition
    // and referenced latern in the root graph.
    if (!old_tensor->users.empty() || IsOutput(root, old_tensor)) {
      DCHECK(old_tensor->defining_op == nullptr)
          << "Defining op should have been removed from the graph";
      AddOutput(*hal_cal_op_, *old_tensor);
      AddOutput(slice_, new_tensor);
    }
  }
}

inline void GraphSlicer::CloneInto(const LrtOpT& old_op) {
  auto& new_op = *RequestNewOp(slice_, old_op);

  for (int i = 0; i < old_op.inputs.size(); ++i) {
    auto old_input = old_op.inputs[i];
    LrtTensor new_input;

    if (tensor_map_.contains(old_input)) {
      // If old_input is already in the map then map[input] is its cloned
      // counterpart in the new graph.
      new_input = tensor_map_[old_input];
    } else {
      // Otherwise, it must be a new subgraph input.
      new_input = RequestNewInput(slice_, *old_input);
      tensor_map_.insert({old_input, new_input});
    }

    AddUse(*new_input, new_op);
  }

  for (int i = 0; i < old_op.outputs.size(); ++i) {
    auto old_output = old_op.outputs[i];

    auto new_output = RequestNewTensor(slice_, *old_output);
    AddOutput(new_op, *new_output);

    // Update the values defined in scope of the new subgraph.
    tensor_map_.insert({old_output, new_output});
  }
}

}  // namespace algo

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_ALGO_H_

// NOLINTEND
