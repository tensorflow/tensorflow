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

#include "tensorflow/lite/experimental/litert/compiler/plugin/algo.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "llvm/ADT/MapVector.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"

namespace litert::internal {
namespace {

void MakeDispatchOp(LiteRtOpT& op) {
  ABSL_DCHECK(op.Inputs().empty());
  ABSL_DCHECK(op.Outputs().empty());
  op.SetOpCode(kLiteRtOpCodeTflCustom);
  detail::SetTflOpCodeInd(op, detail::kDispatchOpCodeTflInd);
  op.ClearCustomOptions();
}

//
// flatlist to partition(s)
//===----------------------------------------------------------------------===//

class DisjointSets {
 public:
  static std::vector<std::vector<LiteRtOp>> GetPartitionsFromFlatList(
      const std::vector<LiteRtOp>& flat_op_list);

 private:
  void Insert(LiteRtOp op, LiteRtOp parent);
  std::vector<std::vector<LiteRtOp>> GetBuckets();
  LiteRtOp GetBucket(LiteRtOp op);
  // NOLINTBEGIN
  llvm::MapVector<LiteRtOp, LiteRtOp> map_;
  // NOLINTEND
};

std::vector<std::vector<LiteRtOp>> DisjointSets::GetPartitionsFromFlatList(
    const std::vector<LiteRtOp>& flat_op_list) {
  DisjointSets disjoint_sets;
  for (auto* op : flat_op_list) {
    disjoint_sets.map_[op] = op;
  }

  for (auto* op : flat_op_list) {
    for (auto* output : op->Outputs()) {
      for (auto* user : output->Users()) {
        if (disjoint_sets.map_.count(user) == 0) {
          continue;
        }
        disjoint_sets.Insert(op, user);
      }
    }
  }

  return disjoint_sets.GetBuckets();
}

void DisjointSets::Insert(LiteRtOp op, LiteRtOp parent) {
  auto* parent_bucket = GetBucket(parent);
  auto* op_bucket = GetBucket(op);
  if (op_bucket == parent_bucket) {
    return;
  }
  map_[op_bucket] = parent_bucket;
}

// Get all disjoint sets.
std::vector<std::vector<LiteRtOp>> DisjointSets::GetBuckets() {
  // NOLINTBEGIN
  std::unordered_map<LiteRtOp, std::vector<LiteRtOp>> invert_map;
  // NOLINTEND
  for (const auto& entry : map_) {
    auto* bucket = GetBucket(entry.first);

    if (invert_map.find(bucket) == invert_map.end()) {
      invert_map.insert_or_assign(bucket, std::vector<LiteRtOp>{});
    }

    invert_map[bucket].push_back(entry.first);
  }

  std::vector<std::vector<LiteRtOp>> res;
  res.reserve(invert_map.size());

  for (auto& entry : invert_map) {
    res.push_back(std::move(entry.second));
  }

  return res;
}

// Gets the pointer which serves as the key for given ops bucket. Collapses
// paths to amortize.
LiteRtOp DisjointSets::GetBucket(LiteRtOp op) {
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
  // Slices "partitions" from "root" into the empty subgraph "slice". Assumes
  // the partition is a valid sub-DAG, and replaces it witha single
  // tfl.custom_op in "root". A reference to that op is returned.
  static LiteRtOp SlicePartitionFromGraph(LiteRtSubgraphT& root,
                                          LiteRtSubgraph slice,
                                          std::vector<LiteRtOp>& partition);

 private:
  explicit GraphSlicer(LiteRtSubgraph slice) : slice_(slice) {}

  void CloneInto(const LiteRtOpT& op);

  void RerouteTensorsThroughCustomOp(const LiteRtSubgraphT& root);

  LiteRtSubgraph slice_;
  // Maps tensor in old subgraph to tensor in new subgraph.
  // NOLINTBEGIN
  llvm::MapVector<LiteRtTensor, LiteRtTensor> tensor_map_;
  // NOLINTEND
  LiteRtOp dispatch_op_ = nullptr;
};

LiteRtOp GraphSlicer::SlicePartitionFromGraph(
    LiteRtSubgraphT& root, LiteRtSubgraph slice,
    std::vector<LiteRtOp>& partition) {
  GraphSlicer slicer(slice);

  // Register input tensors of the sliced partition WRT to their original order
  // in the root subgraph. This ensures the order of input tensors of the
  // later outlined custom op is the same as the order of input tensors of the
  // GraphInputs.
  absl::flat_hash_set<LiteRtTensor> used_tensors;

  // Get all tensors used in the partition.
  for (auto* op : partition) {
    used_tensors.insert(op->Inputs().cbegin(), op->Inputs().cend());
  }
  for (auto* old_input : root.Inputs()) {
    if (used_tensors.contains(old_input)) {
      auto* new_input = &MakeClone(*slicer.slice_, *old_input);
      slicer.slice_->Inputs().push_back(new_input);
      slicer.tensor_map_.insert({old_input, new_input});
    }
  }

  for (auto* op : partition) {
    slicer.CloneInto(*op);
  }

  for (auto* op : partition) {
    Drop(*op);
  }

  // Reuse the storage from the last op in partition to maintain
  // toplogical order.
  slicer.dispatch_op_ = partition.back();

  MakeDispatchOp(*slicer.dispatch_op_);
  slicer.RerouteTensorsThroughCustomOp(root);

  DCE(root);

  return slicer.dispatch_op_;
}

void GraphSlicer::RerouteTensorsThroughCustomOp(const LiteRtSubgraphT& root) {
  for (auto& [old_tensor, new_tensor] : tensor_map_) {
    // Reroute tensors which need to be passed into the scope of the new
    // subgraph to inputs of the custom op.
    if (new_tensor->DefiningOp() == nullptr && !IsConstant(*new_tensor)) {
      AttachInput(old_tensor, *dispatch_op_);
      continue;
    }

    // Reroute custom op as the definer of tensors within the removed partition
    // and referenced later in the root graph.
    if ((!old_tensor->Users().empty() && !IsConstant(*old_tensor)) ||
        FindOutput(root, *old_tensor)) {
      AttachOutput(old_tensor, *dispatch_op_);
      slice_->Outputs().push_back(new_tensor);
    }
  }
}

void GraphSlicer::CloneInto(const LiteRtOpT& old_op) {
  auto& new_op = MakeClone(*slice_, old_op);

  for (auto i = 0; i < old_op.NumInputs(); ++i) {
    auto* old_input = old_op.Inputs().at(i);
    LiteRtTensor new_input;
    if (tensor_map_.contains(old_input)) {
      // If old_input is already in the map then map[input] is its cloned
      // counterpart in the new graph.
      new_input = tensor_map_[old_input];
    } else {
      // Otherwise, it must be a new subgraph input (or constant).
      new_input = &MakeClone(*slice_, *old_input);
      if (!IsConstant(*new_input)) {
        slice_->Inputs().push_back(new_input);
      }

      tensor_map_.insert({old_input, new_input});
    }

    AttachInput(new_input, new_op);
  }

  for (int i = 0; i < old_op.NumOutputs(); ++i) {
    auto* old_output = old_op.Outputs().at(i);
    auto* new_output = &MakeClone(*slice_, *old_output);
    AttachOutput(new_output, new_op);

    // Update the values defined in scope of the new subgraph.
    tensor_map_.insert({old_output, new_output});
  }
}

}  // namespace

std::vector<std::vector<LiteRtOp>> GroupPartitions(
    const std::vector<LiteRtOp>& ops) {
  return DisjointSets::GetPartitionsFromFlatList(ops);
}

LiteRtOp OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
                          std::vector<LiteRtOp>& partition) {
  return GraphSlicer::SlicePartitionFromGraph(root, slice, partition);
}

}  // namespace litert::internal
