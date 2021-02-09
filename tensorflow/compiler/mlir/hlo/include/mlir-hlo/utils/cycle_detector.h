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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CYCLE_DETECTOR_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CYCLE_DETECTOR_H_

#include <vector>

#include "llvm/ADT/DenseMap.h"

namespace mlir {

// -------------------------------------------------------------------

// This file contains a light version of GraphCycles implemented in
// tensorflow/compiler/jit/graphcycles/graphcycles.h
//
// We re-implement it here because we do not want to rely
// on TensorFlow data structures, and hence we can move
// corresponding passes to llvm repo. easily in case necessnary.

// --------------------------------------------------------------------

// This is a set data structure that provides a deterministic iteration order.
// The iteration order of elements only depends on the sequence of
// inserts/deletes, so as long as the inserts/deletes happen in the same
// sequence, the set will have the same iteration order.
//
// Assumes that T can be cheaply copied for simplicity.
template <typename T>
class OrderedSet {
 public:
  // Inserts `value` into the ordered set.  Returns true if the value was not
  // present in the set before the insertion.
  bool Insert(T value) {
    bool new_insertion =
        value_to_index_.insert({value, value_sequence_.size()}).second;
    if (new_insertion) {
      value_sequence_.push_back(value);
    }
    return new_insertion;
  }

  // Removes `value` from the set.  Assumes `value` is already present in the
  // set.
  void Erase(T value) {
    auto it = value_to_index_.find(value);

    // Since we don't want to move values around in `value_sequence_` we swap
    // the value in the last position and with value to be deleted and then
    // pop_back.
    value_to_index_[value_sequence_.back()] = it->second;
    std::swap(value_sequence_[it->second], value_sequence_.back());
    value_sequence_.pop_back();
    value_to_index_.erase(it);
  }

  void Reserve(size_t new_size) {
    value_to_index_.reserve(new_size);
    value_sequence_.reserve(new_size);
  }

  void Clear() {
    value_to_index_.clear();
    value_sequence_.clear();
  }

  bool Contains(T value) const { return value_to_index_.count(value); }
  size_t Size() const { return value_sequence_.size(); }

  const std::vector<T>& GetSequence() const { return value_sequence_; }

 private:
  // The stable order that we maintain through insertions and deletions.
  std::vector<T> value_sequence_;

  // Maps values to their indices in `value_sequence_`.
  llvm::DenseMap<T, int> value_to_index_;
};

// ---------------------------------------------------------------------

// GraphCycles detects the introduction of a cycle into a directed
// graph that is being built up incrementally.
//
// Nodes are identified by small integers.  It is not possible to
// record multiple edges with the same (source, destination) pair;
// requests to add an edge where one already exists are silently
// ignored.
//
// It is also not possible to introduce a cycle; an attempt to insert
// an edge that would introduce a cycle fails and returns false.
//
// GraphCycles uses no internal locking; calls into it should be
// serialized externally.

// Performance considerations:
//   Works well on sparse graphs, poorly on dense graphs.
//   Extra information is maintained incrementally to detect cycles quickly.
//   InsertEdge() is very fast when the edge already exists, and reasonably fast
//   otherwise.
//   FindPath() is linear in the size of the graph.
// The current implementation uses O(|V|+|E|) space.

class GraphCycles {
 public:
  explicit GraphCycles(int32_t num_nodes);
  ~GraphCycles();

  // Attempt to insert an edge from x to y.  If the
  // edge would introduce a cycle, return false without making any
  // changes. Otherwise add the edge and return true.
  bool InsertEdge(int32_t x, int32_t y);

  // Remove any edge that exists from x to y.
  void RemoveEdge(int32_t x, int32_t y);

  // Return whether there is an edge directly from x to y.
  bool HasEdge(int32_t x, int32_t y) const;

  // Contracts the edge from 'a' to node 'b', merging nodes 'a' and 'b'. One of
  // the nodes is removed from the graph, and edges to/from it are added to
  // the remaining one, which is returned. If contracting the edge would create
  // a cycle, does nothing and return no value.
  llvm::Optional<int32_t> ContractEdge(int32_t a, int32_t b);

  // Return whether dest_node `y` is reachable from source_node `x`
  // by following edges. This is non-thread-safe version.
  bool IsReachable(int32_t x, int32_t y);

  // Return a copy of the successors set. This is needed for code using the
  // collection while modifying the GraphCycles.
  std::vector<int32_t> SuccessorsCopy(int32_t node) const;

  // Returns all nodes in post order.
  //
  // If there is a path from X to Y then X appears after Y in the
  // returned vector.
  std::vector<int32_t> AllNodesInPostOrder() const;

  // ----------------------------------------------------
  struct Rep;

 private:
  GraphCycles(const GraphCycles&) = delete;
  GraphCycles& operator=(const GraphCycles&) = delete;

  Rep* rep_;  // opaque representation
};

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CYCLE_DETECTOR_H_
