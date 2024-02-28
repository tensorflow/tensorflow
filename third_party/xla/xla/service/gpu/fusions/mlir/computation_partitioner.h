/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_COMPUTATION_PARTITIONER_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_COMPUTATION_PARTITIONER_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {
namespace mlir_converter {

// Partitions an HLO computation into subgraphs so that all users of a node have
// consistent indexing, i. e. when we compute a node `a` with users `b` and `c`,
// all three nodes will have the same indexing - neither of `b` or `c` will be a
// transpose, reshape, reduce, etc.
//
// Consider the following example, where we assume all nodes affect indexing:
//
//     a   b      Here we create four subgraphs: `a,d,c,e`, `b`, `f` and `g`. If
//      \ /|      `f` and `g` didn't change the indexing, they would be included
//    d  c |      in the `a,d,c,e` subgraph, so we'd have `b` and the rest.
//     \ | |
//       e |      Note that if some users have the same indexing as a node (e.g.
//      / \|      `e` and `g` in the graph to the left), we still have to create
//     f   g      separate subgraphs for `f` and `g`.
//
// The purpose of this partitioning is to allow us to generate code without ever
// having to duplicate instructions: users with incompatible indexing will be in
// different subgraphs, each of which will emit a call to the producer graph.
//
// Note that this partitioning will sometimes create silly subgraphs that should
// (and will) be inlined, e. g. containing only a constant or only a broadcast.
//
// There are two hooks to customize this partitioning:
// - is_subgraph_root: forces the clusterer to start a new subgraph at a given
//   instruction. The instruction is guaranteed to be a in a different subgraph
//   than its users.
// - operand_is_function_argument: marks an operand as a value that is provided
//   to the function as an argument (instead of being generated through a call).
class PartitionedComputation {
 public:
  explicit PartitionedComputation(
      const HloComputation* computation,
      std::function<bool(const HloInstruction*)> is_subgraph_root = nullptr,
      std::function<bool(const HloInstruction*, int)>
          operand_is_function_argument = nullptr);

  struct Subgraph {
    // A unique name of the subgraph. Used for function names.
    std::string name;

    // The instructions that make up this subgraph.
    std::vector<const HloInstruction*> instructions_post_order;

    // The roots. These are guaranteed not to have users inside the subgraph.
    std::vector<const HloInstruction*> roots;

    // For operands that are function arguments (not function calls), stores the
    // mapping from operand to the argument index. The arguments always come
    // after the tensor parameters and output indices; the indices are relative
    // to the argument after the last index argument.
    absl::flat_hash_map<std::pair<const HloInstruction*, int>, int>
        injected_param_indices;

    std::string ToString() const;
  };

  absl::Span<const Subgraph> subgraphs() const { return subgraphs_; }

  const HloComputation& computation() const { return *computation_; }

  const Subgraph& GetRootSubgraph() const {
    return FindSubgraph(computation_->root_instruction());
  }

  // Returns the subgraph containing the given instruction.
  const Subgraph& FindSubgraph(const HloInstruction* instr) const {
    return *instructions_to_subgraphs_.at(instr);
  }

  std::string ToString() const;

 private:
  const HloComputation* computation_;
  std::vector<Subgraph> subgraphs_;
  absl::flat_hash_map<const HloInstruction*, const Subgraph*>
      instructions_to_subgraphs_;
};

// Given a root of a subgraph, returns the corresponding function.
using CallTargetProvider =
    std::function<mlir::func::FuncOp(const HloInstruction* instr)>;

// A collection of PartitionedComputations, starting at a fusion computation and
// including all transitively called computations.
class PartitionedComputations {
 public:
  explicit PartitionedComputations(
      const HloComputation* fusion,
      std::function<bool(const HloInstruction*)> is_subgraph_root = nullptr,
      std::function<bool(const HloInstruction*, int)>
          operand_is_function_argument = nullptr);

  // Partitions the fusion and isolates the given instructions in their own
  // subgraphs. The values of these instructions will be function arguments
  // in their user subgraphs.
  //
  // Example: param -> exp -> transpose -> neg
  // If the transpose is isolated and injected, we will get three subgraphs: one
  // for the exp, one for the transpose and one for the neg. The neg subgraph
  // will have an extra argument for the value of the transpose.
  explicit PartitionedComputations(
      const HloComputation* fusion,
      const absl::flat_hash_set<const HloInstruction*>&
          isolated_and_injected_instructions);

  const PartitionedComputation& FindPartitionedComputation(
      const HloComputation* computation) const {
    return *computation_to_partitioning_.at(computation);
  }

  const PartitionedComputation::Subgraph& FindSubgraph(
      const HloInstruction* instr) const;

  absl::Span<const PartitionedComputation> partitioned_computations() const {
    return partitioned_computations_;
  }

  // Creates a call target lookup function for use with SubgraphToMlir.
  CallTargetProvider CreateCallTargetProvider(
      const absl::flat_hash_map<const PartitionedComputation::Subgraph*,
                                mlir::func::FuncOp>& subgraph_to_func) const;

  // Declares func.func ops for each subgraph in each computation and returns a
  // mapping from subgraph to declared function.
  absl::flat_hash_map<const PartitionedComputation::Subgraph*,
                      mlir::func::FuncOp>
  DeclareFunctions(mlir::ModuleOp module) const;

  std::string ToString() const;

 private:
  std::vector<PartitionedComputation> partitioned_computations_;
  absl::flat_hash_map<const HloComputation*, const PartitionedComputation*>
      computation_to_partitioning_;
};

// Returns an MLIR function declaration for the given subgraph. For subgraphs of
// fusions, the signature is:
//   (ptr, ptr, ..., index, index, ...) -> element type(s)
// For subgraphs of called computations, the signature is:
//   (elemen type, ...) -> element type(s)
//
// Subgraphs of fusions will also have range (xla.range = [lower_bound,
// upper_bound], both bounds are inclusive) annotations on their index
// arguments.
mlir::func::FuncOp CreateSubgraphMlirFunction(
    const PartitionedComputation::Subgraph& subgraph,
    mlir::ImplicitLocOpBuilder& b);

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_COMPUTATION_PARTITIONER_H_
