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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {
namespace mlir_converter {

struct EpilogueSpecification {
  // Creates an epilogue with output indices matching the given root's shape.
  static EpilogueSpecification FromIdentityIndexing(
      const HloInstruction* hero, const HloInstruction* root,
      mlir::MLIRContext* mlir_context);
  // Creates an epilogue with the raw thread/block/symbol indices, as defined
  // by the fusion's thread->output mapping.
  static EpilogueSpecification FromOutputIndexing(
      const HloFusionAnalysis& analysis,
      const std::vector<const HloInstruction*>& heroes,
      const std::vector<const HloInstruction*>& roots,
      const KernelFusionInterface& fusion, mlir::MLIRContext* mlir_context);

  std::vector<const HloInstruction*> heroes;
  std::vector<const HloInstruction*> roots;

  // The ranges of the indices that the subgraph is called with.
  std::vector<int64_t> index_ranges;

  // Indexing maps for each root output. All maps must have the same number of
  // input dimensions.
  std::vector<mlir::AffineMap> root_indexing;
};

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
// There is a hooks to customize this partitioning:
// is_subgraph_root: forces the clusterer to start a new subgraph at a given
// instruction. The instruction is guaranteed to be a in a different subgraph
// than its users.
class PartitionedComputation {
 public:
  explicit PartitionedComputation(
      const HloComputation* computation, mlir::MLIRContext* mlir_context,
      std::function<bool(const HloInstruction*)> is_subgraph_root =
          [](const HloInstruction*) { return false; });

  struct Subgraph {
    // A unique name of the subgraph. Used for function names.
    std::string name;

    // The instructions that make up this subgraph.
    absl::flat_hash_set<const HloInstruction*> instructions;

    // The roots (return values of the function).
    std::vector<const HloInstruction*> roots;

    // The ranges of the indices that the subgraph is called with.
    std::vector<int64_t> index_ranges;

    // Maps from raw indices to root indices.
    std::vector<mlir::AffineMap> root_indexing;

    // For values that are function arguments (not function calls), stores
    // the mapping from value to the starting argument index. The arguments
    // always come after the tensor parameters and output indices; the indices
    // are relative to the argument after the last index argument.
    absl::flat_hash_map<const HloInstruction*, int> injected_value_starts;
    // The sum of the arity of the injected values.
    int num_injected_values = 0;

    std::string ToString() const;

    // Creates a subgraph for the given heroes' epilogue. The heroes values will
    // be injected into the subgraph.
    static Subgraph ForEpilogue(const EpilogueSpecification& epilogue);
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
  // Partition the given fusion computation and optionally generate an epilogue
  // for the given heroes.
  explicit PartitionedComputations(
      const HloComputation* fusion, mlir::MLIRContext* mlir_context,
      std::vector<EpilogueSpecification> epilogues = {});

  const PartitionedComputation& FindPartitionedComputation(
      const HloComputation* computation) const {
    return *computation_to_partitioning_.at(computation);
  }

  const PartitionedComputation::Subgraph& FindSubgraph(
      const HloInstruction* instr) const;

  absl::Span<const PartitionedComputation> partitioned_computations() const {
    return partitioned_computations_;
  }

  // If the fusion has an epilogue (i.e., the heroes are inside the fusion),
  // returns it.
  const std::vector<PartitionedComputation::Subgraph>& epilogues() const {
    return epilogues_;
  }

  const HloComputation* fusion() const { return fusion_; }

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
  const HloComputation* fusion_;
  std::vector<PartitionedComputation::Subgraph> epilogues_;
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
