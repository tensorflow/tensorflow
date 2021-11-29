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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_

#include "absl/types/optional.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace mlir {

// This class will process an HloModule with the supplied BufferAssignment and
// populate the MLIR ModuleOp with the computation converted in the LHLO
// dialect.
class LhloDialectEmitter : public xla::ConstDfsHloVisitorWithDefault {
 public:
  // Initializes internal data structures. It must be called before calling any
  // of the visitors.
  tensorflow::Status Initialize();

  LhloDialectEmitter(const xla::BufferAssignment& assignment,
                     const xla::HloComputation& computation, ModuleOp module)
      : assignment_(std::move(assignment)),
        computation_(computation),
        module_(module),
        builder_(module.getContext()),
        i8_type_(builder_.getIntegerType(8)) {}

  xla::StatusOr<mlir::Operation*> EmitOp(const xla::HloInstruction* instr);

  static xla::StatusOr<mhlo::ScatterDimensionNumbersAttr>
  GetScatterDimensionNumbers(const xla::HloInstruction* instr,
                             mlir::MLIRContext* context);

 private:
  xla::StatusOr<lmhlo::SortOp> EmitSortOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::FusionOp> EmitFusionOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::ScatterOp> EmitScatterOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::SelectAndScatterOp> EmitSelectAndScatterOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<Operation*> EmitCustomCallOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::CholeskyOp> EmitCholesky(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitGemm(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitDnnConvolution(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitDnnBatchNorm(
      const xla::HloCustomCallInstruction* custom_call);

  xla::StatusOr<memref::GetGlobalOp> EmitConstant(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::InfeedOp> EmitInfeedOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::OutfeedOp> EmitOutfeedOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::AllToAllOp> EmitAllToAllOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::AllGatherOp> EmitAllGatherOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::AllReduceOp> EmitAllReduceOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::AllReduceStartOp> EmitAllReduceStartOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::AllReduceDoneOp> EmitAllReduceDoneOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::ReduceScatterOp> EmitReduceScatterOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::CollectivePermuteOp> EmitCollectivePermuteOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::RngGetAndUpdateStateOp> EmitRngGetAndUpdateStateOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::FftOp> EmitFftOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::TriangularSolveOp> EmitTriangularSolveOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<Operation*> EmitBitcast(const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::CaseOp> EmitCaseOp(const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::WhileOp> EmitWhileOp(const xla::HloInstruction* instr);

  xla::Status ImportAsLmhloRegion(xla::HloComputation* computation,
                                  mlir::Region* region);

  // Since LMHLO dialect does not define token types, this enum controls how
  // token operand/results from XLA:HLO are lowered to MLIR.
  enum class TokenLoweringMode {
    kFailToLower,  // Fail lowering if token inputs are encountered.
    kUseNull,      // Use a null Value in the operand list for each token.
    // kSkip,        // Skip any token inputs or outputs (not yet needed)
  };

  // Create LHLO operation operands given an XLA HLO instruction. By default,
  // all XLA HLO operands and results are converted to MLIR and appended to
  // `operands`. If `num_operands` is specified, only the first `num_operand`
  // operands of the instruction are converted to MLIR. The function returns the
  // actual number of operands and results generated for MLIR in `num_arguments`
  // and `num_results`.
  xla::Status CreateOperands(const xla::HloInstruction* instr,
                             absl::optional<int64_t> num_operands,
                             TokenLoweringMode token_mode,
                             SmallVectorImpl<Value>& operands,
                             size_t& num_arguments, size_t& num_results);

  template <typename OpType>
  xla::StatusOr<OpType> CreateOpWithoutAttrs(
      const xla::HloInstruction* instr,
      absl::optional<int64_t> num_operands = absl::nullopt) {
    size_t unused;
    return CreateOpWithoutAttrs<OpType>(instr, unused, unused, num_operands);
  }

  template <typename OpType>
  xla::StatusOr<OpType> CreateOpWithoutAttrs(
      const xla::HloInstruction* instr, size_t& num_arguments,
      size_t& num_results,
      absl::optional<int64_t> num_operands = absl::nullopt);

  template <typename OpType>
  OpType CreateOpWithoutAttrs(const xla::HloInstruction* instr,
                              ValueRange operands);

  xla::StatusOr<mlir::Operation*> CreateOpInFusion(
      const xla::HloInstruction* instr, ValueRange buffer_operands,
      size_t num_arguments, size_t num_results);

  xla::StatusOr<mlir::Operation*> CreateOpInFusion(
      const xla::HloInstruction* instr);

  template <typename T>
  DenseIntElementsAttr GetI64DenseElementsAttr(const T& container) {
    return builder_.getI64TensorAttr(
        {container.data(), static_cast<size_t>(container.size())});
  }

  DenseIntElementsAttr GetWindowElements(
      const xla::Window& window,
      std::function<int64_t(const xla::WindowDimension& dim)> getter) {
    llvm::SmallVector<int64_t, 4> elements;
    elements.reserve(window.dimensions_size());
    for (const xla::WindowDimension& dim : window.dimensions()) {
      elements.push_back(getter(dim));
    }
    return GetI64DenseElementsAttr(elements);
  }

  static mlir::DenseIntElementsAttr GetLayoutAttribute(
      const xla::Layout& layout, Builder* builder);

  tensorflow::Status DefaultAction(const xla::HloInstruction* instr) final;

  // Computation parameters don't need any specific handling when they are
  // visited, they are already processed when we enter a new computation.
  tensorflow::Status HandleParameter(const xla::HloInstruction* instr) final {
    return tensorflow::Status::OK();
  }

  // Helper function that recursively visits the tuple structure in
  // `current_shape`, and reconstruct a matching lmhlo::TupleOp.
  // Each leaf node is converted to an std.view op with corresponding offsets.
  // If no tuple presents, it simply returns a view of the buffer.
  tensorflow::Status GetOrCreateViewImpl(const xla::HloInstruction* instr,
                                         const xla::Shape& current_shape,
                                         xla::ShapeIndex* current_shape_index,
                                         SmallVectorImpl<Value>* values,
                                         TokenLoweringMode token_mode);

  // Helper function to create view/tuple of views to a buffer for a given
  // instruction result. `result_subset` can be used to for instructions that
  // have a tuple result and MLIR conversion needs to convert only one of the
  // tuple elements. Note that if needed, this can be extended to take a list of
  // ShapeIndex values in case we need finer control on what elements of the
  // output tuple to be converted to MLIR.
  tensorflow::Status GetOrCreateView(
      const xla::HloInstruction* instr, SmallVectorImpl<Value>* values,
      const xla::ShapeIndex& result_subset = {},
      TokenLoweringMode token_mode = TokenLoweringMode::kFailToLower);

  xla::StatusOr<Value> GetOrCreateArrayView(
      const xla::HloInstruction* instr, const xla::Shape& current_shape,
      const xla::ShapeIndex& current_shape_index);

  xla::StatusOr<Value> RewriteFusionOperand(const xla::HloInstruction* root,
                                            const xla::Shape& shape,
                                            xla::ShapeIndex* shape_index,
                                            OpBuilder* b, Location loc);

  // Return an MLIR location for an HLO instruction.
  Location getLocation(const xla::HloInstruction* inst) {
    return NameLoc::get(builder_.getIdentifier(inst->name()));
  }

  // This map provides access to MLIR buffers for each HLO buffer allocation.
  // The MLIR buffers are all `memref<{size}xi8>` and correspond to function
  // parameters. It is populated at the beginning of the processing with all
  // the buffer allocations and is unchanged afterward. Every HLOInstruction
  // is using a "slice" of the buffer allocation and providing shape, layout,
  // and Dtype. An MLIR view is used separately to model slices into the
  // allocations (see below).
  llvm::DenseMap<const xla::BufferAllocation*, Value> allocations_;

  // This map provides access to MLIR buffers for each HLO instruction, keyed
  // instruction identity. A slice is contained in a BufferAllocation, and has
  // an offset and a size.
  //
  // As for why we don't use HloInstruction*, see GetOrCreateView(), but
  // mostly we want to leverage better of the aliased buffers.
  //
  // If the HloInstruction is a tuple, all leaf nodes are stored flattened.
  // Otherwise, there will be a single buffer.
  //
  // An MLIR buffer is either an input parameter, or a ViewOp in the case
  // where the slice is only part of its allocation.
  //
  // `slices_` is populated lazily in the `GetOrCreateView()` helper as we
  // process every instruction.
  absl::flat_hash_map<std::pair<const xla::HloInstruction*, xla::ShapeIndex>,
                      Value>
      slices_;

  // The BufferAssignment computed by XLA ahead of time.
  const xla::BufferAssignment& assignment_;

  // The HLO module that will be converted.
  const xla::HloComputation& computation_;

  // This is the MLIR module in which a function will be created for every HLO
  // computation.
  ModuleOp module_;

  // The builder keeps track of the current insertion point in the MLIR
  // module.
  OpBuilder builder_;
  // Convenient "cached" access to this widely used MLIR type (i8).
  Type i8_type_;

  // Map all-reduce-start ops to their LHLO op, so we can connect the
  // all-reduce-done op with the correct token.
  absl::flat_hash_map<const xla::HloInstruction*, lmhlo_gpu::AllReduceStartOp>
      all_reduce_start_ops_;
};

// Populate the MLIR `module` with the computation from the `hlo_module` using
// the provided buffer `assignment`. The returned `Status` indicates success
// or failure in the conversion.
tensorflow::Status HloToLhloModule(const xla::BufferAssignment& assignment,
                                   const xla::HloModule& hlo_module,
                                   ModuleOp module);

tensorflow::Status OptimizeAndConvertHloToLmhlo(
    std::unique_ptr<xla::HloModule> hlo_module, ModuleOp module,
    StringRef platform_name, bool optimize_xla_hlo);
OwningModuleRef HloTextToLhloTranslateFunction(llvm::StringRef input,
                                               MLIRContext* context,
                                               bool optimize_xla_hlo);

// This register the MLIR pass with the command line.
void RegisterMhloToLhloWithXlaPass();

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
