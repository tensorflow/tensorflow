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

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace mlir {

// This class will process an HloModule with the supplied BufferAssignment and
// populate the MLIR ModuleOp with the computation converted in the LHLO
// dialect.
class LhloDialectEmitter : public ::xla::DfsHloVisitorWithDefault {
 public:
  // Initializes internal data structures. It must be called before calling any
  // of the visitors.
  tensorflow::Status Initialize();

  LhloDialectEmitter(const ::xla::BufferAssignment& assignment,
                     const ::xla::HloComputation& computation, ModuleOp module)
      : assignment_(std::move(assignment)),
        computation_(computation),
        module_(module),
        builder_(module.getContext()),
        i8_type_(builder_.getIntegerType(8)) {}

  ::xla::StatusOr<lmhlo::SortOp> EmitSortOp(::xla::HloInstruction* instr);
  ::xla::StatusOr<lmhlo::FusionOp> EmitFusionOp(::xla::HloInstruction* instr);
  ::xla::StatusOr<lmhlo::ScatterOp> EmitScatterOp(::xla::HloInstruction* instr);
  ::xla::StatusOr<mhlo::ScatterDimensionNumbers> GetScatterDimensionNumbers(
      ::xla::HloInstruction* instr);

 private:
  template <typename OpType>
  ::xla::StatusOr<OpType> CreateOpWithoutAttrs(::xla::HloInstruction* instr);

  template <typename T>
  DenseIntElementsAttr getI64DenseElementsAttr(const T& container) {
    return builder_.getI64TensorAttr(
        {container.data(), static_cast<size_t>(container.size())});
  }

  tensorflow::Status DefaultAction(::xla::HloInstruction* instr) final;

  // Computation parameters don't need any specific handling when they are
  // visited, they are already processed when we enter a new computation.
  tensorflow::Status HandleParameter(::xla::HloInstruction* instr) final {
    return tensorflow::Status::OK();
  }

  tensorflow::Status HandleSort(::xla::HloInstruction* instr) final;
  tensorflow::Status HandleFusion(::xla::HloInstruction* instr) final;
  tensorflow::Status HandleScatter(::xla::HloInstruction* instr) final;

  // Helper function that recursively visits the tuple structure in
  // `current_shape`, and reconstruct a matching lmhlo::TupleOp.
  // Each leaf node is converted to an std.view op with corresponding offsets.
  // If no tuple presents, it simply returns a view of the buffer.
  tensorflow::Status GetOrCreateViewImpl(const ::xla::HloInstruction* instr,
                                         const ::xla::Shape& current_shape,
                                         ::xla::ShapeIndex* current_shape_index,
                                         SmallVectorImpl<Value>* values);

  // Helper function to create view/tuple of views to a buffer for a given
  // instruction result.
  tensorflow::Status GetOrCreateView(const ::xla::HloInstruction* instr,
                                     SmallVectorImpl<Value>* values);

  ::xla::StatusOr<Value> GetOrCreateArrayView(
      const ::xla::HloInstruction* instr, const ::xla::Shape& current_shape,
      const ::xla::ShapeIndex& current_shape_index);

  ::xla::StatusOr<Value> RewriteFusionOperand(const ::xla::HloInstruction* root,
                                              const ::xla::Shape& shape,
                                              ::xla::ShapeIndex* shape_index,
                                              OpBuilder* b, Location loc);

  // Return an MLIR location for an HLO instruction.
  Location getLocation(::xla::HloInstruction* inst) {
    return NameLoc::get(builder_.getIdentifier(inst->name()),
                        builder_.getContext());
  }

  // This map provides access to MLIR buffers for each HLO buffer allocation.
  // The MLIR buffers are all `memref<{size}xi8>` and correspond to function
  // parameters. It is populated at the beginning of the processing with all the
  // buffer allocations and is unchanged afterward. Every HLOInstruction is
  // using a "slice" of the buffer allocation and providing shape, layout, and
  // Dtype. An MLIR view is used separately to model slices into the allocations
  // (see below).
  llvm::DenseMap<const ::xla::BufferAllocation*, Value> allocations_;

  // This map provides access to MLIR buffers for each HLO instruction, keyed
  // instruction identity. A slice is contained in a BufferAllocation, and has
  // an offset and a size.
  //
  // As for why we don't use HloInstruction*, see GetOrCreateView(), but mostly
  // we want to leverage better of the aliased buffers.
  //
  // If the HloInstruction is a tuple, all leaf nodes are stored flattened.
  // Otherwise, there will be a single buffer.
  //
  // An MLIR buffer is either an input parameter, or a ViewOp in the case where
  // the slice is only part of its allocation.
  //
  // `slices_` is populated lazily in the `GetOrCreateView()` helper as we
  // process every instruction.
  absl::flat_hash_map<std::pair<const xla::HloInstruction*, xla::ShapeIndex>,
                      Value>
      slices_;

  // The BufferAssignment computed by XLA ahead of time.
  const ::xla::BufferAssignment& assignment_;

  // The HLO module that will be converted.
  const ::xla::HloComputation& computation_;

  // This is the MLIR module in which a function will be created for every HLO
  // computation.
  ModuleOp module_;

  // The builder keeps track of the current insertion point in the MLIR module.
  OpBuilder builder_;
  // Convenient "cached" access to this widely used MLIR type (i8).
  Type i8_type_;
};

// Populate the MLIR `module` with the computation from the `hlo_module` using
// the provided buffer `assignment`. The returned `Status` indicates success
// or failure in the conversion.
tensorflow::Status HloToLhloModule(const ::xla::BufferAssignment& assignment,
                                   const ::xla::HloModule& hlo_module,
                                   ModuleOp module);

OwningModuleRef HloTextToLhloTranslateFunction(llvm::StringRef input,
                                               mlir::MLIRContext* context);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
