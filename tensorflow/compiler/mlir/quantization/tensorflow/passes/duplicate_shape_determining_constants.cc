/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <iterator>
#include <memory>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

// Required to use LLVM_DEBUG macro.
#define DEBUG_TYPE "quant-duplicate-shape-determining-constants"

namespace mlir {
namespace quant {
namespace {

// This pass duplicates constants that affect or determine the shape of a tensor
// after being used in a computation for some op. Some specific operands of TF
// ops (like the `dim` argument for `TF::ExpandDimsOp`) determine the shape of
// the resulting tensor. If these operands are constants, they are duplicated
// and replace the shape-determining operands. Each duplicated constant will
// only be used as the shape-determining operand; it will not replace other
// usages of the original constant. If the operands are not constants (i.e.
// results of some other computation), then the pass recursively traverses the
// call tree upwards and duplicates all constants found in the subtree in a
// similar manner.
//
// This pass may be used to avoid placing shape-determining constants in the CPU
// graph and pass them as arguments to the TPU graph (via `TPUPartitionedCall`).
// If this happens, the XLA compiler cannot recognize such arguments as
// constants and may result in an error.
//
// A set of predefined ops and operand indices is used to determine whether an
// operand is a target for constant duplication.
class DuplicateShapeDeterminingConstantsPass
    : public PassWrapper<DuplicateShapeDeterminingConstantsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      DuplicateShapeDeterminingConstantsPass)

  StringRef getArgument() const final {
    return "quant-duplicate-shape-determining-constants";
  }

  StringRef getDescription() const final {
    return "Duplicates shape-determining constants. A shape-determining "
           "constant is a constant that are transitively used to change or "
           "determine the shape of a tensor. For example, the second argument "
           "'dim' to TF::ExpandDimsOp specifies the dimension index to expand.";
  }

  void runOnOperation() override;
};

// Returns True iff the otuput value of `op` is either a compile time constant
// or bounded from the XLA compiler's perspective, even if it is not a
// `ConstOp`.
bool IsOutputCompileTimeConstantOrBounded(Operation* op) {
  return llvm::isa_and_nonnull<TF::ShapeOp, TF::ShapeNOp, TF::RankOp,
                               TF::SizeOp, TF::TensorArraySizeV3Op,
                               TF::XlaSetBoundOp>(op);
}

// Recursively duplicate constants for `op_operands` upward.
void RecursivelyDuplicateConstantsForOperands(
    llvm::ArrayRef<OpOperand*> op_operands) {
  // Target operands to duplicate if it is a ConstOp.
  llvm::SmallVector<OpOperand*, 4> duplication_targets{op_operands.begin(),
                                                       op_operands.end()};

  int target_idx = 0;
  while (target_idx < duplication_targets.size()) {
    OpOperand* curr_operand = duplication_targets[target_idx];
    target_idx++;

    Operation* owning_op = curr_operand->getOwner();
    Operation* defining_op = curr_operand->get().getDefiningOp();

    if (llvm::isa_and_nonnull<TF::ConstOp>(defining_op)) {
      // No need to clone if this is the only use.
      if (defining_op->hasOneUse()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Not duplicating constant operand since it has only one "
                      "usage. Op: "
                   << curr_operand->getOperandNumber()
                   << ", operand idx: " << curr_operand->getOperandNumber()
                   << ", loc: " << owning_op->getLoc() << "\n");
        continue;
      }

      mlir::OpBuilder builder{owning_op->getContext()};
      builder.setInsertionPointAfter(defining_op);
      auto const_op_cloned = builder.clone(*defining_op);

      // Replace the operand with the duplicated op.
      owning_op->setOperand(curr_operand->getOperandNumber(),
                            const_op_cloned->getResult(0));

      LLVM_DEBUG(llvm::dbgs()
                 << "Duplicated constant operand from: "
                 << owning_op->getName().getStringRef()
                 << ", operand idx: " << curr_operand->getOperandNumber()
                 << ", loc: " << const_op_cloned->getLoc() << "\n");
    } else if (IsOutputCompileTimeConstantOrBounded(defining_op)) {
      // Stop the recursion early when the output of the defining op is
      // considered compile-time constant from the XLA compiler's perspective.
      continue;
    } else if (!defining_op) {
      // One example for this case is when `curr_operand` is a function
      // argument.
      owning_op->emitWarning()
          << "Operand idx (zero-based): " << curr_operand->getOperandNumber()
          << " does not have a defining op and cannot be duplicated.";
    } else {
      // If the operand's defining is not a ConstOp, recursively traverse
      // "upwards" to find ConstOps that transitively produces the current
      // operand and duplicate them.
      auto op_operands = defining_op->getOpOperands();
      absl::c_transform(
          op_operands, std::back_inserter(duplication_targets),
          [](OpOperand& op_operand) -> OpOperand* { return &op_operand; });
    }
  }
}

// Evaluate `operand_idx` w.r.t. `op`'s operands. If `operand_idx` is a positive
// number or a zero, it is returned as it is. If it is a negative number, it
// means it is counting backwards and will return the zero-based operand index
// for `op`.
//
// `operand_idx` should be within the range: [-num_operands, num_operands - 1].
int EvaluateOperandIdx(const int operand_idx, Operation& op) {
  if (operand_idx < 0) {
    // Calculate the actual index if a negative value is provided for
    // `operand_idx`.
    return op.getNumOperands() + operand_idx;
  }
  return operand_idx;
}

// Returns the pointers to operands at `operand_indices` of `op`.
llvm::SmallVector<OpOperand*> GetOperands(Operation& op,
                                          llvm::ArrayRef<int> operand_indices) {
  llvm::SmallVector<OpOperand*> operands{};
  for (const int operand_idx : operand_indices) {
    const int evaluated_operand_idx = EvaluateOperandIdx(operand_idx, op);
    operands.emplace_back(&op.getOpOperand(evaluated_operand_idx));
  }

  return operands;
}

// Represents an op type and its operand indices that should be "compile time
// constant" from the XLA compiler's point of view.
template <typename OpT, int... OperandIdx>
struct CompileTimeConstantOperand {
  static_assert(
      sizeof...(OperandIdx) > 0,
      "CompileTimeConstantOperand should have at least one operand index.");

  using OpType = OpT;

  // Returns the indices of operands that should be compile time constants.
  static constexpr std::array<int, sizeof...(OperandIdx)> OperandIndices() {
    return {OperandIdx...};
  }
};

// Finds all op of type `T::OpType` `func_op` and recursively duplicates
// constants used at the op's operands at `T::OperandIndices()`. It sequentially
// does the same thing for `Ts`.
template <typename T, typename... Ts>
void DuplicateShapeDeterminingConstants(func::FuncOp func_op) {
  for (auto op : func_op.getOps<typename T::OpType>()) {
    RecursivelyDuplicateConstantsForOperands(
        GetOperands(*op, T::OperandIndices()));
  }

  // Do the same thing for the rest of `Ts`.
  if constexpr (sizeof...(Ts) != 0) {
    DuplicateShapeDeterminingConstants<Ts...>(func_op);
  }
}

void DuplicateShapeDeterminingConstantsPass::runOnOperation() {
  func::FuncOp func_op = getOperation();

  DuplicateShapeDeterminingConstants<
      // go/keep-sorted start
      CompileTimeConstantOperand<TF::AllToAllOp, 1>,  // $group_assignment
      CompileTimeConstantOperand<TF::ArgMaxOp, 1>,    // $dimension
      CompileTimeConstantOperand<TF::ArgMinOp, 1>,    // $dimension
      // $orig_input_shape
      CompileTimeConstantOperand<TF::AvgPool3DGradOp, 0>,
      // $orig_input_shape
      CompileTimeConstantOperand<TF::AvgPoolGradOp, 0>,
      // $block_shape, $crops
      CompileTimeConstantOperand<TF::BatchToSpaceNDOp, 1, 2>,
      CompileTimeConstantOperand<TF::BatchToSpaceOp, 1>,      // $crops
      CompileTimeConstantOperand<TF::BincountOp, 1>,          // $size
      CompileTimeConstantOperand<TF::BroadcastArgsOp, 0, 1>,  // $s0, $s1
      // $s0, $s1
      CompileTimeConstantOperand<TF::BroadcastGradientArgsOp, 0, 1>,
      CompileTimeConstantOperand<TF::BroadcastToOp, 1>,  // $shape
      /// $group_assignment
      CompileTimeConstantOperand<TF::CollectiveAssignGroupV2Op, 0>,
      // $source_target_pairs
      CompileTimeConstantOperand<TF::CollectivePermuteOp, 1>,
      // $group_size, $group_key
      CompileTimeConstantOperand<TF::CollectiveReduceV2Op, 1, 2>,
      CompileTimeConstantOperand<TF::ConcatV2Op, -1>,  // (variadic) $axis
      // $filter_sizes
      CompileTimeConstantOperand<TF::Conv2DBackpropFilterOp, 1>,
      CompileTimeConstantOperand<TF::Conv2DBackpropInputOp, 0>,  // $input_sizes
      // $filter_sizes
      CompileTimeConstantOperand<TF::Conv3DBackpropFilterV2Op, 1>,
      // $input_sizes
      CompileTimeConstantOperand<TF::Conv3DBackpropInputV2Op, 0>,
      // $group_assignment
      CompileTimeConstantOperand<TF::CrossReplicaSumOp, 1>,
      CompileTimeConstantOperand<TF::CumprodOp, 1>,              // $axis
      CompileTimeConstantOperand<TF::CumsumOp, 1>,               // $axis
      CompileTimeConstantOperand<TF::CumulativeLogsumexpOp, 1>,  // $axis
      // $filter_sizes
      CompileTimeConstantOperand<TF::DepthwiseConv2dNativeBackpropFilterOp, 1>,
      // $input_sizes
      CompileTimeConstantOperand<TF::DepthwiseConv2dNativeBackpropInputOp, 0>,
      CompileTimeConstantOperand<TF::EmptyOp, 0>,  // $shape
      // $element_shape, $max_num_elements
      CompileTimeConstantOperand<TF::EmptyTensorListOp, 0, 1>,
      CompileTimeConstantOperand<TF::ExpandDimsOp, 1>,   // $dim
      CompileTimeConstantOperand<TF::FillOp, 0>,         // $dims
      CompileTimeConstantOperand<TF::GatherV2Op, 2>,     // $axis
      CompileTimeConstantOperand<TF::IRFFT2DOp, 1>,      // $fft_length
      CompileTimeConstantOperand<TF::IRFFT3DOp, 1>,      // $fft_length
      CompileTimeConstantOperand<TF::IRFFTOp, 1>,        // $fft_length
      CompileTimeConstantOperand<TF::InTopKV2Op, 2>,     // $k
      CompileTimeConstantOperand<TF::LinSpaceOp, 2>,     // $num
      CompileTimeConstantOperand<TF::ListDiffOp, 0, 1>,  // $x, $y
      // $k, $padding_value
      CompileTimeConstantOperand<TF::MatrixDiagPartV3Op, 1, 2>,
      // $k, $num_rows, $num_cols, $padding_value
      CompileTimeConstantOperand<TF::MatrixDiagV2Op, 1, 2, 3, 4>,
      // $k, $num_rows, $num_cols, $padding_value
      CompileTimeConstantOperand<TF::MatrixDiagV3Op, 1, 2, 3, 4>,
      CompileTimeConstantOperand<TF::MatrixSetDiagV2Op, 2>,  // $k
      CompileTimeConstantOperand<TF::MatrixSetDiagV3Op, 2>,  // $k
      CompileTimeConstantOperand<TF::MaxOp, 1>,  // $reduction_indices
      // $ksize, $strides
      CompileTimeConstantOperand<TF::MaxPoolGradGradV2Op, 3, 4>,
      // $ksize, $strides
      CompileTimeConstantOperand<TF::MaxPoolGradV2Op, 2, 3>,
      CompileTimeConstantOperand<TF::MaxPoolV2Op, 1, 2>,   // $ksize, $strides
      CompileTimeConstantOperand<TF::MeanOp, 1>,           // $reduction_indices
      CompileTimeConstantOperand<TF::MirrorPadGradOp, 1>,  // $paddings
      CompileTimeConstantOperand<TF::MirrorPadOp, 1>,      // $paddings
      CompileTimeConstantOperand<TF::MultinomialOp, 1>,    // $num_samples
      // $max_output_size
      CompileTimeConstantOperand<TF::NonMaxSuppressionV3Op, 2>,
      // $max_output_size
      CompileTimeConstantOperand<TF::NonMaxSuppressionV4Op, 2>,
      CompileTimeConstantOperand<TF::OneHotOp, 1>,  // $depth
      CompileTimeConstantOperand<TF::PadOp, 1>,     // $paddings
      CompileTimeConstantOperand<TF::PadV2Op, 1>,   // $paddings
      // $shape
      CompileTimeConstantOperand<TF::ParameterizedTruncatedNormalOp, 0>,
      CompileTimeConstantOperand<TF::RFFT2DOp, 1>,                // $fft_length
      CompileTimeConstantOperand<TF::RFFT3DOp, 1>,                // $fft_length
      CompileTimeConstantOperand<TF::RFFTOp, 1>,                  // $fft_length
      CompileTimeConstantOperand<TF::RandomStandardNormalOp, 0>,  // $shape
      CompileTimeConstantOperand<TF::RandomUniformIntOp, 0>,      // $shape
      CompileTimeConstantOperand<TF::RandomUniformOp, 0>,         // $shape
      // $start, $limit, $delta
      CompileTimeConstantOperand<TF::RangeOp, 0, 1, 2>,
      CompileTimeConstantOperand<TF::ReshapeOp, 1>,                // $shape
      CompileTimeConstantOperand<TF::ResizeBilinearOp, 1>,         // $size
      CompileTimeConstantOperand<TF::ResizeNearestNeighborOp, 1>,  // $size
      // $begin, $end, $strides
      CompileTimeConstantOperand<TF::ResourceStridedSliceAssignOp, 1, 2, 3>,
      CompileTimeConstantOperand<TF::ReverseOp, 1>,        // $dims
      CompileTimeConstantOperand<TF::ReverseV2Op, 1>,      // $axis
      CompileTimeConstantOperand<TF::ScatterNdOp, 2>,      // $shape
      CompileTimeConstantOperand<TF::SegmentSumV2Op, 2>,   // $num_segments
      CompileTimeConstantOperand<TF::SliceOp, 1, 2>,       // $begin, $size
      CompileTimeConstantOperand<TF::SparseToDenseOp, 1>,  // $output_shape
      CompileTimeConstantOperand<TF::StackV2Op, 0>,        // $max_size
      // $num_samples
      CompileTimeConstantOperand<TF::StatelessMultinomialOp, 1>,
      // $shape, $begin, $end, $strides
      CompileTimeConstantOperand<TF::StridedSliceGradOp, 0, 1, 2, 3>,
      // $begin, $end, $strides
      CompileTimeConstantOperand<TF::StridedSliceOp, 1, 2, 3>,
      CompileTimeConstantOperand<TF::SumOp, 1>,  // $reduction_indices
      CompileTimeConstantOperand<TF::TensorArraySplitV3Op, 2>,  // $lengths
      CompileTimeConstantOperand<TF::TensorArrayV3Op, 0>,       // $size
      // $element_shape
      CompileTimeConstantOperand<TF::TensorListFromTensorOp, 1>,
      // $element_shape, $num_elements
      CompileTimeConstantOperand<TF::TensorListReserveOp, 0, 1>,
      // $begin, $end, $strides
      CompileTimeConstantOperand<TF::TensorStridedSliceUpdateOp, 1, 2, 3>,
      CompileTimeConstantOperand<TF::TileOp, 1>,                // $multiples
      CompileTimeConstantOperand<TF::TopKV2Op, 1>,              // $k
      CompileTimeConstantOperand<TF::TransposeOp, 1>,           // $perm
      CompileTimeConstantOperand<TF::TruncatedNormalOp, 0>,     // $shape
      CompileTimeConstantOperand<TF::UnsortedSegmentMaxOp, 2>,  // $num_segments
      CompileTimeConstantOperand<TF::UnsortedSegmentMinOp, 2>,  // $num_segments
      CompileTimeConstantOperand<TF::UnsortedSegmentSumOp, 2>,  // $num_segments
      // $broadcast_dims
      CompileTimeConstantOperand<TF::XlaBroadcastHelperOp, 2>,
      // $window_strides, $padding, $lhs_dilation, $rhs_dilation,
      // $feature_group_count
      CompileTimeConstantOperand<TF::XlaConvOp, 2, 3, 4, 5, 6>,
      // $window_strides, $padding, $lhs_dilation, $rhs_dilation,
      // $feature_group_count
      CompileTimeConstantOperand<TF::XlaConvV2Op, 2, 3, 4, 5, 6>,
      CompileTimeConstantOperand<TF::XlaDynamicSliceOp, 2>,  // $slice_indices
      CompileTimeConstantOperand<TF::XlaGatherOp, 2>,        // $slice_sizes
      // $padding_low, $padding_high, $padding_interior
      CompileTimeConstantOperand<TF::XlaPadOp, 2, 3, 4>,
      // $window_dimensions, $window_strides, $base_dilations,
      // $window_dilations, $padding
      CompileTimeConstantOperand<TF::XlaReduceWindowOp, 2, 3, 4, 5, 6>,
      // $dim_index
      CompileTimeConstantOperand<TF::XlaRemoveDynamicDimensionSizeOp, 1>,
      // $window_dimensions, $window_strides, $padding
      CompileTimeConstantOperand<TF::XlaSelectAndScatterOp, 1, 2, 3>,
      CompileTimeConstantOperand<TF::XlaSetBoundOp, 1>,  // $bound
      // $dim_index
      CompileTimeConstantOperand<TF::XlaSetDynamicDimensionSizeOp, 1>
      // go/keep-sorted end
      >(func_op);
}

static PassRegistration<DuplicateShapeDeterminingConstantsPass> pass{};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDuplicateShapeDeterminingConstantsPass() {
  return std::make_unique<DuplicateShapeDeterminingConstantsPass>();
}

}  // namespace quant
}  // namespace mlir
