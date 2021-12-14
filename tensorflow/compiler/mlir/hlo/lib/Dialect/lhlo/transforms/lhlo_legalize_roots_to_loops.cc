/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO dialect to Affine dialect.
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/fusion_utils.h"
#include "mlir-hlo/Dialect/lhlo/transforms/lhlo_elemental_utils.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/utils/codegen_utils.h"
#include "mlir-hlo/utils/placement_utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using mlir::codegen_utils::calcMultiDimIndex;

namespace mlir {
namespace lmhlo {

#define GEN_PASS_CLASSES
#include "mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.h.inc"

namespace {

template <typename LHLO_OpTy>
LogicalResult elemwiseLowerHelper(
    OpBuilder& b, Location loc, Operation* op, Value output_linear_index,
    const ShapeConstraintAnalysis* shape_constraint_analysis) {
  if (!isa<LHLO_OpTy>(op) || !op->hasTrait<mlir::OpTrait::Elementwise>())
    return failure();

  Value result_memref = cast<LmhloOp>(op).getResultBuffer();
  Value memref = result_memref;
  if (shape_constraint_analysis) {
    Value leader_memref =
        shape_constraint_analysis->GetLeaderValueWithSameShape(result_memref);
    if (leader_memref != nullptr) memref = leader_memref;
  }
  // TODO(disc): Replace with memref.Delinearize
  auto multidim_index = calcMultiDimIndex(b, loc, output_linear_index, memref);
  SmallVector<Value, 4> operand_values;
  for (Value operand_memref : op->getOperands().drop_back()) {
    Value operand_data = createLoadOrUseCachedValue(
        loc, &b, operand_memref, multidim_index, b.saveInsertionPoint());
    operand_values.push_back(operand_data);
  }
  auto res = LhloOpToStdScalarOp::map<LHLO_OpTy>(
      llvm::cast<LHLO_OpTy>(op),
      result_memref.getType().cast<MemRefType>().getElementType(),
      operand_values, &b);
  createOffsetStore(b, loc, res, result_memref, output_linear_index);
  return success();
}

template <typename LHLO_OpTy>
LogicalResult miscLowerHelper(
    OpBuilder& b, Location loc, Operation* opaque_op, Value output_linear_index,
    const ShapeConstraintAnalysis* shape_constraint_analysis) {
  LHLO_OpTy op = dyn_cast<LHLO_OpTy>(opaque_op);
  if (!op) return failure();
  Value result_memref = cast<LmhloOp>(&*op).getResultBuffer();
  Value memref = result_memref;
  if (shape_constraint_analysis) {
    Value leader_memref =
        shape_constraint_analysis->GetLeaderValueWithSameShape(result_memref);
    if (leader_memref != nullptr) {
      memref = leader_memref;
    }
  }
  llvm::SmallVector<Value, 4> output_multidim_index =
      calcMultiDimIndex(b, loc, output_linear_index, memref);
  Value operand_data = elementalLower(&b, loc, op, output_multidim_index,
                                      /*check_cache=*/true);
  createOffsetStore(b, loc, operand_data, result_memref, output_linear_index);
  return success();
}

template <typename First>
LogicalResult elemwiseLowerHelperOr(
    OpBuilder& b, Location loc, Operation* op, Value output_linear_index,
    const ShapeConstraintAnalysis* shape_constraint_analysis) {
  return elemwiseLowerHelper<First>(b, loc, op, output_linear_index,
                                    shape_constraint_analysis);
}

template <typename First, typename Second, typename... Rest>
LogicalResult elemwiseLowerHelperOr(
    OpBuilder& b, Location loc, Operation* op, Value output_linear_index,
    const ShapeConstraintAnalysis* shape_constraint_analysis) {
  return success(
      succeeded(elemwiseLowerHelperOr<First>(b, loc, op, output_linear_index,
                                             shape_constraint_analysis)) ||
      succeeded(elemwiseLowerHelperOr<Second, Rest...>(
          b, loc, op, output_linear_index, shape_constraint_analysis)));
}

LogicalResult lowerHelper(
    OpBuilder& b, Location loc, Operation* op, Value output_linear_index,
    const ShapeConstraintAnalysis* shape_constraint_analysis) {
  if (succeeded(
          elemwiseLowerHelperOr<
#define GET_SUPPORTED_OP_LIST
#include "mlir-hlo/utils/disc_supported_list.h.inc"
              >(b, loc, op, output_linear_index, shape_constraint_analysis)) ||
      // TODO(disc): Upstream is on the way for more Ops
      succeeded(miscLowerHelper<RealDynamicSliceOp>(
          b, loc, op, output_linear_index, shape_constraint_analysis)) ||
      succeeded(miscLowerHelper<DynamicBroadcastInDimOp>(
          b, loc, op, output_linear_index, shape_constraint_analysis)) ||
      succeeded(miscLowerHelper<BroadcastInDimOp>(
          b, loc, op, output_linear_index, shape_constraint_analysis))) {
    return success();
  }
  return failure();
}

// we don't do inbound check for kLoop Schedule
// LoopSplit pass will do this.
//
/* %num_elements = ElementsIn(root_shape)
 * loop.for %idx = 0 to %num_elements step 1 {
 *   %multidim_indices_0..n = getMultidimIndices(%idx);
 *   %operand_0 = load %operand0[]
 *   %operand_1 = load %operand1[]
 *   emit calculation..
 * }
 */
LogicalResult lowerWithScheduleLoop(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false, bool parallel_loop = true,
    const ShapeConstraintAnalysis* shape_constraint_analysis = nullptr) {
  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  auto zero = b.create<arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));
  auto one = b.create<arith::ConstantOp>(loc, b.getIndexType(),
                                         b.getIntegerAttr(b.getIndexType(), 1));
  auto num_elements =
      codegen_utils::emitNumElementsComputation(b, loc, dominant_op);
  Value var;
  if (parallel_loop) {
    SmallVector<Value, 2> vars;
    (void)createParallelAndSetInsPt(b, loc, vars, {zero}, {num_elements}, {one},
                                    {});
    var = vars[0];
  } else {
    (void)createLoopAndSetInsPt(b, loc, var, zero, num_elements, one, {});
  }
  for (Operation* root_op : root_ops) {
    if (failed(lowerHelper(b, loc, root_op, var, shape_constraint_analysis)))
      return failure();
  }
  // remove the root_op if it has no other users except the memref
  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  } else {
    assert(parent != nullptr && "Parent must be provided for fusion lowering");
    cleanUnusedLhloOps(parent);
  }
  return success();
}

bool isOnGpu(Operation* op) {
  if (isa<FusionOp>(op))
    // TODO(disc): Revisit this when fusion on cpu is suppported
    return true;
  assert(isa<LmhloOp>(op) && "Unexpected usage of isOnGpu");
  auto result_memref = cast<LmhloOp>(op).getResultBuffer();
  auto memory_space =
      result_memref.getType().cast<MemRefType>().getMemorySpace();
  return memory_space && memory_space.isa<StringAttr>() &&
         memory_space.cast<StringAttr>().getValue() ==
             mhlo::placement_utils::c_gpu;
}

}  // namespace

// Expand the root ops in a fused func into a parrallel loop or a set of
// nested loops. This pass must be executed after the fusion pass, and works
// together with the InputInlineFusion pass after it for fusion codegen.
//
// TODO(disc): Currently this pass supports lmhlo.FusionOp to have lmhlo ops
// inside, not mhlo. It's mainly because we now do fusion on lmhlo, not mhlo.
// The fusion pass can be moved to mhlo after shape dialect is brought in to
// represent shape calculation on tensor layer, and we would be able to do shape
// calculation lowering for mhlo.FusionOp. Reconsider the fusion representation
// after these are done, a lmhlo.FusionOp with mhlo inside would be more
// friendly to the legacy FusedIrEmitter.
class LhloLegalizeRootsToParallelLoops
    : public LhloLegalizeRootsToParallelLoopsPassBase<
          LhloLegalizeRootsToParallelLoops> {
  void runOnFunction() override {
    auto func = getFunction();
    OpBuilder b(func);
    SmallVector<Operation*, 4> gpu_non_fusion_worklist;
    SmallVector<Operation*, 4> cpu_non_fusion_worklist;
    SmallVector<Operation*, 4> gpu_fusion_worklist;
    for (mlir::Operation& op : func.body().getOps()) {
      if (isa<FusionOp>(&op)) {
        // TODO(disc): Revisit this when fusion on cpu is supported
        gpu_fusion_worklist.push_back(&op);
      } else if (isa<LmhloOp>(&op)) {
        if (isOnGpu(&op))
          gpu_non_fusion_worklist.push_back(&op);
        else
          cpu_non_fusion_worklist.push_back(&op);
      }
    }

    for (Operation* op : cpu_non_fusion_worklist) {
      // Only for calculating shapes when the backend is gpu. A simple schedule
      // should be sufficient for performance.
      // TODO(disc): Revisit this when the backend is cpu and the calculation is
      // for data.
      if (failed(lowerWithScheduleLoop({op}, op, nullptr,
                                       /*non_fusion=*/true,
                                       /*parallel_loop=*/false))) {
        op->emitError() << "failed to lower to loops";
        signalPassFailure();
        return;
      }
    }

    for (Operation* op : gpu_non_fusion_worklist) {
      // TODO(disc): single nodes with non kLoop schedule like ReduceOp
      // is not implemented yet. Currently ReduceOp is lowered with loop
      // schedule, which means for poor performance.
      if (failed(lowerWithScheduleLoop({op}, op, nullptr,
                                       /*non_fusion=*/true,
                                       /*parallel_loop=*/true))) {
        op->emitError() << "failed to lower to loops";
        signalPassFailure();
        return;
      }
    }

    for (Operation* fusion : gpu_fusion_worklist) {
      auto fusion_op = cast<FusionOp>(fusion);
      FusionPattern fusion_pattern(fusion_op);
      auto root_ops = fusion_pattern.getRootOps();
      auto fused_block = &(fusion_op.region().front());
      SmallVector<Operation*, 4> op_list;
      fused_block->walk(
          [&](LmhloOp op) { op_list.push_back(op.getOperation()); });
      ShapeConstraintAnalysis shape_constraint_analysis(op_list);

      // No need to do codegen, return directly.
      if (root_ops.empty()) {
        return;
      }
      // Make a loop to write the buffer into init value for each
      // ColReduction root. This will be further lowered to a init_kernel
      // TODO(disc): Code upstream is on the way
      // maybeEmitInitLoops(b, root_ops);

      // 1, If any reduce op among the 'root_ops', follow the schedule of it;
      //    or else, follow the schedule of kLoop.
      // 2, If there are a mixer of column reductions and row reductions,
      //    follow the schedule of the row reduction, and implement all the
      //    column reduction with the 'pure atomic' way, which has no
      //    requirement on the schedule.
      // TODO(disc): the support of row reduction and 'pure atomic' reduction
      auto fusion_type = fusion_pattern.getFusionType();
      auto dominant_op = fusion_pattern.getDominantOp();
      switch (fusion_type) {
        case FusionType::kRowReduction:
          dominant_op->emitError() << "Unsupported kRowReduction Schedule";
          signalPassFailure();
          return;

        case FusionType::kColReduction:
          dominant_op->emitError() << "Unsupported kColReduction Schedule";
          signalPassFailure();
          return;

        case FusionType::kLoop:
          if (failed(lowerWithScheduleLoop(root_ops, dominant_op, fused_block,
                                           /*non_fusion*/ false,
                                           /*parallel_loop*/ true,
                                           &shape_constraint_analysis))) {
            dominant_op->emitError() << "failed to lower to loops";
            signalPassFailure();
            return;
          }
          break;
        default:
          dominant_op->emitError() << "Unknown fusion type";
          signalPassFailure();
          return;
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
createLhloLegalizeRootsToParallelLoopsPass() {
  return std::make_unique<LhloLegalizeRootsToParallelLoops>();
}

}  // namespace lmhlo
}  // namespace mlir
