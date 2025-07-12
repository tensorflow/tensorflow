/* Copyright 2023 The JAX Authors.

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/infer_vector_layout_extensions.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_INFERVECTORLAYOUTPASS
#define GEN_PASS_DEF_INFERVECTORLAYOUTPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

using ImplicitDim = VectorLayout::ImplicitDim;

static constexpr int kLayoutLog = 10;

bool is_fully_replicated(const Layout &layout) {
  static LayoutOffsets replicated_offsets = {std::nullopt, std::nullopt};
  return layout.has_value() && layout->offsets() == replicated_offsets;
}

TiledLayoutAttr getMemRefLayout(Value ref) {
  if (auto erase_op = ref.getDefiningOp<tpu::EraseLayoutOp>()) {
    ref = erase_op.getOperand();
  }
  return cast<TiledLayoutAttr>(cast<MemRefType>(ref.getType()).getLayout());
}

LogicalResult verifyDivisibleIndex(Value tiled_index, int64_t tiling, int dim,
                                   Operation *op) {
  if (!isGuaranteedDivisible(tiled_index, tiling)) {
    return op->emitOpError("cannot statically prove that index in dimension ")
           << dim << " is a multiple of " << tiling;
  }
  return success();
}

// TODO(apaszke): Test that this pass fills in NoLayout for all operations that
// have corresponding native instructions.
class VectorLayoutInferer {
 public:
  explicit VectorLayoutInferer(int hardware_generation,
                               std::array<int64_t, 2> target_shape,
                               const TpuTilingFlags &tpu_tiling_flags)
      : hardware_generation_(hardware_generation),
        target_shape_({target_shape[0], target_shape[1]}),
        default_tiling_(target_shape),
        tpu_tiling_flags_(tpu_tiling_flags) {}

#define TPU_CHECK_OP(cond, msg) \
  if (!(cond)) {                \
    op->emitOpError(msg);       \
    return failure();           \
  }

#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

  LogicalResult inferBlock(
      Block &block,
      const std::function<LogicalResult(Operation *)> &match_terminator) {
    for (Operation &any_op : block.without_terminator()) {
      VLOG(kLayoutLog) << Print(&any_op);
      if (any_op.hasAttr("in_layout") || any_op.hasAttr("out_layout")) {
        if (auto op = dyn_cast<tpu::AssumeLayoutOp>(any_op)) {
          TPU_CHECK_OP(
              any_op.hasAttr("in_layout") && any_op.hasAttr("out_layout"),
              "expect layout attributes in tpu::AssumeLayoutOp");
          continue;
        } else {
          any_op.emitOpError("layout attributes already attached");
          return failure();
        }
      }

      // TODO: b/342235360 - This check is temporary while we increase and test
      // support for offsets outside of the first tile. When support is more
      // broad, any op without support should check it within their own rule.
      if (!isa<vector::BroadcastOp, vector::ExtractStridedSliceOp>(any_op)) {
        const SmallVector<Layout> layouts_in = getLayoutFromOperands(&any_op);
        for (const Layout &layout : layouts_in) {
          if (layout &&
              layout->offsets()[1].value_or(0) >= layout->tiling()[1]) {
            force_first_tile_offsets_ = true;
          }
        }
      }

      bool has_vector_io = false;
      for (auto op : any_op.getOperands()) {
        has_vector_io |= isa<VectorType>(op.getType());
      }
      for (auto r : any_op.getResults()) {
        has_vector_io |= isa<VectorType>(r.getType());
      }
      if (!has_vector_io && any_op.getRegions().empty()) {
        SmallVector<Layout, 4> in_layout(any_op.getNumOperands(), kNoLayout);
        if (any_op.getNumResults() == 0) {
          setInLayout(&any_op, in_layout);
        } else if (any_op.getNumResults() == 1) {
          setLayout(&any_op, in_layout, kNoLayout);
        } else {
          any_op.emitOpError("Multi-result ops not supported");
          return failure();
        }
      } else if (isa<arith::ExtSIOp, tpu::ExtFOp>(any_op)) {
        if (inferExt(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::SIToFPOp>(any_op);
                 op &&
                 cast<VectorType>(op.getIn().getType())
                         .getElementTypeBitWidth() <
                     cast<VectorType>(op.getType()).getElementTypeBitWidth()) {
        if (inferExt(&any_op).failed()) {
          return failure();
        }
      } else if (isa<arith::TruncIOp, tpu::TruncFOp>(any_op)) {
        if (inferTrunc(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::FPToSIOp>(any_op);
                 op &&
                 cast<VectorType>(op.getOperand().getType())
                         .getElementTypeBitWidth() >
                     cast<VectorType>(op.getType()).getElementTypeBitWidth()) {
        if (inferTrunc(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::SelectOp>(any_op)) {
        auto true_ty = dyn_cast<VectorType>(op.getTrueValue().getType());
        auto false_ty = dyn_cast<VectorType>(op.getFalseValue().getType());
        TPU_CHECK_OP(static_cast<bool>(true_ty) == static_cast<bool>(false_ty),
                     "Only one side of arith is a vector?");
        if (inferElementwise(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ExtUIOp>(any_op)) {
        auto in_ty = dyn_cast<VectorType>(op.getIn().getType());
        auto out_ty = dyn_cast<VectorType>(op.getType());
        TPU_CHECK_OP(static_cast<bool>(in_ty) == static_cast<bool>(out_ty),
                     "Input and output are not both vectors?");
        auto in_bitwidth = in_ty ? in_ty.getElementTypeBitWidth()
                                 : op.getIn().getType().getIntOrFloatBitWidth();
        if (in_bitwidth == 1) {
          if (inferElementwise(&any_op).failed()) {
            return failure();
          }
        } else {
          if (inferExt(&any_op).failed()) {
            return failure();
          }
        }
      } else if (isa<arith::CmpIOp>(any_op) || isa<arith::CmpFOp>(any_op)) {
        Operation *op = &any_op;  // For TPU_CHECK_OP macros, which use the `op`
                                  // variable in scope
        auto lhs_ty = dyn_cast<VectorType>(any_op.getOperand(0).getType());
        auto rhs_ty = dyn_cast<VectorType>(any_op.getOperand(1).getType());
        TPU_CHECK_OP(static_cast<bool>(lhs_ty) == static_cast<bool>(rhs_ty),
                     "Only one side of cmp is a vector?");
        // TODO(tlongeri): Check that TPU generation supports comparison.
        if (inferElementwise(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ConstantOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<cf::AssertOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<memref::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::IfOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::ForOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::WhileOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RotateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::DynamicRotateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::ConcatenateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StridedLoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StridedStoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::MatmulOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::EraseLayoutOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::IotaOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::GatherOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::DynamicGatherOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::BitcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::TraceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::PRNGRandomBitsOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RegionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::BroadcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ExtractOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::MultiDimReductionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ShapeCastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::VectorStoreOp>(any_op)) {
        if (inferStore<tpu::VectorStoreOp>(op,
                                           /*has_mask=*/op.getMask() != nullptr)
                .failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::StoreOp>(any_op)) {
        if (inferStore<vector::StoreOp>(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::TransposeOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ExtractStridedSliceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (OpTrait::hasElementwiseMappableTraits(&any_op)) {
        // We put elementwise rule to the end in case the overriding rule.
        if (inferElementwise(&any_op).failed()) {
          return failure();
        }
      } else if (mlir::tpu::extensions::canInferVectorLayout(any_op)) {
        if (mlir::tpu::extensions::inferVectorLayout(any_op, target_shape_)
                .failed()) {
          return failure();
        }
      } else {
        return any_op.emitError("Not implemented: Unsupported operation: ")
               << any_op.getName() << " in infer-vector-layout pass";
      }
      CHECK(any_op.getNumResults() == 0 || any_op.hasAttr("out_layout"));
      CHECK(any_op.getNumOperands() == 0 || any_op.hasAttr("in_layout"));
      force_first_tile_offsets_ = false;
    }
    return match_terminator(block.getTerminator());
  }

  LogicalResult infer(arith::ConstantOp op) {
    if (op.getType().isSignlessIntOrIndexOrFloat()) {
      setOutLayout(op, kNoLayout);
      return success();
    }
    if (auto ty = dyn_cast<VectorType>(op.getType())) {
      auto elems = dyn_cast<DenseElementsAttr>(op.getValue());
      TPU_CHECK_OP(ty.getElementType().isSignlessIntOrIndexOrFloat(),
                   "expected scalar element type in vector");
      TPU_CHECK_OP(ty.getRank() > 0, "rank 0 vectors unsupported");
      TPU_CHECK_OP(elems, "expected vector constants to use DenseElementsAttr");
      auto bitwidth = ty.getElementTypeBitWidth();
      if (bitwidth == 1) {
        // i1 is a special case where the layout bitwidth can be different from
        // the element bitwidth, see comment in VectorLayout class
        bitwidth = kNativeBitwidth;
      }
      if (elems.isSplat()) {
        if (ty.getRank() == 1) {
          // Here, we choose to lay out along lanes arbitrarily. It would be
          // equally valid to go with sublanes. Still, this value is so easy
          // to relayout that it shouldn't really make a difference.
          setOutLayout(op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                                        nativeTiling(bitwidth),
                                        ImplicitDim::kSecondMinor));
        } else {  // ty.getRank() >= 2
          setOutLayout(
              op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
        }
      } else {
        TPU_CHECK_OP(ty.getElementTypeBitWidth() == kNativeBitwidth,
                     "Only 32-bit non-splat constants supported");
        if (ty.getRank() == 1) {
          if (ty.getDimSize(0) <= target_shape_[0]) {
            // Use 2D layout with replication.
            NYI("small 1D constants");
          } else {  // NOLINT(readability-else-after-return)
            NYI("large 1D constants");
          }
        } else {  // ty.getRank() >= 2
          setOutLayout(op, VectorLayout(kNativeBitwidth, {0, 0},
                                        default_tiling_, ImplicitDim::kNone));
        }
      }
      return success();
    }
    op.emitOpError("unsupported constant type");
    return failure();
  }

  LogicalResult infer(cf::AssertOp op) {
    setInLayout(op, {kNoLayout});
    return success();
  }

  LogicalResult infer(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return inferBlock(
        op.getBody().front(), [this](Operation *op) -> LogicalResult {
          TPU_CHECK_OP(isa<func::ReturnOp>(op),
                       "Expected func.return terminator");
          for (Value o : op->getOperands()) {
            TPU_CHECK_OP(!isa<VectorType>(o.getType()),
                         "vector returns unsupported");
          }
          SmallVector<Layout, 4> in_layout(op->getNumOperands(), {kNoLayout});
          setInLayout(op, in_layout);
          return success();
        });
  }

  LogicalResult infer(memref::LoadOp op) {
    TPU_CHECK_OP(op.getType().isSignlessIntOrIndexOrFloat(),
                 "memref.load with non-scalar result");
    SmallVector<Layout, 5> in_layout(op.getNumOperands(), {kNoLayout});
    setLayout(op, in_layout, kNoLayout);
    return success();
  }

  LogicalResult infer(scf::IfOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 1, "expected one operand");
    setInLayout(op, {kNoLayout});
    if (inferBlock(*op.thenBlock(), match_yield).failed()) {
      op.emitOpError("failed to infer layout for then branch");
      return failure();
    }
    auto then_yield = op.thenBlock()->getTerminator();
    TPU_CHECK_OP(then_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and then branch yield operands do not match");
    auto then_yield_in_layouts = getLayoutFromOperands(then_yield);
    if (auto else_block = op.elseBlock()) {
      if (inferBlock(*else_block, match_yield).failed()) {
        op.emitOpError("failed to infer layout for else branch");
        return failure();
      }
    }
    if (op->getNumResults() == 0) {
      return success();
    }
    // If the if op has results, it should have both then and else regions with
    // yield op.
    auto else_yield = op.elseBlock()->getTerminator();
    TPU_CHECK_OP(else_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and else branch yield operands do not match");
    auto else_yield_in_layouts = getLayoutFromOperands(else_yield);
    // Find a compatible layout from then and else branches for each result. For
    // example, if we yield offset (*, *) in then branch and offset (*, 0) in
    // else branch, the result offset should be (*, 0).
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    for (auto [then_layout, else_layout, result] : llvm::zip_equal(
             then_yield_in_layouts, else_yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!then_layout.has_value()) {
          return op.emitOpError(
                     "expected a vector layout for then yield input ")
                 << out_idx;
        }
        if (!else_layout.has_value()) {
          return op.emitOpError(
                     "expected a vector layout for else yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            then_layout.value(), else_layout.value(), vty.getShape());
        // If no compatible layout is found in layouts for then and else
        // branches, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(
              then_layout->bitwidth(), {0, 0},
              nativeTiling(then_layout->bitwidth()), ImplicitDim::kNone);
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (then_layout.has_value()) {
          return op.emitOpError("expected no layout for then yield input ")
                 << out_idx;
        }
        if (else_layout.has_value()) {
          return op.emitOpError("expected no layout for else yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    setInLayout(then_yield, out_layouts);
    setInLayout(else_yield, out_layouts);
    setOutLayout(op, out_layouts);
    return success();
  }

  LogicalResult infer(scf::ForOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op.getRegion().hasOneBlock(),
                 "expected one block for scf.for");
    TPU_CHECK_OP(
        op.getNumRegionIterArgs() == op.getNumResults(),
        "expected num_region_iter_args is equal to num_results in scf.for");
    TPU_CHECK_OP(
        op->getNumOperands() == 3 + op.getNumResults(),
        "expected num_operands is equal to 3 + num_results in scf.for");

    auto in_layouts = getLayoutFromOperands(op);
    // Drop the input layouts for lower bound, upper bound. But keep the layout
    // for step because it matches with induction variable in arguments.
    auto arg_layouts = ArrayRef<Layout>(in_layouts).drop_front(2);
    if (assumeLayoutsForBlockArgs(*op.getBody(), arg_layouts).failed() ||
        inferBlock(*op.getBody(), match_yield).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for body in "
          "scf.for op");
    }
    auto yield_op = op.getBody()->getTerminator();
    auto yield_in_layouts = getLayoutFromOperands(yield_op);

    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    bool require_reinfer = false;
    for (auto [in_layout, yield_layout, result] :
         llvm::zip_equal(arg_layouts.drop_front(
                             1),  // Drop the layout for induction variable.
                         yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!in_layout.has_value()) {
          return op.emitOpError("expected a vector layout for input ")
                 << out_idx;
        }
        if (!yield_layout.has_value()) {
          return op.emitOpError("expected a vector layout for yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            in_layout.value(), yield_layout.value(), vty.getShape());
        // If no compatible layout is found in layouts for input and
        // yield, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(in_layout->bitwidth(), {0, 0},
                                           nativeTiling(in_layout->bitwidth()),
                                           ImplicitDim::kNone);
        }
        if (!require_reinfer &&
            (compatible_layout.value() != in_layout.value() ||
             compatible_layout.value() != yield_layout.value())) {
          require_reinfer = true;
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (in_layout.has_value()) {
          return op.emitOpError("expected no layout for input ") << out_idx;
        }
        if (yield_layout.has_value()) {
          return op.emitOpError("expected no layout for yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    if (require_reinfer) {
      // Force same layouts in input layout but skip the first 3 layouts for
      // lower bound, upper bound and step.
      std::copy(out_layouts.begin(), out_layouts.end(), in_layouts.begin() + 3);

      // Terminator in the loop will carry layouts to the next loop but
      // the loop's block args' layouts are determined by the initial inputs. We
      // need to force the same layouts for all in order to make layouts be
      // consistent across all branches. To ensure that, we need to reprocess
      // layout inference for the entire body with the final consolidated
      // layout.
      clearBlockLayouts(*op.getBody());
      if (assumeLayoutsForBlockArgs(*op.getBody(),
                                    ArrayRef<Layout>(in_layouts).drop_front(2))
              .failed() ||
          inferBlock(*op.getBody(), match_yield).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for body in "
            "scf.for op");
      }
    }
    setInLayout(yield_op, out_layouts);
    setLayout(op, in_layouts, out_layouts);
    return success();
  }

  LogicalResult infer(scf::WhileOp op) {
    static LogicalResult (*match_condition)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::ConditionOp>(op), "expected condition terminator");
      return success();
    };
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op.getNumRegions() == 2, "expected two blocks for scf.while");

    SmallVector<Layout, 4> in_layouts = getLayoutFromOperands(op);

    if (assumeLayoutsForBlockArgs(*op.getBeforeBody(), in_layouts).failed() ||
        inferBlock(*op.getBeforeBody(), match_condition).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for before body in "
          "scf.while op");
    }

    if (assumeLayoutsForBlockArgs(*op.getAfterBody(), in_layouts).failed() ||
        inferBlock(*op.getAfterBody(), match_yield).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for after body in "
          "scf.while op");
    }

    auto *cond_op = op.getBeforeBody()->getTerminator();
    auto cond_in_layouts = getLayoutFromOperands(cond_op);
    auto *yield_op = op.getAfterBody()->getTerminator();
    auto yield_in_layouts = getLayoutFromOperands(yield_op);

    // Find a compatible layout from condition body and loop body for each
    // result. For example, if we yield offset (*, *) in condition body and
    // offset (*, 0) in loop body, the result offset should be (*, 0).
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    bool require_reinfer = false;
    for (auto [in_layout, cond_layout, yield_layout, result] : llvm::zip_equal(
             in_layouts, ArrayRef<Layout>(cond_in_layouts).drop_front(1),
             yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!in_layout.has_value()) {
          return op.emitOpError("expected a vector layout for whileOp input ")
                 << out_idx;
        }
        if (!cond_layout.has_value()) {
          return op.emitOpError("expected a vector layout for condition input ")
                 << out_idx + 1;  // ConditionOp's first input is 1 bit bool.
        }
        if (!yield_layout.has_value()) {
          return op.emitOpError("expected a vector layout for yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            cond_layout.value(), yield_layout.value(), vty.getShape());
        if (compatible_layout.has_value()) {
          compatible_layout = VectorLayout::join(
              in_layout.value(), compatible_layout.value(), vty.getShape());
        }
        // If no compatible layout is found in layouts for input, condition and
        // yield, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(in_layout->bitwidth(), {0, 0},
                                           nativeTiling(in_layout->bitwidth()),
                                           ImplicitDim::kNone);
        }
        if (!require_reinfer &&
            (compatible_layout.value() != in_layout.value() ||
             compatible_layout.value() != cond_layout.value() ||
             compatible_layout.value() != yield_layout.value())) {
          require_reinfer = true;
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (in_layout.has_value()) {
          return op.emitOpError("expected no layout for whileOp input ")
                 << out_idx;
        }
        if (cond_layout.has_value()) {
          return op.emitOpError("expected no layout for condition input ")
                 << out_idx + 1;  // ConditionOp's first input is 1 bit bool.
        }
        if (yield_layout.has_value()) {
          return op.emitOpError("expected no layout for yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    if (require_reinfer) {
      clearBlockLayouts(*op.getBeforeBody());
      clearBlockLayouts(*op.getAfterBody());
      // Terminator in the loop will carry layouts to the next loop but
      // the loop's block args' layouts are determined by the initial inputs. We
      // need to force the same layouts for all in order to make layouts be
      // consistent across all branches. To ensure that, we need to reprocess
      // layout inference for the entire body with the final consolidated
      // layout.
      if (assumeLayoutsForBlockArgs(*op.getBeforeBody(), out_layouts)
              .failed() ||
          inferBlock(*op.getBeforeBody(), match_condition).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for before body in "
            "scf.while op");
      }
      if (assumeLayoutsForBlockArgs(*op.getAfterBody(), out_layouts).failed() ||
          inferBlock(*op.getAfterBody(), match_yield).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for after body in "
            "scf.while op");
      }
    }
    std::copy(out_layouts.begin(), out_layouts.end(),
              cond_in_layouts.begin() + 1);  // Skip the first 1 bit bool.
    setInLayout(cond_op, cond_in_layouts);
    setInLayout(yield_op, out_layouts);
    setLayout(op, out_layouts, out_layouts);
    return success();
  }

  // TODO(b/347016737): deprecate the static rotate.
  LogicalResult infer(tpu::RotateOp op) {
    auto bitwidth = op.getType().getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Rotate with non-32-bit data");
    }
    if (op.getType().getRank() < 2) {
      NYI("Unsupported 1D shape");
    }
    auto layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                               ImplicitDim::kNone);
    setLayout(op, layout, layout);
    return success();
  }

  LogicalResult infer(tpu::DynamicRotateOp op) {
    auto bitwidth = op.getType().getElementTypeBitWidth();
    // TODO(b/347067057): Support dynamic rotate with packed dtype.
    if (bitwidth != 32) {
      NYI("Rotate with non-32-bit data");
    }
    if (op.getType().getRank() < 2) {
      NYI("Unsupported 1D shape");
    }
    // TODO(b/337384645): Currently we assume {0, 0} offsets in the input
    // layout. Relax this assumption.
    auto layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                               ImplicitDim::kNone);
    // Calculate the offsets for the output layout.
    LayoutOffsets offsets_out = layout.offsets();
    // We assume there are no implicit dims.
    int tiling_dim = op.getDimension() - (op.getType().getRank() - 2);
    if (auto amount = op.getAmount().getDefiningOp<arith::ConstantOp>();
        amount && (tiling_dim == 0 || tiling_dim == 1)) {
      if (auto integer_attr = dyn_cast<IntegerAttr>(amount.getValue())) {
        const int64_t tile_size = layout.tiling()[tiling_dim];
        const int64_t dim_size = op.getType().getShape()[op.getDimension()];
        const int64_t shift = integer_attr.getValue().getSExtValue();
        if (dim_size % tile_size != 0) {
          offsets_out[tiling_dim] = (dim_size - (shift % dim_size)) % tile_size;
        }
      }
    }
    auto out_layout = VectorLayout(bitwidth, offsets_out,
                                   nativeTiling(bitwidth), ImplicitDim::kNone);
    setLayout(op, {layout, kNoLayout}, out_layout);
    return success();
  }

  LogicalResult infer(tpu::ConcatenateOp op) {
    TPU_CHECK_OP(!op.getSources().empty(),
                 "Need at least one vector to concatenate");
    int64_t res_rank = op.getType().getRank();
    uint32_t dimension = op.getDimension();
    TPU_CHECK_OP(0 <= dimension && dimension < res_rank,
                 "Expect a valid concatenate dimension");
    VectorType res_ty = op.getResult().getType();

    std::optional<int64_t> tiling_dim;
    if (dimension == res_ty.getRank() - 1) {
      tiling_dim = 1;
    } else if (dimension == res_ty.getRank() - 2) {
      tiling_dim = 0;
    }

    if (tiling_dim.has_value()) {
      int64_t starting_point = 0;

      Layout first_layout = getLayout(op.getSources().front());
      SmallVector<Layout, 4> op_layouts = getLayoutFromOperands(op);
      SmallVector<Layout> in_layouts;
      in_layouts.reserve(op.getSources().size());
      int8_t bitwidth = first_layout->bitwidth();

      // Set implicit dim to treat 1D as (1, N) and tile it as (1, 128)
      std::array<int64_t, 2> tiling =
          res_rank == 1 ? std::array<int64_t, 2>{1L, target_shape_[1]}
                        : nativeTiling(bitwidth);
      ImplicitDim implicit_dim =
          res_rank == 1 ? ImplicitDim::kSecondMinor : ImplicitDim::kNone;
      std::array<int64_t, 2> vreg_slice =
          VectorLayout::vregSlice(target_shape_, bitwidth, tiling);
      for (int i = 0; i < op.getSources().size(); ++i) {
        // Compute the offset per source.
        // Ex: for a cat of (10, 128), (10, 128) on dim 0, where the
        // vreg_slice for that dim is 8, the first source starts at
        // offset 0, and overflows the vreg
        // by 2, so the offset for the second input is 2.
        ArrayRef<int64_t> op_shape =
            cast<VectorType>(op.getSources()[i].getType()).getShape();
        Layout op_layout = op_layouts[i];
        int64_t offset_amount = starting_point % vreg_slice[tiling_dim.value()];
        if (offset_amount >= tiling[tiling_dim.value()]) {
          return op.emitError(
              "Not implemented: Input offsets outside of the first tile");
        }
        SmallVector<int64_t> in_idx{op_layout->offsets()[0].value_or(0),
                                    op_layout->offsets()[1].value_or(0)};
        in_idx[tiling_dim.value()] = offset_amount;
        starting_point += op_shape[dimension];
        in_layouts.push_back(VectorLayout(bitwidth, {in_idx[0], in_idx[1]},
                                          tiling, implicit_dim));
      }
      SmallVector<int64_t> res_layout_offsets(
          {first_layout->offsets()[0].value_or(0),
           first_layout->offsets()[1].value_or(0)});
      res_layout_offsets[tiling_dim.value()] = 0;
      // TODO(mvoz): A tiny optimization we could do here later is to
      // no-op setting tiling when sublane dim size is aligned to sublane
      // tiling.
      VectorLayout res_layout =
          VectorLayout(bitwidth, {res_layout_offsets[0], res_layout_offsets[1]},
                       tiling, implicit_dim);
      setLayout(op, in_layouts, res_layout);
      return success();
    } else {
      Layout layout = getLayout(op.getSources().front());
      // When concatenating vectors with replicated offsets, we want to reset
      // the replicated offset to zero. Because we are not sure if the
      // replicated value from each vector are same.
      layout = VectorLayout(
          layout->bitwidth(),
          {layout->offsets()[0].value_or(0), layout->offsets()[1].value_or(0)},
          layout->tiling(), layout->implicit_dim());
      SmallVector<Layout> in_layouts(op->getNumOperands(), layout);
      setLayout(op, in_layouts, layout);
      return success();
    }
  }

  LogicalResult infer(tpu::LoadOp op) {
    auto res_ty = op.getResult().getType();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();

    // We expect the result is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && res_ty.getShape()[0] == target_shape_[0] &&
                     res_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit loads supported");
    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    auto out_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                   ImplicitDim::kNone);
    setLayout(op, in_layout, out_layout);
    return success();
  }

  LogicalResult infer(tpu::StridedLoadOp op) {
    auto vty = op.getResult().getType();
    int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Strided load with non 32-bit data");
    }
    if (vty.getRank() < 2) {
      NYI("Strided load with 1D vector");
    }
    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    setLayout(op, in_layout,
              VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                           ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(tpu::StridedStoreOp op) {
    auto vty = op.getValueToStore().getType();
    int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Strided store with non 32-bit data");
    }
    if (vty.getRank() < 2) {
      NYI("Strided store with 1D vector");
    }
    auto store_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone);
    SmallVector<Layout, 5> in_layout{op->getNumOperands(), kNoLayout};
    in_layout[0] = store_layout;
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::MatmulOp op) {
    auto lhs_bitwidth = op.getLhs().getType().getElementTypeBitWidth();
    auto rhs_bitwidth = op.getRhs().getType().getElementTypeBitWidth();
    auto acc_bitwidth = op.getAcc().getType().getElementTypeBitWidth();
    auto res_bitwidth = op.getResult().getType().getElementTypeBitWidth();
    TPU_CHECK_OP(acc_bitwidth == kNativeBitwidth,
                 "Expected 32-bit acc in tpu::MatmulOp");
    TPU_CHECK_OP(res_bitwidth == kNativeBitwidth,
                 "Expected 32-bit result in tpu::MatmulOp");
    auto lhs_layout = VectorLayout(
        lhs_bitwidth, {0, 0}, nativeTiling(lhs_bitwidth), ImplicitDim::kNone);
    auto rhs_layout = VectorLayout(
        rhs_bitwidth, {0, 0}, nativeTiling(rhs_bitwidth), ImplicitDim::kNone);
    auto acc_layout = VectorLayout(
        acc_bitwidth, {0, 0}, nativeTiling(acc_bitwidth), ImplicitDim::kNone);
    setLayout(op, {lhs_layout, rhs_layout, acc_layout},
              VectorLayout(kNativeBitwidth, {0, 0}, default_tiling_,
                           ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(tpu::StoreOp op) {
    auto store_ty = op.getValueToStore().getType();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();

    // We expect the value to store is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && store_ty.getShape()[0] == target_shape_[0] &&
                     store_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit stores supported");
    auto store_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone);
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::EraseLayoutOp op) {
    setLayout(op, kNoLayout, kNoLayout);
    return success();
  }

  LogicalResult infer(tpu::GatherOp op) {
    auto src_layout = getLayout(op.getSource());
    setLayout(op, src_layout, src_layout);
    return success();
  }

  LogicalResult infer(tpu::DynamicGatherOp op) {
    // TODO(jevinjiang): we could preserve some offsets such as replicated
    // offset but since we are forcing all operands and result to be the same
    // layout, we can set all offsets to zero for now. Also maybe we should
    // consider adding this to elementwise rule.
    const int bitwidth = op.getType().getElementTypeBitWidth();
    if (bitwidth != 8 && bitwidth != 16 && bitwidth != 32) {
      return op.emitOpError(
          "Not implemented: Only 8-, 16- or 32-bit gathers supported");
    }
    if (bitwidth != op.getIndices().getType().getElementTypeBitWidth()) {
      return op.emitOpError(
          "Not implemented: Gather indices and result have different "
          "bitwidths");
    }
    VectorLayout layout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                        ImplicitDim::kNone);
    setLayout(op, {layout, layout}, layout);
    return success();
  }

  LogicalResult infer(tpu::BitcastOp op) {
    // Note we have verified the shapes in verify().
    auto in_ty = cast<VectorType>(op.getInput().getType());
    auto out_ty = cast<VectorType>(op.getOutput().getType());
    auto in_bitwidth = in_ty.getElementTypeBitWidth();
    auto out_bitwidth = out_ty.getElementTypeBitWidth();
    auto src_layout = getLayout(op.getInput());
    LayoutOffsets src_offsets = src_layout->offsets();
    auto implicit_dim = src_layout->implicit_dim();
    if (src_offsets[0].value_or(0) * in_bitwidth % out_bitwidth != 0) {
      // Force offset to zero if the input offset on the second minor dimension
      // is not a multiple of the ratio of output and input bitwidth.
      src_offsets[0] = 0;
    } else if (!src_offsets[0].has_value() && in_bitwidth > out_bitwidth) {
      // We can't preserve replicated offset for decreasing bitwidth.
      src_offsets[0] = 0;
    }
    // Force implicit dim to None if the bitwidth changes. Because we expect 2nd
    // minor dim size ratio matches the bitwidth ratio in input and output.
    if (in_bitwidth != out_bitwidth) {
      if (in_ty.getRank() < 2 || out_ty.getRank() < 2) {
        return op.emitOpError(
            "Not implemented: bitcast between different bitwidths on a 1D "
            "vector.");
      }
      implicit_dim = ImplicitDim::kNone;
    }
    // TODO(b/348485035): Instead of forcing to native tiling, bitcast should
    // keep the input tiling and infer bitcastable tiling for output. For
    // example, it is valid to bitcast vector<8x128xi32> with tile (1, 128) to
    // vector<8x128xbf16> with tile (2, 128).
    setLayout(
        op,
        VectorLayout(in_bitwidth, src_offsets, nativeTiling(in_bitwidth),
                     implicit_dim),
        VectorLayout(out_bitwidth,
                     {src_offsets[0].has_value()
                          ? src_offsets[0].value() * in_bitwidth / out_bitwidth
                          : src_offsets[0],
                      src_offsets[1]},
                     nativeTiling(out_bitwidth), implicit_dim));
    return success();
  }

  LogicalResult infer(tpu::TraceOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    TPU_CHECK_OP(op->getNumResults() == 0, "results unsupported");
    return inferBlock(*op.getBody(), match_yield);
  }

  LogicalResult infer(tpu::RegionOp op) {
    static LogicalResult (*match_region)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    auto body_result =
        inferBlock((*op).getRegion(0).getBlocks().front(), match_region);
    if (body_result.failed()) {
      return op.emitOpError("failed to infer vector layout in region body");
    }
    auto yield_op = op.getBody()->getTerminator();
    auto yield_in_layouts = getLayoutFromOperands(yield_op);
    setInLayout(yield_op, yield_in_layouts);
    setOutLayout(op, yield_in_layouts);
    return success();
  }

  LogicalResult infer(tpu::IotaOp op) {
    auto ty = op.getResult().getType();
    const int bitwidth = ty.getElementTypeBitWidth();
    TPU_CHECK_OP(ty.getRank() >= 2, "iota rank below 2D unsupported");
    LayoutOffsets offsets = {std::nullopt, std::nullopt};
    if (llvm::is_contained(op.getDimensions(), ty.getRank() - 2)) {
      offsets[0] = 0;
    }
    if (llvm::is_contained(op.getDimensions(), ty.getRank() - 1)) {
      offsets[1] = 0;
    }
    setOutLayout(op, VectorLayout(bitwidth, offsets, nativeTiling(bitwidth),
                                  ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(vector::BroadcastOp op) {
    auto some_src_ty = op.getSourceType();
    auto res_ty = op.getResultVectorType();
    TPU_CHECK_OP(res_ty.getRank() > 0, "rank 0 vectors unsupported");
    if (some_src_ty.isSignlessIntOrIndexOrFloat()) {
      auto bitwidth = some_src_ty.getIntOrFloatBitWidth();
      // TODO(b/320725357): We need a better design for mask layout. For now, we
      // always set layout bitwidth of Vmask to 32bit.
      if (bitwidth == 1) {
        bitwidth = kNativeBitwidth;
      }
      if (res_ty.getRank() == 1) {
        // We use a full vreg tile, because only then its layout can be changed
        // for free.
        setLayout(
            op, kNoLayout,
            VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                         nativeTiling(bitwidth), ImplicitDim::kSecondMinor));
      } else {  // rank >= 2  // NOLINT(readability-else-after-return)
        setLayout(op, kNoLayout,
                  VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
      }
      return success();
    }
    if (auto src_ty = dyn_cast<VectorType>(some_src_ty)) {
      auto some_layout = getLayout(op.getSource());
      TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
      auto &layout = *some_layout;
      if (layout.implicit_dim() != ImplicitDim::kNone && src_ty.getRank() > 1) {
        VectorLayout layout_2d(layout.bitwidth(), layout.offsets(),
                               layout.tiling(), ImplicitDim::kNone);
        if (layout_2d.equivalentTo(layout, src_ty.getShape(), target_shape_)) {
          // TODO(b/342237796): Stop preferring 2D layouts (if given the choice)
          // and defer the work, if any, to relayout.
          layout = layout_2d;
        }
      }
      auto src_tiled_ishape = layout.getImplicitTiledDims(src_ty.getShape(), 1);
      auto dst_tiled_ishape = layout.getImplicitTiledDims(res_ty.getShape(), 1);
      if (src_tiled_ishape[0] != dst_tiled_ishape[0] &&
          layout.offsets()[0] != std::nullopt) {
        // TODO(tlongeri): Remove this. We support non-native tiling now, but
        // things may still break downstream due to missing relayouts.
        LayoutOffsets offsets = layout.offsets();
        if (layout.tiling()[0] == 1 && layout.bitwidth() == kNativeBitwidth) {
          offsets[0] = std::nullopt;
        }
        layout = VectorLayout(layout.bitwidth(), offsets,
                              nativeTiling(layout.bitwidth()),
                              layout.implicit_dim());
      }
      LayoutOffsets offsets = layout.offsets();
      for (int i = 0; i < 2; ++i) {
        if (src_tiled_ishape[i] != dst_tiled_ishape[i]) {
          offsets[i] = std::nullopt;
        }
      }
      setLayout(op, layout,
                VectorLayout(layout.bitwidth(), offsets, layout.tiling(),
                             layout.implicit_dim()));
      return success();
    }
    op.emitOpError("unsupported broadcast source type");
    return failure();
  }

  LogicalResult infer(vector::ExtractOp op) {
    TPU_CHECK_OP(!op.hasDynamicPosition(), "dynamic indices not supported");
    TPU_CHECK_OP(
        op.getSourceVectorType().getElementTypeBitWidth() == kNativeBitwidth,
        "Only 32-bit types supported");
    auto layout = getLayout(op.getVector());
    TPU_CHECK_OP(layout.has_value(), "missing vector layout");
    if (VectorType res_vty = dyn_cast<VectorType>(op.getResult().getType());
        res_vty != nullptr) {
      if (res_vty.getRank() == 1 &&
          layout->implicit_dim() == ImplicitDim::kNone) {
        const int64_t second_minor_idx = op.getStaticPosition().back();
        const LayoutOffset second_minor_offset = layout->offsets()[0];
        const LayoutOffset res_second_minor_offset =
            second_minor_offset.has_value()
                ? (*second_minor_offset + second_minor_idx) %
                      layout->vregSlice(target_shape_)[0]
                : LayoutOffset();
        // TODO: b/342235360 - We should already support this but it needs
        //                     testing.
        TPU_CHECK_OP(!res_second_minor_offset.has_value() ||
                         *res_second_minor_offset < layout->tiling()[0],
                     "Not implemented: Slice does not start on the first tile "
                     "of a VReg");
        setLayout(op, layout,
                  VectorLayout(layout->bitwidth(),
                               {res_second_minor_offset, layout->offsets()[1]},
                               layout->tiling(), ImplicitDim::kSecondMinor));
      } else {
        TPU_CHECK_OP(layout->layout_rank() <= res_vty.getRank(),
                     "Internal error: Layout has too many dimensions for "
                     "vector type (invalid vector.extract?)")
        setLayout(op, layout, layout);
      }
    } else {
      setLayout(op,
                VectorLayout(kNativeBitwidth, {0, 0}, layout->tiling(),
                             layout->implicit_dim()),
                kNoLayout);
    }
    return success();
  }

  LogicalResult infer(vector::LoadOp op) {
    auto src_ty = getMemRefType(op.getBase());
    auto res_ty = op.getVectorType();
    TPU_CHECK_OP(src_ty.getRank() == res_ty.getRank(),
                 "memref and vector rank mismatch");
    int64_t rank = res_ty.getRank();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();
    if (kNativeBitwidth % bitwidth != 0) {
      return op.emitOpError("Unsupported bitwidth");
    }
    const int packing = kNativeBitwidth / bitwidth;
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           src_ty.getRank(), src_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    CHECK_EQ(op->getNumOperands(), op.getIndices().size() + 1);
    // Infer the static offset on a given tiling dimension.
    auto infer_offset = [&](int64_t &offset,
                            int64_t tiling_dim) -> LogicalResult {
      int dim = rank - tiling.size() + tiling_dim;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        offset =
            cast<IntegerAttr>(cst_op.getValue()).getInt() % tiling[tiling_dim];
        return success();
      }
      if (failed(
              verifyDivisibleIndex(tiled_index, tiling[tiling_dim], dim, op))) {
        return failure();
      }
      offset = 0;
      return success();
    };

    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D loads");
      const int64_t lane_tiling = packing * target_shape_[1];
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % lane_tiling == 0, "Unsupported tiling for 1D load");
      int64_t offset;
      if (failed(infer_offset(offset, 0))) {
        return failure();
      }
      // TODO(apaszke): We could generate replicated loads for short values.
      setLayout(op, in_layout,
                VectorLayout(bitwidth, {0, offset % lane_tiling},
                             {1, lane_tiling}, ImplicitDim::kSecondMinor));
    } else {  // rank >= 2
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ loads");
      LayoutOffsets offsets = {0, 0};
      const auto tile_src_shape = src_ty.getShape().take_back(2);
      const auto tile_res_shape = res_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_res_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d load");
      // We can load starting from any row if the source has few columns,
      // because the tiling structure degenerates to regular layout there.
      // There is also no extra need for alignment if we load a single sublane.
      // TODO(apaszke): Also no need to align if we don't exceed the base chunk!
      if (bitwidth == 32 &&
          (tile_src_shape[1] <= target_shape_[1] || num_sublanes == 1)) {
        offsets[0] = 0;
      } else if (failed(infer_offset(*offsets[0], 0))) {
        return failure();
      }
      if (failed(infer_offset(*offsets[1], 1))) {
        return failure();
      }
      std::array<int64_t, 2> layout_tiling{tiling[0], tiling[1]};
      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_res_shape[1] > target_shape_[1]) {
        // We can strided load sublanes if we're loading a single sublane for
        // multiple times. Enabling this helps load one entire row from memref
        // more efficiently.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, offsets, {1, layout_tiling[1]},
                               ImplicitDim::kNone));
      } else if (num_sublanes == 1 && bitwidth == 32 &&
                 tiling == target_shape_) {
        // We can use replicated loads if we're only loading a single sublane.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, {std::nullopt, offsets[1]},
                               layout_tiling, ImplicitDim::kNone));
      } else if (bitwidth == 32 &&
                 canReinterpretToUntiledMemref(
                     op.getBase(), target_shape_,
                     /*allow_minormost_padding=*/true) &&
                 *(src_ty.getShape().end() - 2) > 1) {
        // Since it is untiled, we can load from any arbitrary address which
        // means we can always set the sublane offset to 0.
        // Note: if the src_shape[-2] == 1, we can just use the tiling from ref.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, {0, offsets[1].value_or(0)},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
      } else {
        setLayout(
            op, in_layout,
            VectorLayout(bitwidth, offsets, layout_tiling, ImplicitDim::kNone));
      }
    }
    return success();
  }

  LogicalResult infer(vector::ExtractStridedSliceOp op) {
    auto input_layout = getLayout(op.getVector());
    TPU_CHECK_OP(input_layout, "missing vector layout");
    auto offsets_attr = op.getOffsets().getValue();
    auto strides_attr = op.getStrides().getValue();
    auto offsets = llvm::map_to_vector(offsets_attr, [](auto attr) {
      return cast<IntegerAttr>(attr).getInt();
    });
    input_layout->insertImplicit<int64_t>(offsets, 0);
    auto vreg_slice = input_layout->vregSlice(target_shape_);
    LayoutOffsets new_layout_offsets;
    if (input_layout->offsets()[0].has_value()) {
      new_layout_offsets[0] =
          (*(offsets.end() - 2) + *input_layout->offsets()[0]) % vreg_slice[0];
    }
    if (input_layout->offsets()[1].has_value()) {
      new_layout_offsets[1] =
          (*(offsets.end() - 1) + *input_layout->offsets()[1]) % vreg_slice[1];
    }
    for (auto stride : strides_attr) {
      TPU_CHECK_OP(cast<IntegerAttr>(stride).getInt() == 1,
                   "Only trivial strides supported.");
    }

    setLayout(
        op, input_layout,
        VectorLayout(input_layout->bitwidth(), new_layout_offsets,
                     input_layout->tiling(), input_layout->implicit_dim()));
    return success();
  }

  LogicalResult infer(vector::MultiDimReductionOp op) {
    auto src_ty = op.getSourceVectorType();
    auto dst_ty = dyn_cast<VectorType>(op.getDestType());
    TPU_CHECK_OP(dst_ty, "only reductions with vector results supported");
    llvm::ArrayRef<int64_t> dims = op.getReductionDims();
    int64_t src_rank = src_ty.getRank();
    auto acc_layout = getLayout(op.getAcc());
    TPU_CHECK_OP(is_fully_replicated(acc_layout),
                 "only constant accumulators supported");
    TPU_CHECK_OP(
        src_ty.getElementTypeBitWidth() == 32 ||
            src_ty.getElementTypeBitWidth() == 16,
        "only 32-bit (and 16-bit only on some targets) reductions supported");
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto &src_layout = *some_src_layout;
    std::array<bool, 2> reduces;
    switch (src_layout.implicit_dim()) {
      case VectorLayout::ImplicitDim::kNone:
        reduces = {
            std::find(dims.begin(), dims.end(), src_rank - 2) != dims.end(),
            std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end()};
        break;
      case VectorLayout::ImplicitDim::kSecondMinor:
        reduces = {false, std::find(dims.begin(), dims.end(), src_rank - 1) !=
                              dims.end()};
        break;
      case VectorLayout::ImplicitDim::kMinor:
        reduces = {
            std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end(),
            false};
        break;
    }
    if ((reduces[0] || reduces[1]) &&
        !src_layout.hasNativeTiling(target_shape_)) {
      src_layout = VectorLayout(src_layout.bitwidth(), src_layout.offsets(),
                                nativeTiling(src_layout.bitwidth()),
                                src_layout.implicit_dim());
    }
    LayoutOffsets out_offsets = src_layout.offsets();
    for (int i = 0; i < out_offsets.size(); ++i) {
      if (reduces[i]) {
        out_offsets[i] = std::nullopt;
      }
    }
    ImplicitDim out_implicit_dim = src_layout.implicit_dim();
    if ((reduces[0] && reduces[1]) ||
        (src_layout.implicit_dim() != ImplicitDim::kNone &&
         (reduces[0] || reduces[1]))) {
      TPU_CHECK_OP(
          dst_ty.getRank() > 0 && *(dst_ty.getShape().end() - 1) == 1,
          "Not implemented: reductions over both trailing dimensions are only "
          "supported when the resulting value has a trailing axis of size 1");
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[0]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[1]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kMinor;
    }
    setLayout(op, {src_layout, acc_layout},
              VectorLayout(src_layout.bitwidth(), out_offsets,
                           src_layout.tiling(), out_implicit_dim));
    return success();
  }

  LogicalResult infer(vector::ShapeCastOp op) {
    auto src_ty = op.getSourceVectorType();
    auto src_shape = src_ty.getShape();
    auto res_ty = op.getResultVectorType();
    auto res_shape = res_ty.getShape();
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto layout = *some_src_layout;
    const unsigned bitwidth = src_ty.getElementTypeBitWidth();
    const std::array<int64_t, 2> native_tiling = nativeTiling(bitwidth);
    const std::array<int64_t, 2> src_tiled_ishape =
        layout.getImplicitTiledDims(src_shape, 1);
    const std::array<int64_t, 2> vreg_slice = layout.vregSlice(target_shape_);

    // TODO(tlongeri): Be smarter about trying implicit dims. We should probably
    // only add them when folding dimensions, and remove them when unfolding.
    // The ordering of candidate implicit dims is important! Inserting an
    // implicit second minor can make a reshape possible, but also very
    // inefficient. We should always prefer to try with None first.
    SmallVector<ImplicitDim, 3> candidate_implicit_dims;
    if (res_shape.size() >= 2) {
      candidate_implicit_dims.push_back(ImplicitDim::kNone);
    }
    if (!res_shape.empty()) {
      candidate_implicit_dims.push_back(ImplicitDim::kSecondMinor);
      candidate_implicit_dims.push_back(ImplicitDim::kMinor);
    }
    // TODO(b/340625465): Add case with both implicit dims once we support it.

    // See if we can get implicit tiled dimensions to match. This is always a
    // no-op.
    for (const ImplicitDim implicit_dim : candidate_implicit_dims) {
      const std::array<int64_t, 2> res_tiled_ishape =
          VectorLayout::getImplicitTiledDims(implicit_dim, res_shape, 1);
      if (src_tiled_ishape == res_tiled_ishape) {
        // Nothing changes in the tiled dimensions
        setLayout(op, layout,
                  VectorLayout(layout.bitwidth(), layout.offsets(),
                               layout.tiling(), implicit_dim));
        return success();
      }
    }

    // See if we can do sublane or lane (un)folding.
    for (const ImplicitDim implicit_dim : candidate_implicit_dims) {
      const std::array<int64_t, 2> res_tiled_ishape =
          VectorLayout::getImplicitTiledDims(implicit_dim, res_shape, 1);
      // Sublane (un)folding. We attempt to reduce the sublane tiling, which
      // might make this reshape a no-op. We use do-while to handle the packed
      // 1D tilings that use 1 in the sublane dimension.
      int64_t sublane_tiling = vreg_slice[0];
      do {
        auto src_res_tiled_equal = src_tiled_ishape[1] == res_tiled_ishape[1];
        auto vreg_num_elements =
            target_shape_[0] * target_shape_[1] * layout.packing();
        auto single_subline_mod_1024 =
            (sublane_tiling == 1 &&
             src_tiled_ishape[1] % vreg_num_elements == 0 &&
             res_tiled_ishape[1] % vreg_num_elements == 0);
        if ((src_res_tiled_equal || single_subline_mod_1024) &&
            src_tiled_ishape[0] % sublane_tiling == 0 &&
            res_tiled_ishape[0] % sublane_tiling == 0) {
          std::array<int64_t, 2> tiling = {sublane_tiling, target_shape_[1]};
          // TODO(b/343808585): We shouldn't force second minor offset to 0 when
          //                    unfolding, it's still a no-op, but we need to
          //                    add support in apply-vector-layout.
          LayoutOffsets offsets = {0, layout.offsets()[1]};
          setLayout(
              op,
              VectorLayout(layout.bitwidth(), offsets, tiling,
                           layout.implicit_dim()),
              VectorLayout(layout.bitwidth(), offsets, tiling, implicit_dim));
          return success();
        }
        sublane_tiling /= 2;
      } while (sublane_tiling >= layout.packing());
      // Lane (un)folding.
      if (src_tiled_ishape[1] != res_tiled_ishape[1] &&
          src_tiled_ishape[1] % layout.tiling()[1] == 0 &&
          res_tiled_ishape[1] % layout.tiling()[1] == 0) {
        const int packing = kNativeBitwidth / bitwidth;
        const auto elements_per_vreg = native_tiling[0] * native_tiling[1];
        // When we shapecast from input shape
        // (..., m * target_shape_[1] * packing) to output shape
        // (..., target_shape_[1]), the reshape becomes no-op when input is
        // densely packed with tiling (1, target_shape_[1] * packing) and output
        // has the native tiling.
        if (res_tiled_ishape[1] == target_shape_[1] &&
            res_tiled_ishape[0] % native_tiling[0] == 0 &&
            src_tiled_ishape[1] % elements_per_vreg == 0) {
          // Inferring in_layout to have tiling (1, 128 * packing) triggers any
          // necessary relayout before shapecast.
          setLayout(op,
                    VectorLayout(layout.bitwidth(), {0, 0},
                                 {1, target_shape_[1] * packing},
                                 layout.implicit_dim()),
                    VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                                 implicit_dim));
          return success();
        }

        // When we shapecast from input shape (..., target_shape_[1]) to output
        // shape (..., m * target_shape_[1] * packing), the reshape becomes
        // no-op when input has the native tiling and output is densely packed
        // with tiling (1, target_shape_[1] * packing).
        if (src_tiled_ishape[1] == target_shape_[1] &&
            src_tiled_ishape[0] % native_tiling[0] == 0 &&
            res_tiled_ishape[1] % elements_per_vreg == 0) {
          setLayout(
              op,
              VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                           layout.implicit_dim()),
              VectorLayout(layout.bitwidth(), {0, 0},
                           {1, target_shape_[1] * packing}, implicit_dim));
          return success();
        }
      }
    }

    // Try adding a singleton innermost dim to the actual *implicit* shape.
    if (res_shape.size() >= 2 &&
        res_shape.take_back(2) == ArrayRef<int64_t>({src_tiled_ishape[1], 1})) {
      TPU_CHECK_OP(bitwidth == kNativeBitwidth,
                   "Insertion of minor dim that is not a no-op only "
                   "supported for 32-bit types");
      setLayout(op,
                VectorLayout(layout.bitwidth(), layout.offsets(), native_tiling,
                             layout.implicit_dim()),
                VectorLayout(layout.bitwidth(), {0, std::nullopt},
                             native_tiling, ImplicitDim::kNone));
      return success();
    }

    // Shape cast (..., m, n, k * target_shape_[1]) -> (..., m, n * k *
    // target_shape_[1]) for 32-bit types. We allow multiple major or minor
    // dimensions to be folded or unfolded.
    if (kNativeBitwidth == bitwidth && res_shape.size() >= 2 &&
        src_shape.size() >= 2 && src_shape.back() % native_tiling[1] == 0 &&
        res_shape.back() % native_tiling[1] == 0 &&
        (mlir::tpu::canFoldMinorDimsToSize(src_shape, res_shape.back()) ||
         mlir::tpu::canFoldMinorDimsToSize(res_shape, src_shape.back()))) {
      // TODO(jsreeram): Add support for picking space-efficient tilings for
      // small 2nd minor dim shapes.
      // Example 1: (4, 2, 1024) -> (4, 2048) If we infer src and tgt layout to
      // be (1, 128), it is no-op because essentially we just shufflle the VREGs
      // in VREG array.
      // Example 2: (4, 256) -> (1, 1024) is actually sublane
      // shuffle inside each vreg from [0, 1, 2, 3, 4,..7] to [0, 4, 1, 5, ...]
      setLayout(op,
                VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                             ImplicitDim::kNone),
                VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                             ImplicitDim::kNone));
      return success();
    }
    op.emitOpError("infer-vector-layout: unsupported shape cast");
    return failure();
  }

  template <typename Op>
  LogicalResult inferStore(Op op, bool has_mask = false) {
    auto ref_ty = getMemRefType(op.getBase());
    auto store_ty = op.getValueToStore().getType();
    TPU_CHECK_OP(ref_ty.getRank() == store_ty.getRank(),
                 "memref and vector rank mismatch");
    int64_t rank = ref_ty.getRank();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();
    if (kNativeBitwidth % bitwidth != 0) {
      return op.emitOpError("Unsupported bitwidth");
    }
    const int packing = kNativeBitwidth / bitwidth;
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           ref_ty.getRank(), ref_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    // Infer the static offset on a given tiling dimension.
    auto infer_offset = [&](int64_t &offset,
                            int64_t tiling_dim) -> LogicalResult {
      int dim = rank - tiling.size() + tiling_dim;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        offset =
            cast<IntegerAttr>(cst_op.getValue()).getInt() % tiling[tiling_dim];
        return success();
      }
      if (failed(
              verifyDivisibleIndex(tiled_index, tiling[tiling_dim], dim, op))) {
        return failure();
      }
      offset = 0;
      return success();
    };

    Layout store_layout;
    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D store");
      const int64_t lane_tiling = packing * target_shape_[1];
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % lane_tiling == 0,
                   "Unsupported 1D tiling for 1D store");
      int64_t offset;
      if (failed(infer_offset(offset, 0))) {
        return failure();
      }
      store_layout = VectorLayout(bitwidth, {0, offset % lane_tiling},
                                  {1, lane_tiling}, ImplicitDim::kSecondMinor);
    } else {  // rank >= 2  // NOLINT(readability-else-after-return)
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ store");
      LayoutOffsets offsets = {0, 0};
      const auto tile_ref_shape = ref_ty.getShape().take_back(2);
      const auto tile_store_shape = store_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_store_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d store");
      // We can store starting from any row if the source has few columns,
      // because the tiling structure degenerates to regular layout there.
      // There is also no extra need for alignment if we store a single sublane.
      // TODO(apaszke): Also no need to align if we don't exceed the base chunk!
      if (bitwidth == 32 &&
          (tile_ref_shape[1] <= target_shape_[1] || num_sublanes == 1)) {
        offsets[0] = 0;
      } else if (failed(infer_offset(*offsets[0], 0))) {
        return failure();
      }
      if (failed(infer_offset(*offsets[1], 1))) {
        return failure();
      }
      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_store_shape[1] > target_shape_[1]) {
        // We can strided store sublanes if we're storing a single sublane for
        // multiple times. Enabling this helps store one entire row to memref
        // more efficiently.
        store_layout =
            VectorLayout(bitwidth, offsets, {1, tiling[1]}, ImplicitDim::kNone);
      } else if (bitwidth == 32 &&
                 // We accept padding in the minormost dim, because
                 // apply_vector_layout will properly mask stores.
                 canReinterpretToUntiledMemref(
                     op.getBase(), target_shape_,
                     /*allow_minormost_padding=*/true)) {
        // Since it is untiled, we can store to any arbitrary address which
        // means the sublane offset can be any value and we can fold it to
        // 2nd minor index.
        auto prev_store_layout = getLayout(op.getValueToStore());
        TPU_CHECK_OP(prev_store_layout.has_value(), "missing vector layout");
        offsets[0] = prev_store_layout->offsets()[0].value_or(0);
        if (offsets[1].value_or(0) >= tiling[1]) {
          offsets[1] = 0;
        }
        store_layout = VectorLayout(bitwidth, offsets, nativeTiling(bitwidth),
                                    ImplicitDim::kNone);
      } else {
        store_layout = VectorLayout(bitwidth, offsets, {tiling[0], tiling[1]},
                                    ImplicitDim::kNone);
      }
    }
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    if (has_mask) {
      // Mask layout should be the same as the layout of value to store.
      in_layout.push_back(store_layout);
    }
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::TransposeOp op) {
    auto permutation = op.getPermutation();
    TPU_CHECK_OP(permutation.size() > 1,
                 "Vector and scalar transpose should be a no-op and removed");

    auto some_layout = getLayout(op.getVector());
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    auto src_ty = op.getSourceVectorType();
    TPU_CHECK_OP(permutation.size() == src_ty.getRank(),
                 "Transpose permutation has incorrect rank");
    bool untiled_tiled_swap = false;
    // TODO(mvoz): Expand to more general cases. b/419268277
    if (permutation.size() == 3 && permutation[0] == 1 && permutation[1] == 0) {
      untiled_tiled_swap = true;
    } else {
      for (auto dim : permutation.drop_back(2)) {
        TPU_CHECK_OP(dim < src_ty.getRank() - 2,
                     "Unsupported transpose permutation - minor dims into "
                     "major > 3 dimensions");
      }
      for (auto dim : permutation.take_back(2)) {
        TPU_CHECK_OP(dim >= src_ty.getRank() - 2,
                     "Unsupported transpose permutation - major dims into "
                     "minor > 3 dimensions");
      }
    }
    Layout required_layout = some_layout;
    // Require native tiling if we're going to use the XLU, or doing a
    // major/minor permute.
    if (untiled_tiled_swap ||
        permutation[permutation.size() - 1] == permutation.size() - 2) {
      auto native_tiling = nativeTiling(layout.bitwidth());
      required_layout = VectorLayout(layout.bitwidth(), LayoutOffsets{0, 0},
                                     native_tiling, ImplicitDim::kNone);
    }
    setLayout(op, required_layout, required_layout);
    return success();
  }

  LogicalResult inferExt(Operation *op) {
    TPU_CHECK_OP(op->getNumOperands() == 1, "expect 1 operand");
    TPU_CHECK_OP(op->getNumResults() == 1, "expect 1 result");
    auto src_ty = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op->getResult(0).getType());
    unsigned src_bitwidth = src_ty.getElementTypeBitWidth();
    unsigned dst_bitwidth = dst_ty.getElementTypeBitWidth();
    auto some_layout = getLayout(op->getOperand(0));
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    if (isa<tpu::ExtFOp>(op)) {
      TPU_CHECK_OP(dst_bitwidth == 32 || dst_bitwidth == 16,
                   "Only supported extensions to 32-bit (float32) or 16-bit "
                   "(bfloat16)");
    }
    auto &layout = *some_layout;
    Layout src_layout;
    Layout dst_layout;
    if (layout.tiling() == nativeTiling(src_bitwidth)) {
      // If the source is already in native tiling, we can unpack it directly.
      std::array<int64_t, 2> dst_native_tiling = nativeTiling(dst_bitwidth);
      LayoutOffsets offsets = {layout.offsets()[0]
                                   ? *layout.offsets()[0] % dst_native_tiling[0]
                                   : LayoutOffset(),
                               layout.offsets()[1]};
      DCHECK_LT(offsets[1].value_or(0), dst_native_tiling[1]);
      src_layout = VectorLayout(src_bitwidth, offsets, layout.tiling(),
                                layout.implicit_dim());
      dst_layout = VectorLayout(dst_bitwidth, offsets, dst_native_tiling,
                                layout.implicit_dim());
    } else if (dst_bitwidth == 32 &&
               default_tiling_[0] % layout.tiling()[0] == 0 &&
               default_tiling_[1] == layout.tiling()[1]) {
      // All layouts that subdivide the rows of the result native tiling evenly
      // can be handled uniformly with the default case, by preserving the
      // tiling through the op.
      // TODO(jevinjiang): we can relax this for non-32bit as well.
      src_layout = layout;
      dst_layout = VectorLayout(dst_bitwidth, layout.offsets(),
                                src_layout->tiling(), layout.implicit_dim());
    } else if (layout.packing() > target_shape_[0]) {
      // When the input dtype has packing greater than the sublane count, we
      // can't preserve its native tiling in the output (the tile would be too
      // big to fit in a vreg). At the same time, we can't use the default
      // tiling either, because the tile size in the input dtype is smaller than
      // a sublane.
      // For example, for int2 on the target with 8 sublanes, subelements are
      // unpacked into 16 consecutive sublanes.
      // TODO(b/401624977): Perhaps there is a better layout for this case, or
      // if it's impossible, such layout should be used everywhere for int2, not
      // just ExtOp.
      std::array<int64_t, 2> src_native_tiling = nativeTiling(src_bitwidth);
      std::array<int64_t, 2> dst_native_tiling = nativeTiling(dst_bitwidth);
      LayoutOffsets src_offsets = {
          layout.offsets()[0] ? *layout.offsets()[0] % src_native_tiling[0]
                              : LayoutOffset(),
          layout.offsets()[1] ? *layout.offsets()[1] % src_native_tiling[1]
                              : LayoutOffset()};
      LayoutOffsets dst_offsets = {
          layout.offsets()[0] ? *layout.offsets()[0] % dst_native_tiling[0]
                              : LayoutOffset(),
          layout.offsets()[1] ? *layout.offsets()[1] % dst_native_tiling[1]
                              : LayoutOffset()};
      src_layout = VectorLayout(src_bitwidth, src_offsets, src_native_tiling,
                                layout.implicit_dim());
      dst_layout = VectorLayout(dst_bitwidth, dst_offsets, dst_native_tiling,
                                layout.implicit_dim());
    } else {
      LayoutOffsets offsets = {
          layout.offsets()[0] ? *layout.offsets()[0] % default_tiling_[0]
                              : LayoutOffset(),
          layout.offsets()[1] ? *layout.offsets()[1] % default_tiling_[1]
                              : LayoutOffset()};
      src_layout = VectorLayout(src_bitwidth, offsets, default_tiling_,
                                layout.implicit_dim());
      dst_layout = VectorLayout(dst_bitwidth, offsets, default_tiling_,
                                layout.implicit_dim());
    }
    setLayout(op, src_layout, dst_layout);
    return success();
  }

  LogicalResult inferTrunc(Operation *op) {
    TPU_CHECK_OP(op->getNumOperands() == 1, "expect 1 operand");
    TPU_CHECK_OP(op->getNumResults() == 1, "expect 1 result");
    auto src_ty = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op->getResult(0).getType());
    auto some_layout = getLayout(op->getOperand(0));
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    bool select_native = allUsersRequireNativeTiling(op->getResult(0));
    // We might want to reconsider enabling native this aggressively in cases
    // when it would introduce a lot of padding (e.g. when the value only has
    // a small second minor size, but large minor size).
    if (dst_ty.getElementTypeBitWidth() == 16) {
      // TPUv6 has good support for compute in 16-bit and cheap retiling between
      // large 2nd minor and the default tiling, so we bias towards large tiles.
      select_native |= hardware_generation_ >= 6 ||
                       tpu_tiling_flags_.use_x16_large_second_minor;
    } else if (dst_ty.getElementTypeBitWidth() == 8) {
      select_native |= tpu_tiling_flags_.use_x8_large_second_minor;
    } else if (dst_ty.getElementTypeBitWidth() == 4) {
      select_native |= tpu_tiling_flags_.use_x4_large_second_minor;
    } else if (dst_ty.getElementTypeBitWidth() == 2) {
      // Force it to native tiling. See comments in `inferExt`.
      select_native = true;
    } else {
      return op->emitOpError("Unsupported target bitwidth for truncation");
    }
    auto src_layout =
        VectorLayout(layout.bitwidth(), layout.offsets(),
                     nativeTiling(layout.bitwidth()), layout.implicit_dim());
    auto dst_layout = VectorLayout(
        dst_ty.getElementTypeBitWidth(), layout.offsets(),
        select_native ? nativeTiling(dst_ty.getElementTypeBitWidth())
                      : src_layout.tiling(),
        layout.implicit_dim());
    setLayout(op, src_layout, dst_layout);
    return success();
  }

  LogicalResult inferElementwise(Operation *op) {
    TPU_CHECK_OP(op->getNumResults() == 1, "only one result supported");
    TPU_CHECK_OP(op->getNumOperands() > 0,
                 "elementwise ops with no operands unsupported");
    // Elementwise operators can be parameterized by both scalars and shaped
    // types, so make sure we infer layout based on a shaped-typed operand.
    std::optional<VectorLayout> out_layout_candidate;
    std::optional<VectorLayout> out_layout;
    SmallVector<std::optional<Layout>, 4> in_layouts;
    int64_t bitwidth = -1;
    // Find the bitwidth of the operands/results. They must all be the same
    // except for the case of i1s, which use a "fake" bitwidth for layouts.
    // They can be relayouted (in principle) to any other fake bitwidth, so we
    // don't commit to their bitwidth. See comments in VectorLayout class.
    for (Value val : llvm::concat<Value>(op->getOperands(), op->getResults())) {
      if (const VectorType vty = dyn_cast<VectorType>(val.getType())) {
        const int64_t val_bitwidth = vty.getElementTypeBitWidth();
        if (val_bitwidth != 1) {
          if (bitwidth == -1) {
            bitwidth = val_bitwidth;
          } else if (bitwidth != val_bitwidth) {
            return op->emitOpError(
                "Mismatched bitwidth in elementwise for non-i1 "
                "operands/results");
          }
        }
      }
    }
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (auto vty = dyn_cast<VectorType>(op->getOperand(i).getType())) {
        auto some_layout = getLayout(op->getOperand(i));
        TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
        auto &layout = *some_layout;
        if (bitwidth == -1) {
          // All operands/results are i1s, just commit to the first bitwidth
          DCHECK(!out_layout.has_value());
          bitwidth = layout.bitwidth();
          out_layout = layout;
          in_layouts.push_back(layout);
        } else if (bitwidth != layout.bitwidth()) {
          DCHECK_EQ(vty.getElementTypeBitWidth(), 1);
          in_layouts.push_back(std::nullopt);
        } else if (is_fully_replicated(some_layout)) {
          // If the input is fully replicated, don't use it to commit to any
          // layout. Replicated values are easy to relayout.
          in_layouts.push_back(std::nullopt);
          out_layout_candidate = layout;
        } else if (!out_layout) {
          // TODO(apaszke): There are probably smarter ways to choose layout.
          out_layout = layout;
          in_layouts.push_back(some_layout);
        } else {
          if (auto new_out =
                  VectorLayout::join(layout, *out_layout, vty.getShape())) {
            out_layout = *new_out;
            in_layouts.push_back(some_layout);
          } else {
            // When we detect a layout conflict we cannot reconcile, we remove
            // any replication bits that might have been present in out_layout,
            // since there is no guarantee that the conflicting inputs could
            // even become replicated.
            DCHECK_EQ(out_layout->bitwidth(), bitwidth);
            out_layout =
                VectorLayout(bitwidth,
                             {out_layout->offsets()[0].value_or(0),
                              out_layout->offsets()[1].value_or(0)},
                             out_layout->tiling(), out_layout->implicit_dim());
            in_layouts.push_back(std::nullopt);
          }
        }
      } else {
        TPU_CHECK_OP(op->getOperand(i).getType().isSignlessIntOrIndexOrFloat(),
                     "expected only vector and scalar operands");
        in_layouts.push_back({kNoLayout});
      }
    }
    Layout final_out_layout = std::nullopt;
    if (auto out_vty = dyn_cast<VectorType>(op->getResult(0).getType())) {
      if (out_layout) {
        final_out_layout = *out_layout;
      } else if (out_layout_candidate) {
        final_out_layout = *out_layout_candidate;
      } else {
        op->emitOpError(
            "Elementwise op has no vector operands but returns a vector?");
        return failure();
      }
    }
    CHECK_EQ(in_layouts.size(), op->getNumOperands()) << Print(op);
    SmallVector<Layout, 4> final_in_layouts;
    for (int i = 0; i < in_layouts.size(); ++i) {
      if (in_layouts[i]) {
        final_in_layouts.push_back(*in_layouts[i]);
      } else {
        final_in_layouts.push_back(final_out_layout);
      }
    }
    setLayout(op, final_in_layouts, final_out_layout);
    return success();
  }

  LogicalResult infer(tpu::PRNGRandomBitsOp op) {
    auto res_ty = dyn_cast<VectorType>(op->getResult(0).getType());
    TPU_CHECK_OP(res_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit random bit generation supported");
    // TODO: b/342054464 - Support implicit dims for PRNGRandomBitsOp.
    LayoutOffsets offsets = {0, 0};
    setOutLayout(
        op, VectorLayout(kNativeBitwidth, offsets,
                         nativeTiling(kNativeBitwidth), ImplicitDim::kNone));
    return success();
  }

  bool allUsersRequireNativeTiling(Value x) {
    for (Operation *user : getNontrivialTransitiveUsers(x)) {
      if (isa<tpu::MatmulOp>(user)) {
        continue;
      }
      if (auto reduce = dyn_cast<vector::MultiDimReductionOp>(user)) {
        bool reduces_tiled_dims = false;
        for (int64_t dim : reduce.getReductionDims()) {
          if (dim >= reduce.getSourceVectorType().getRank() - 2) {
            reduces_tiled_dims = true;
            break;
          }
        }
        if (reduces_tiled_dims) {
          continue;
        }
      }
      if (auto transpose = dyn_cast<tpu::TransposeOp>(user)) {
        auto perm = transpose.getPermutation();
        auto rank = perm.size();
        // Only permutations that actually swap the last two dims need it.
        if (rank >= 2 && perm[rank - 1] == rank - 2 &&
            perm[rank - 2] == rank - 1) {
          continue;
        }
        // Fall through.
      }
      if (auto store = dyn_cast<vector::StoreOp>(user)) {
        auto maybe_tiling = verifyMemoryTiling(
            store, getMemRefLayout(store.getBase()).getTiles(),
            store.getMemRefType().getRank(),
            store.getMemRefType().getElementTypeBitWidth());
        if (maybe_tiling) {
          auto tiling = *maybe_tiling;
          if (tiling ==
              nativeTiling(store.getMemRefType().getElementTypeBitWidth())) {
            continue;
          }
        }
        // Fall through.
      }
      return false;
    }
    return true;
  }

  LogicalResult assumeLayoutsForBlockArgs(Block &block,
                                          ArrayRef<Layout> layouts) {
    auto op = block.getParentOp();
    if (layouts.size() != block.getNumArguments()) {
      return op->emitOpError(
          "Block arguments must have the same number of layouts");
    }
    // Use tpu.assume_layout to annotate every block argument with the layout of
    // the corresponding operand and replace all uses of the block argument with
    // the result of tpu.assume_layout.
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(op->getLoc(), &block);
    for (auto [iter_arg, layout] :
         llvm::zip_equal(block.getArguments(), layouts)) {
      if (!dyn_cast<VectorType>(iter_arg.getType())) {
        continue;
      }
      if (llvm::any_of(iter_arg.getUsers(), [](Operation *user) {
            return isa<tpu::AssumeLayoutOp>(user);
          })) {
        return op->emitOpError("Expected no assume layout for block arguments");
      }
      auto assume_layout_op =
          builder.create<AssumeLayoutOp>(iter_arg.getType(), iter_arg);
      setLayout(assume_layout_op, layout, layout);
      iter_arg.replaceUsesWithIf(assume_layout_op, [&](OpOperand &operand) {
        return operand.getOwner() != assume_layout_op;
      });
    }
    return success();
  }

  void clearBlockLayouts(Block &block) {
    block.walk([&](Operation *op) {
      // We need to remove assume_layout ops in each block. Otherwise, we will
      // create extra assume_layout ops for nested blocks.
      if (auto assume_op = dyn_cast<tpu::AssumeLayoutOp>(op)) {
        assume_op.getResult().replaceAllUsesWith(assume_op.getInput());
        assume_op->erase();
        return WalkResult::advance();
      }
      op->removeAttr("in_layout");
      op->removeAttr("out_layout");
      return WalkResult::advance();
    });
  }

  Layout getLayout(Value v) {
    auto op = v.getDefiningOp();
    CHECK(op);
    auto op_result = dyn_cast<OpResult>(v);
    CHECK(op_result);
    auto result_index = op_result.getResultNumber();
    auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
    CHECK(out_attrs.size() > result_index);
    auto layout = cast<VectorLayoutAttr>(out_attrs[result_index]).getLayout();
    if (force_first_tile_offsets_ &&
        layout->offsets()[1].value_or(0) >= layout->tiling()[1]) {
      // Force the out-of-first-tile offset to be zero.
      layout = VectorLayout(layout->bitwidth(), {layout->offsets()[0], 0},
                            layout->tiling(), layout->implicit_dim());
    }
    return layout;
  }

  SmallVector<Layout, 4> getLayoutFromOperands(Operation *op) {
    SmallVector<Layout, 4> layouts;
    layouts.reserve(op->getNumOperands());
    for (const auto &operand : op->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        layouts.push_back(getLayout(operand));
      } else {
        layouts.push_back(kNoLayout);
      }
    }
    return layouts;
  }

 private:
  std::optional<absl::Span<const int64_t>> verifyMemoryTiling(
      Operation *op, ArrayRef<xla::Tile> mem_tiling, int64_t rank,
      int8_t bitwidth) {
    const int packing = kNativeBitwidth / bitwidth;
    if (bitwidth == 32) {
      if (mem_tiling.size() != 1) {
        op->emitOpError("Only one-level tiling supported for 32-bit loads");
        return std::nullopt;
      }
    } else if (bitwidth < 32) {
      int64_t rows_per_tile;
      if (rank == 1) {
        if (mem_tiling.size() != 3) {
          op->emitOpError(
              "Only three-level tiling supported for 1D memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        auto second = mem_tiling[1].dimensions();
        if (first.size() != 1 || first[0] % (packing * target_shape_[1]) != 0) {
          op->emitOpError("Invalid first-level tile in 1D memory op");
          return std::nullopt;
        }
        rows_per_tile = first[0] / target_shape_[1];
        if (second.size() != 1 || second[0] != target_shape_[1]) {
          op->emitOpError("Invalid second-level tile in 1D memory op");
          return std::nullopt;
        }
      } else {
        if (mem_tiling.size() != 2) {
          op->emitOpError(
              "Only two-level tiling supported for 2D+ memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        rows_per_tile = first[0];
      }
      auto row_compressed = mem_tiling[mem_tiling.size() - 1].dimensions();
      if (row_compressed.size() != 2) {
        op->emitOpError("Expected 2D tiling for packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] != (32 / bitwidth) || row_compressed[1] != 1) {
        op->emitOpError("Expected compressed packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] > rows_per_tile) {
        op->emitOpError("Packing cannot introduce padding");
        return std::nullopt;
      }
    } else {
      op->emitOpError("Loads of types wider than 32-bit unsupported");
      return std::nullopt;
    }
    return mem_tiling[0].dimensions();
  }

  std::array<int64_t, 2> nativeTiling(int8_t bitwidth) {
    return {default_tiling_[0] * kNativeBitwidth / bitwidth,
            default_tiling_[1]};
  }

  int hardware_generation_;
  std::array<int64_t, 2> target_shape_;
  std::array<int64_t, 2> default_tiling_;
  TpuTilingFlags tpu_tiling_flags_;

  // TODO(b/342235360): Deprecate force_first_tile_offsets_ once we fully
  // remove the restriction that offsets must fall within the first tile.
  bool force_first_tile_offsets_ = false;

  // TODO(apaszke): This is not really native on newer generations of TPUs.
  // Get rid of this temporary stopgap.
  static constexpr int8_t kNativeBitwidth = 32;
};

struct InferVectorLayoutPass
    : public impl::InferVectorLayoutPassBase<InferVectorLayoutPass> {
  InferVectorLayoutPass(int hardware_generation,
                        std::array<int64_t, 2> target_shape,
                        TpuTilingFlags tpu_tiling_flags) {
    this->hardware_generation = hardware_generation;
    this->sublane_count = target_shape[0];
    this->lane_count = target_shape[1];
    this->tpu_tiling_flags = tpu_tiling_flags;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      getOperation().emitError("hardware_generation must be set")
          << hardware_generation;
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    VectorLayoutInferer run(hardware_generation, {sublane_count, lane_count},
                            tpu_tiling_flags);
    if (run.infer(func).failed()) {
      signalPassFailure();
    }
  }

  TpuTilingFlags tpu_tiling_flags;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferVectorLayoutPass(
    int hardware_generation, std::array<int64_t, 2> target_shape,
    const TpuTilingFlags &tpu_tiling_flags) {
  return std::make_unique<InferVectorLayoutPass>(
      hardware_generation, target_shape, tpu_tiling_flags);
}

}  // namespace mlir::tpu
