/* Copyright 2025 The OpenXLA Authors.

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
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeSize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace xla::cpu {

#define GEN_PASS_DECL_ADDLOOPUNROLLFLAGSPASS
#define GEN_PASS_DEF_ADDLOOPUNROLLFLAGSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

class AddLoopUnrollFlagsPass
    : public impl::AddLoopUnrollFlagsPassBase<AddLoopUnrollFlagsPass> {
 public:
  using AddLoopUnrollFlagsPassBase::AddLoopUnrollFlagsPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp func_op = getOperation();
    mlir::MLIRContext* context = func_op.getContext();

    llvm::DenseMap<mlir::scf::ForOp, int64_t> nested_iteration_bits;
    func_op->walk([&](mlir::scf::ForOp for_op) {
      RecursiveWalk(for_op, nested_iteration_bits);
      return mlir::WalkResult::skip();
    });

    auto loop_unroll = mlir::LLVM::LoopUnrollAttr::get(
        context,
        /*disable=*/mlir::BoolAttr::get(context, true), /*count=*/nullptr,
        /*runtimeDisable=*/nullptr,
        /*full=*/nullptr,
        /*followupUnrolled=*/nullptr, /*followupRemainder=*/nullptr,
        /*followupAll=*/nullptr);
    auto loop_annotation = mlir::LLVM::LoopAnnotationAttr::get(
        context,
        /*disableNonforced=*/nullptr, /*vectorize=*/nullptr,
        /*interleave=*/nullptr, loop_unroll,
        /*unrollAndJam=*/nullptr, /*licm=*/nullptr, /*distribute=*/nullptr,
        /*pipeline=*/nullptr, /*peeled=*/nullptr, /*unswitch=*/nullptr,
        /*mustProgress=*/nullptr,
        /*isVectorized=*/nullptr,
        /*startLoc=*/nullptr,
        /*endLoc=*/nullptr,
        /*parallelAccesses=*/std::nullopt);

    for (auto& [for_op, bits] : nested_iteration_bits) {
      if (bits >= 256) {
        for_op->setAttr(mlir::LLVM::LoopAnnotationAttr::getMnemonic(),
                        loop_annotation);
      }
    }
  }

 private:
  static int64_t MinElementBits(mlir::scf::ForOp& for_op) {
    mlir::DataLayout data_layout = mlir::DataLayout::closest(for_op);

    int64_t min_element_bytes = std::numeric_limits<int64_t>::max();
    for (mlir::Type type : for_op.getResultTypes()) {
      if (auto tensor_type = mlir::dyn_cast<mlir::TensorType>(type)) {
        llvm::TypeSize size =
            data_layout.getTypeSizeInBits(tensor_type.getElementType());
        int64_t element_bytes = size.getFixedValue();
        min_element_bytes = std::min(min_element_bytes, element_bytes);
      }
    }

    return min_element_bytes;
  }

  static int64_t RecursiveWalk(
      mlir::scf::ForOp for_op,
      llvm::DenseMap<mlir::scf::ForOp, int64_t>& nested_iteration_bits) {
    auto lb = for_op.getLowerBound();
    auto ub = for_op.getUpperBound();
    auto step = for_op.getStep();

    std::optional<int64_t> this_trip_count =
        mlir::constantTripCount(lb, ub, step);

    if (!this_trip_count.has_value()) {
      return 0;
    }

    int64_t this_trip_bytes = *this_trip_count * MinElementBits(for_op);

    int64_t nested_bits = 0;
    for_op.getBody()->walk([&](mlir::scf::ForOp for_op) {
      nested_bits += RecursiveWalk(for_op, nested_iteration_bits);
      return mlir::WalkResult::skip();
    });

    nested_iteration_bits.insert({for_op, nested_bits});

    if (nested_bits == 0) {
      return this_trip_bytes;
    }

    return this_trip_bytes * nested_bits;
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateAddLoopUnrollFlagsPass(
    int32_t max_nested_bits) {
  AddLoopUnrollFlagsPassOptions options;
  options.max_nested_bits_ = max_nested_bits;
  return std::make_unique<AddLoopUnrollFlagsPass>(options);
}

}  // namespace xla::cpu
