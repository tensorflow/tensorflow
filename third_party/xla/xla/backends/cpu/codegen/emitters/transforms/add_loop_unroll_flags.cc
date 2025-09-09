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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeSize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

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

    llvm::DenseMap<mlir::scf::ForOp, int64_t> nested_iteration_map;
    func_op->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp for_op) {
      RecursiveWalk(for_op, nested_iteration_map);
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
        /*parallelAccesses=*/{});

    for (auto& [for_op, nested_iterations] : nested_iteration_map) {
      if (nested_iterations > max_nested_iterations_) {
        for_op->setAttr(mlir::LLVM::LoopAnnotationAttr::getMnemonic(),
                        loop_annotation);
      }
    }
  }

 private:
  // Recursively insert the number of nested accessed bits for each loop.
  static int64_t RecursiveWalk(
      mlir::scf::ForOp for_op,
      llvm::DenseMap<mlir::scf::ForOp, int64_t>& nested_iteration_map) {
    auto lb = for_op.getLowerBound();
    auto ub = for_op.getUpperBound();
    auto step = for_op.getStep();

    std::optional<int64_t> this_trip_count =
        mlir::constantTripCount(lb, ub, step);

    if (!this_trip_count.has_value()) {
      return 0;
    }

    int64_t nested_iterations = 0;
    for_op.getBody()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::scf::ForOp for_op) {
          nested_iterations += RecursiveWalk(for_op, nested_iteration_map);
          return mlir::WalkResult::skip();
        });

    nested_iteration_map.insert({for_op, nested_iterations});

    if (nested_iterations == 0) {
      return *this_trip_count;
    }

    return *this_trip_count * nested_iterations;
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateAddLoopUnrollFlagsPass(
    int32_t max_nested_iterations) {
  AddLoopUnrollFlagsPassOptions options;
  options.max_nested_iterations_ = max_nested_iterations;
  return std::make_unique<AddLoopUnrollFlagsPass>(options);
}

}  // namespace xla::cpu
