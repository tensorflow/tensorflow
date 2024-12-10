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

#include <memory>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::xla {
namespace {

#define GEN_PASS_DEF_PREVENTMMAV3LOOPUNROLLINGPASS
#include "xla/service/gpu/fusions/triton/xla_triton_passes.h.inc"

struct PreventMmaV3LoopUnrollingPass
    : public impl::PreventMmaV3LoopUnrollingPassBase<
          PreventMmaV3LoopUnrollingPass> {
  // TODO(b/344841434): Remove this if NVIDIA fixes compile-time issue.
  // PTX sometimes unrolls wgmma loops that can cause a 1000x slow down in
  // compilation time. Most unrolling has already been done before PTX;
  // this function adds a 'nounroll' pragma immediately within every scf.for
  // loop that contains at least one triton_nvidia_gpu.warp_group_dot,
  // preventing ptxas from unrolling it.
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mod.walk([&](mlir::scf::ForOp forOp) -> void {
      if (forOp.getOps<mlir::triton::nvidia_gpu::WarpGroupDotOp>().empty()) {
        return;
      }
      auto builder = mlir::OpBuilder::atBlockBegin(forOp.getBody());
      // type, constraints, pack are all unused because the asm doesn't take
      // any arguments or give any results, but we need to set them to
      // something.
      builder.create<mlir::triton::ElementwiseInlineAsmOp>(
          forOp.getLoc(), builder.getI32Type(),
          /*asm_string=*/".pragma \"nounroll\";", /*constraints=*/"=r",
          /*isPure=*/false, /*pack=*/1, mlir::ArrayRef<mlir::Value>{});
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreatePreventMmaV3LoopUnrollingPass() {
  return std::make_unique<PreventMmaV3LoopUnrollingPass>();
}

}  // namespace mlir::triton::xla
