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

// This file contains the analysis and transformation to rewrite kernel
// functions such that information about alignment, aliasing and zero offsets
// steming from the tf_framework uses is propagated.

#include <memory>

#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct PropagateTfAbiKnowledgeToKernelsPass
    : public PropagateTfAbiKnowledgeToKernelsBase<
          PropagateTfAbiKnowledgeToKernelsPass> {
  void runOnFunction() override {
    FuncOp function = getFunction();
    // We currently only handle entry functions and do not propagate across
    // functions.
    if (function.getAttrOfType<mlir::UnitAttr>(
            tf_framework::TFFrameworkDialect::kTFEntryAttrName)) {
      // For all operands of this function, we know they are aligned. Also, by
      // construction of kernel generator, we know that there is no offset.
      // TODO(herhut): Insert asserts in debug mode to check this.
      for (auto argument : function.getArguments()) {
        if (argument.getType().isa<BaseMemRefType>()) {
          allocated_by_runtime.insert(argument);
        }
      }
    }

    // For locally allocated values, we know they are aligned and have offset
    // zero. Further, they also do not alias with other memrefs, except in
    // benign ways. This is ensured by the reuse analysis.
    function.walk([&](tf_framework::TFAllocOp op) {
      Value allocated = op.getResult();
      allocated_by_runtime.insert(allocated);
      no_alias.insert(allocated);
    });

    // Next, take what we have and propagate it through known operations.
    propagateThroughUses();

    // Now look at launches and make use of the knowledge we have.
    function.walk([&](gpu::LaunchFuncOp launch) {
      auto module = launch.getParentOfType<ModuleOp>();
      auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(launch.kernel());

      if (!kernel || kernel.isExternal()) return;

      // Count the position of kernel operands independently, as they do not
      // coincide with laucnh operands as memref parameters get expanded when
      // lowered to llvm.
      int kernel_p = 0;
      OpBuilder b = OpBuilder::atBlockBegin(&kernel.body().front());
      Value zero;
      Value one;
      auto loc = kernel.getLoc();
      for (auto operand : launch.operands()) {
        auto memref = operand.getType().dyn_cast<MemRefType>();
        if (!memref) {
          // Scalar argument, advance kernel position by one.
          kernel_p++;
          continue;
        }
        if (allocated_by_runtime.contains(operand)) {
          // This was allocated by the tf runtime, so it is aligned, has no
          // offset, an inner stride of 1 and the two pointers in the descriptor
          // coincide. Rewrite the kernel accordingly.
          Value alloc_ptr = kernel.getArgument(kernel_p);
          Value align_ptr = kernel.getArgument(kernel_p + 1);
          alloc_ptr.replaceAllUsesWith(align_ptr);
          Value offset = kernel.getArgument(kernel_p + 2);
          if (!zero) {
            zero = b.create<LLVM::ConstantOp>(loc, offset.getType(),
                                              b.getIndexAttr(0));
          }
          offset.replaceAllUsesWith(zero);
          // The stride is the last argument belonging to this memref.
          Value inner_stride =
              kernel.getArgument(kernel_p + 2 + memref.getRank() * 2);
          if (!one) {
            one = b.create<LLVM::ConstantOp>(loc, offset.getType(),
                                             b.getIndexAttr(1));
          }
          inner_stride.replaceAllUsesWith(one);
          kernel.setArgAttr(
              kernel_p + 1, LLVM::LLVMDialect::getAlignAttrName(),
              b.getIndexAttr(
                  tf_framework::TFFrameworkDialect::kAllocationAlignment));
        }
        if (no_alias.contains(operand)) {
          // TODO(herhut): We also need to check whether any of the other args
          //     are aliases. This is currently never the case by construction
          //     but we could use the alias analysis from buffer placement here
          //     to make sure.
          // Add the no_alias attribute to the correspondign pointer.
          kernel.setArgAttr(kernel_p + 1,
                            LLVM::LLVMDialect::getNoAliasAttrName(),
                            b.getBoolAttr(true));
        }
        // Advance base, aligned, offset, strides and sizes many arguments.
        kernel_p += memref.getRank() * 2 + 3;
      }
    });
  }

 private:
  void propagateThroughUses() {
    llvm::SmallVector<Value, 4> worklist(allocated_by_runtime.begin(),
                                         allocated_by_runtime.end());

    while (!worklist.empty()) {
      Value candidate = worklist.pop_back_val();
      for (auto user : candidate.getUsers()) {
        if (auto reshape = dyn_cast<MemRefReshapeOp>(user)) {
          // Reshape propagates alignment, offset and innermost stride.
          // TODO(herhut): This should be a trait.
          if (allocated_by_runtime.insert(reshape.result()).second) {
            worklist.push_back(reshape.result());
          }
        }
        if (auto cast = dyn_cast<MemRefReinterpretCastOp>(user)) {
          // Check that we have offset 0.
          if (cast.getStaticOffset(0) != 0) continue;
          if (allocated_by_runtime.insert(cast.result()).second) {
            worklist.push_back(cast.result());
          }
        }
      }
    }
  }

  // Set of values that were allocated by the tf runtime and hence are aligned
  // and have no offset.
  llvm::DenseSet<Value> allocated_by_runtime;
  // Set of values we know do not alias other values.
  llvm::DenseSet<Value> no_alias;
};

}  // namespace

std::unique_ptr<FunctionPass> CreatePropagateTfAbiKnowledgeToKernels() {
  return std::make_unique<PropagateTfAbiKnowledgeToKernelsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
