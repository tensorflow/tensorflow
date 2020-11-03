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

#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct PropagateTensorFlowABIKnowledgePass
    : public PropagateTensorFlowABIKnowledgePassBase<
          PropagateTensorFlowABIKnowledgePass> {
  explicit PropagateTensorFlowABIKnowledgePass(
      llvm::ArrayRef<uint32_t> same_shape) {
    same_shape_ = same_shape;
  }

  void runOnOperation() override {
    // We know due to tensorflow ABI that the offset is always 0 and that the
    // innermost stride is always 1. To make this visible to the compiler,
    // we insert constants into the code and replace usages accordingly.
    // We do not change the signature so that we keep a somewhat stable ABI
    // that is easy to undertand by tools.
    // We also know that tensorflow aligns all allocated pointers by 16, so
    // we pass this on. Furthermore, we know that arguments never alias. More
    // precicely, they may only alias (due to reuse) if the kernel does not
    // read from a position it previously has written to. We express this with
    // the noalias attribute.
    mlir::LLVM::LLVMFuncOp func = getOperation();

    // This only works if the function is local and we can rewrite it.
    if (func.isExternal()) return;

    auto function_list =
        func.getParentOfType<ModuleOp>().getOps<mlir::FuncOp>();
    if (function_list.empty()) {
      func.emitError() << "No possible kernel function found";
      return signalPassFailure();
    }
    auto func_iterator = function_list.begin();
    if (std::next(func_iterator) != function_list.end()) {
      func.emitError() << "More than one possible kernel function detected";
      return signalPassFailure();
    }
    // Note that this dereference is necessary to prevent a
    // stack-use-after-return error.
    auto func_type = (*func_iterator).getType();

    mlir::OpBuilder b(func.getBody());
    // Steal the LLVM representation of the index type from the third argument.
    auto index_type = func.getArgument(3).getType();
    mlir::Value one = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 1));
    mlir::Value zero = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 0));
    uint32_t arg_pos = 0;
    std::vector<uint32_t> positions;
    // Collect the agument and return types of the surrounding function.
    auto arg_types = llvm::to_vector<4>(llvm::concat<const mlir::Type>(
        func_type.getInputs(), func_type.getResults()));
    for (mlir::Type arg_type : arg_types) {
      if (!arg_type.isa<mlir::MemRefType>()) {
        func.emitError() << "argument of surrounding func is not ranked memref";
        return signalPassFailure();
      }
      positions.push_back(arg_pos);
      // Set alignment and aliasing on the pointers.
      func.setArgAttr(arg_pos + 1, "llvm.noalias", b.getBoolAttr(true));
      func.setArgAttr(arg_pos + 1, "llvm.align", b.getIndexAttr(16));
      // Replace the offset with zero. Offset is argument number 3.
      func.getArgument(arg_pos + 2).replaceAllUsesWith(zero);
      // Forward over base_ptr, aligned_ptr, offset, size and stride arguments.
      arg_pos += 3 + arg_type.cast<mlir::MemRefType>().getRank() * 2;
      // Replace the last stride with constant 1.
      func.getArgument(arg_pos - 1).replaceAllUsesWith(one);
    }

    // If we have knowledge that some arguments have the same shape, we
    // can use that here. Simply replace usages of the shape parameters within
    // the function body to a single shape parameter.
    if (same_shape_.empty()) {
      return;
    }
    auto first = same_shape_.front();
    auto first_offset = positions.at(first);
    auto first_type = arg_types[first].cast<mlir::ShapedType>();
    uint32_t rank = first_type.getRank();
    for (int i = 1, e = same_shape_.size(); i < e; ++i) {
      uint32_t same = same_shape_[i];
      uint32_t same_offset = positions.at(same);
      auto same_type = arg_types[same].cast<mlir::ShapedType>();
      if (same_type.getRank() != rank) {
        func.emitOpError() << "same shape constraints on arguments with "
                              "non-matching shapes: #"
                           << first << " and #" << same;
        return signalPassFailure();
      }

      for (uint32_t i = 0; i < 2 * rank; ++i) {
        // Replace uses for second arg data with first arg.
        auto same_arg = func.getArgument(same_offset + 3 + i);
        auto first_arg = func.getArgument(first_offset + 3 + i);
        same_arg.replaceAllUsesWith(first_arg);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::LLVM::LLVMFuncOp>>
CreatePropagateTensorFlowABIKnowledgePass(llvm::ArrayRef<uint32_t> same_shape) {
  return std::make_unique<PropagateTensorFlowABIKnowledgePass>(same_shape);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
