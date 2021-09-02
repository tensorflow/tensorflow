// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- memcpy_pattern.cc
//---------------------------------------------------------===//
//
// Pattern to lower mlir::gpu::memcpy Ops to tfrt cuda dialect.
//
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/memcpy_pattern.h"

#include <assert.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/pass/pass.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

using llvm::ArrayRef;

// Creates tfrt::gpu::MemCopyOp from mlir::gpu::MemcpyOp.
struct MemcpyRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<mlir::gpu::MemcpyOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      mlir::gpu::MemcpyOp>::GpuAsyncOpConversionPattern;

  FailureOr<Value> matchAndRewriteOp(
      mlir::gpu::MemcpyOp op, Value chain, Value stream,
      ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (!all_of(operands, [](Value operand) {
          return operand.getType().isa<tfrt::gpu::BufferType>();
        }))
      return rewriter.notifyMatchFailure(op, "expected buffer operands");

    BlockAndValueMapping mapping;
    for (auto pair : llvm::zip_first(op->getOperands(), operands))
      mapping.map(std::get<0>(pair), std::get<1>(pair));

    rewriter.eraseOp(op);

    return rewriter
        .create<tfrt::gpu::MemCopyOp>(op.getLoc(), mapping.lookup(op.dst()),
                                      mapping.lookup(op.src()), stream, chain)
        .getResult();
  }
};

}  // namespace

void populateMemcpyConversionPattern(RewritePatternSet& patterns) {
  patterns.add<MemcpyRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
