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

// This file implements logic for lowering LHLO GPU dialect to TFRT CUDA
// dialect.

#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gpu_passes.h"

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/PassDetail.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/ccl_pattern.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/custom_call_pattern.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gemm_pattern.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/memcpy_pattern.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/pass/pass.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct LmhloGpuAsyncConversionPass
    : public LmhloGpuAsyncConversionPassBase<LmhloGpuAsyncConversionPass> {
 private:
  void runOnFunction() override {
    auto* context = &getContext();

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    auto buffer_type = tfrt::gpu::BufferType::get(context);
    converter.addConversion([&](BaseMemRefType) { return buffer_type; });

    ConversionTarget target(*context);
    target
        .addIllegalDialect<lmhlo_gpu::LmhloGpuDialect, mlir::gpu::GPUDialect>();
    target.addLegalDialect<tfrt::compiler::TFRTDialect, tfrt::gpu::GpuDialect,
                           xla::gpu::XlirDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getType()) &&
             converter.isLegal(&op.body());
    });
    target.addDynamicallyLegalOp<tfrt::gpu::conversion::AsyncExecuteOp>(
        [&](tfrt::gpu::conversion::AsyncExecuteOp op) {
          return converter.isLegal(&op.body());
        });

    RewritePatternSet patterns(context);
    populateCclConversionPattern(patterns);
    populateCustomCallConversionPattern(patterns);
    populateGemmConversionPattern(patterns);
    populateMemcpyConversionPattern(patterns);
    populateFuncOpTypeConversionPattern(patterns, converter);

    ConversionTarget wrap_target(*context);
    wrap_target
        .addLegalDialect<lmhlo_gpu::LmhloGpuDialect, mlir::gpu::GPUDialect>();
    wrap_target.addLegalOp<lmhlo::AllGatherOp, lmhlo::AllReduceOp,
                           lmhlo::ReduceScatterOp, lmhlo::AllToAllOp,
                           lmhlo::CollectivePermuteOp, lmhlo::CustomCallOp>();
    tfrt::gpu::populateGpuAsyncConversionPatterns(patterns, converter,
                                                  wrap_target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createLmhloGpuAsyncConversionPass() {
  return std::make_unique<LmhloGpuAsyncConversionPass>();
}

}  // namespace tensorflow
