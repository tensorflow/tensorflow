/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_library_ops.h"

#include <cstdint>

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/xla_gpu_api.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"

namespace xla::gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

using arith::ConstantIntOp;
using arith::ConstantOp;

//===----------------------------------------------------------------------===//
// Converts lmhlo_gpu.gemm op to XLA:GPU runtime calls
//===----------------------------------------------------------------------===//

TypedValue<DotDimensionNumbersType> getDotDimensionNumbers(
    XlaGpuApi &api, ImplicitLocOpBuilder &b, ModuleOp module,
    lmhlo_gpu::GEMMOp op) {
  mhlo::DotDimensionNumbersAttr attr = op.getDotDimensionNumbersAttr();
  SmallVector<Value> args = {
      api.getI32List(b, attr.getLhsBatchingDimensions()),
      api.getI32List(b, attr.getRhsBatchingDimensions()),
      api.getI32List(b, attr.getLhsContractingDimensions()),
      api.getI32List(b, attr.getRhsContractingDimensions()),
  };

  auto api_func = api.getCreateDotDimensionsNumbers(b, module);
  auto call = b.create<func::CallOp>(api_func.getSymName(),
                                     api_func.getResultTypes(), args);

  return call.getResult(0).cast<TypedValue<DotDimensionNumbersType>>();
}

TypedValue<DotPrecisionType> getDotPrecision(XlaGpuApi &api,
                                             ImplicitLocOpBuilder &b,
                                             ModuleOp module,
                                             lmhlo_gpu::GEMMOp op) {
  SmallVector<int64_t> precision = llvm::to_vector(
      llvm::map_range(op.getPrecisionConfigAttr(), [](Attribute attr) {
        auto value = attr.cast<mhlo::PrecisionAttr>().getValue();
        return static_cast<int64_t>(value);
      }));

  SmallVector<Value> args = {api.getI32List(b, precision)};

  auto api_func = api.getCreateDotPrecision(b, module);
  auto call = b.create<func::CallOp>(api_func.getSymName(),
                                     api_func.getResultTypes(), args);

  return call.getResult(0).cast<TypedValue<DotPrecisionType>>();
}

TypedValue<DotConfigType> getDotConfig(XlaGpuApi &api, ImplicitLocOpBuilder &b,
                                       ModuleOp module, lmhlo_gpu::GEMMOp op) {
  int32_t algorithm = op.getAlgorithm().value_or(-1);

  SmallVector<Value> args = {b.create<ConstantIntOp>(algorithm, 32),
                             b.create<ConstantOp>(op.getAlphaRealAttr()),
                             b.create<ConstantOp>(op.getAlphaImagAttr()),
                             b.create<ConstantOp>(op.getBetaAttr()),
                             getDotDimensionNumbers(api, b, module, op),
                             getDotPrecision(api, b, module, op)};

  auto api_func = api.getCreateDotConfig(b, module);
  auto call = b.create<func::CallOp>(api_func.getSymName(),
                                     api_func.getResultTypes(), args);

  return call.getResult(0).cast<TypedValue<DotConfigType>>();
}

TypedValue<TraceType> getTrace(XlaGpuApi &api, ImplicitLocOpBuilder &b,
                               ModuleOp module, lmhlo_gpu::GEMMOp op) {
  // Get original HLO operation name from the location.
  Value hlo_op = b.create<IREE::Input::ByteBufferConstantOp>(
      b.getType<IREE::Input::ByteBufferType>(),
      /*name=*/b.getStringAttr("hlo_op"),
      /*value=*/mhlo::GetDebugNameFromLocation(op.getLoc()),
      /*alignment=*/nullptr, /*mime_type=*/nullptr);

  auto api_func = api.getCreateTrace(b, module);
  auto call = b.create<func::CallOp>(
      api_func.getSymName(), api_func.getResultTypes(), ValueRange(hlo_op));

  return call.getResult(0).cast<TypedValue<TraceType>>();
}

TypedValue<ExecutionContextType> getExecutionContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func.getArguments().front().cast<TypedValue<ExecutionContextType>>();
}

struct ConvertGemmOp : public OpConversionPattern<lmhlo_gpu::GEMMOp> {
  ConvertGemmOp(TypeConverter &converter, MLIRContext *ctx,
                DeBufferization &state, XlaGpuApi &api)
      : OpConversionPattern(converter, ctx), state(state), api(api) {}

  LogicalResult matchAndRewrite(
      lmhlo_gpu::GEMMOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto *block = op->getBlock();
    auto module = op->getParentOfType<ModuleOp>();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto dot_config = getDotConfig(api, b, module, op);
    auto trace = getTrace(api, b, module, op);

    // Export arguments to buffer views.
    auto lhs = state.remapped[block][op.getA()];
    auto rhs = state.remapped[block][op.getB()];
    auto out = state.remapped[block][op.getC()];

    if (!lhs || !rhs || !out) {
      return rewriter.notifyMatchFailure(
          op, "missing memref to tensor mapping for lmhlo_gpu.gemm arguments");
    }

    // Arguments to a gemm dispatch.
    SmallVector<Value> args = {getExecutionContext(op),
                               api.getBufferView(b, lhs),
                               api.getBufferView(b, rhs),
                               api.getBufferView(b, out),
                               dot_config,
                               trace};

    // TODO(ezhulenev): Should we import buffer view back and update remapping?
    auto api_func = api.getDispatchGemm(b, module);
    b.create<func::CallOp>(api_func.getSymName(), api_func.getResultTypes(),
                           args);

    rewriter.eraseOp(op);
    return success();
  }

  DeBufferization &state;
  XlaGpuApi &api;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateLibraryOpsConversionPatterns(RewritePatternSet &patterns,
                                          TypeConverter &converter,
                                          DeBufferization &state,
                                          XlaGpuApi &api) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertGemmOp>(converter, ctx, state, api);
}

}  // namespace xla::gpu
