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

#include "tensorflow/compiler/xla/mlir/backends/openxla/conversion/convert_library_ops.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "third_party/iree/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/Input/InputDialect.h"
#include "third_party/iree/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/openxla/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"

namespace xla::gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

using arith::ConstantIndexOp;
using arith::ConstantIntOp;
using arith::ConstantOp;

//===----------------------------------------------------------------------===//
// Helper class to set up OpenXLA runtime API declarations
//===----------------------------------------------------------------------===//

// XLA GPU <-> StreamExecutor integration as API declarations.
class XlaGpuApi {
 public:
  // Imports `@xla_gpu.dot_dimension_numbers.create` into the module.
  func::FuncOp getCreateDotDimensionsNumbers(OpBuilder &b, ModuleOp module);

  // Imports `@xla_gpu.dot_precision.create` into the module.
  func::FuncOp getCreateDotPrecision(OpBuilder &b, ModuleOp module);

  // Imports `@xla_gpu.dot_config.create` into the module.
  func::FuncOp getCreateDotConfig(OpBuilder &b, ModuleOp module);

  // Imports `@xla_gpu.gemm.dispatch` into the module.
  func::FuncOp getDispatchGemm(OpBuilder &b, ModuleOp module);

  // Imports `@xla_gpu.trace.create` into the module.
  func::FuncOp getCreateTrace(OpBuilder &b, ModuleOp module);

 private:
  SymbolTable &symTable(ModuleOp module);

  func::FuncOp addDecl(OpBuilder &b, ModuleOp module, std::string_view name,
                       FunctionType function_type);

  SymbolTableCollection sym_table_;
};

Type getI64ListType(MLIRContext *ctx) {
  return IREE::Input::ListType::get(ctx, IntegerType::get(ctx, 64));
}

func::FuncOp XlaGpuApi::getCreateDotDimensionsNumbers(OpBuilder &b,
                                                      ModuleOp module) {
  auto i64_list = getI64ListType(b.getContext());
  SmallVector<Type> args = {/*lhs_batch_dimensions=*/i64_list,
                            /*rhs_batch_dimensions=*/i64_list,
                            /*lhs_contracting_dimensions=*/i64_list,
                            /*rhs_contracting_dimensions=*/i64_list};
  SmallVector<Type> rets = {b.getType<DotDimensionNumbersType>()};
  return addDecl(b, module, "xla_gpu.dot_dimension_numbers.create",
                 FunctionType::get(b.getContext(), args, rets));
}

func::FuncOp XlaGpuApi::getCreateDotPrecision(OpBuilder &b, ModuleOp module) {
  SmallVector<Type> args = {getI64ListType(b.getContext())};
  SmallVector<Type> rets = {b.getType<DotPrecisionType>()};
  return addDecl(b, module, "xla_gpu.dot_precision.create",
                 FunctionType::get(b.getContext(), args, rets));
}

func::FuncOp XlaGpuApi::getCreateDotConfig(OpBuilder &b, ModuleOp module) {
  SmallVector<Type> args = {b.getI32Type(),  // algorithm
                            b.getF64Type(),  // alpha_real
                            b.getF64Type(),  // alpha_imag
                            b.getF64Type(),  // beta
                            b.getType<DotDimensionNumbersType>(),
                            b.getType<DotPrecisionType>()};
  SmallVector<Type> rets = {b.getType<DotConfigType>()};
  return addDecl(b, module, "xla_gpu.dot_config.create",
                 FunctionType::get(b.getContext(), args, rets));
}

func::FuncOp XlaGpuApi::getDispatchGemm(OpBuilder &b, ModuleOp module) {
  auto execution_context = b.getType<ExecutionContextType>();
  auto buffer_view = b.getType<IREE::Input::BufferViewType>();
  SmallVector<Type> args = {execution_context,
                            buffer_view,  // lhs
                            buffer_view,  // rhs
                            buffer_view,  // out
                            b.getType<DotConfigType>(),
                            b.getType<TraceType>()};
  return addDecl(b, module, "xla_gpu.gemm.dispatch",
                 FunctionType::get(b.getContext(), args, /*rets=*/TypeRange()));
}

func::FuncOp XlaGpuApi::getCreateTrace(OpBuilder &b, ModuleOp module) {
  SmallVector<Type> args = {b.getType<IREE::Input::ByteBufferType>()};
  SmallVector<Type> rets = {b.getType<TraceType>()};
  return addDecl(b, module, "xla_gpu.trace.create",
                 FunctionType::get(b.getContext(), args, rets));
}

SymbolTable &XlaGpuApi::symTable(ModuleOp module) {
  return sym_table_.getSymbolTable(module);
}

func::FuncOp XlaGpuApi::addDecl(OpBuilder &b, ModuleOp module,
                                std::string_view name,
                                FunctionType function_type) {
  if (auto fn = sym_table_.lookupNearestSymbolFrom<func::FuncOp>(
          module, b.getStringAttr(name)))
    return fn;

  Location loc = UnknownLoc::get(module->getContext());

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());

  auto fn = b.create<func::FuncOp>(loc, name, function_type);
  fn.setPrivate();
  symTable(module).insert(fn);
  return fn;
}

//===----------------------------------------------------------------------===//
// Helper functions to build arguments to API functions.
//===----------------------------------------------------------------------===//

// Creates `iree_input.list<i64>` list from the given values.
TypedValue<IREE::Input::ListType> getI64List(ImplicitLocOpBuilder &b,
                                             ArrayRef<int64_t> values) {
  MLIRContext *ctx = b.getContext();

  Value size = b.create<ConstantIndexOp>(values.size());
  Value list = b.create<IREE::Input::ListCreateOp>(getI64ListType(ctx), size);

  if (!values.empty()) b.create<IREE::Input::ListResizeOp>(list, size);
  for (auto indexed : llvm::enumerate(values)) {
    Value index = b.create<ConstantIndexOp>(indexed.index());
    Value value = b.create<ConstantIntOp>(indexed.value(), 64);
    b.create<IREE::Input::ListSetOp>(list, index, value);
  }

  return list.cast<TypedValue<IREE::Input::ListType>>();
}

//===----------------------------------------------------------------------===//
// Converts lmhlo_gpu.gemm op to OpenXLA runtime calls
//===----------------------------------------------------------------------===//

TypedValue<DotDimensionNumbersType> getDotDimensionNumbers(
    XlaGpuApi &api, ImplicitLocOpBuilder &b, ModuleOp module,
    lmhlo_gpu::GEMMOp op) {
  mhlo::DotDimensionNumbersAttr attr = op.getDotDimensionNumbersAttr();
  SmallVector<Value> args = {getI64List(b, attr.getLhsBatchingDimensions()),
                             getI64List(b, attr.getRhsBatchingDimensions()),
                             getI64List(b, attr.getLhsContractingDimensions()),
                             getI64List(b, attr.getRhsContractingDimensions())};

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

  SmallVector<Value> args = {getI64List(b, precision)};

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
                std::shared_ptr<DeBufferization> state,
                std::shared_ptr<XlaGpuApi> api)
      : OpConversionPattern(converter, ctx),
        state(std::move(state)),
        api(std::move(api)) {}

  LogicalResult matchAndRewrite(
      lmhlo_gpu::GEMMOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto *block = op->getBlock();
    auto module = op->getParentOfType<ModuleOp>();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto dot_config = getDotConfig(*api, b, module, op);
    auto trace = getTrace(*api, b, module, op);

    // Export arguments to buffer views.
    auto lhs = state->remapped[block][op.getA()];
    auto rhs = state->remapped[block][op.getB()];
    auto out = state->remapped[block][op.getC()];

    if (!lhs || !rhs || !out) {
      return rewriter.notifyMatchFailure(
          op, "missing memref to tensor mapping for lmhlo_gpu.gemm arguments");
    }

    // Arguments to a gemm dispatch.
    SmallVector<Value> args = {getExecutionContext(op)};
    for (TypedValue<TensorType> src : {lhs, rhs, out}) {
      auto export_op = b.create<IREE::Input::TensorExportOp>(
          b.getType<IREE::Input::BufferViewType>(), src,
          /*source_dims=*/ValueRange());
      args.push_back(export_op.getResult());
    }
    args.push_back(dot_config);
    args.push_back(trace);

    // TODO(ezhulenev): Should we import buffer view back and update remapping?
    auto api_func = api->getDispatchGemm(b, module);
    b.create<func::CallOp>(api_func.getSymName(), api_func.getResultTypes(),
                           args);

    rewriter.eraseOp(op);
    return success();
  }

  std::shared_ptr<DeBufferization> state;
  std::shared_ptr<XlaGpuApi> api;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateLibraryOpsConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    std::shared_ptr<DeBufferization> state) {
  auto api = std::make_shared<XlaGpuApi>();
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertGemmOp>(converter, ctx, state, api);
}

}  // namespace xla::gpu
