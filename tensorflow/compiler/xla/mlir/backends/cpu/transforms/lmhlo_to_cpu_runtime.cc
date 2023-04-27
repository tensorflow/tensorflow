/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/type_converter.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_CONVERTLMHLOTOCPURUNTIMEPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;

using xla_cpu::PartitionIdOp;
using xla_cpu::ReplicaIdOp;

using xla::runtime::AppendCustomCallAttrs;
using xla::runtime::CustomCallDeclarations;

class ConvertLmhloToCpuRuntimePass
    : public impl::ConvertLmhloToCpuRuntimePassBase<
          ConvertLmhloToCpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }
};

// Copies memrefs with non-identity layouts (e.g. results of memref.subviews)
// to newly allocated memrefs, ensuring all outputs have flat layouts.
// TODO(jreiffers): If the memref just as an offset, but its layout is otherwise
// default, the copy is overkill.
SmallVector<Value> EnsureFlatMemrefs(ValueRange values,
                                     ImplicitLocOpBuilder& b) {
  SmallVector<Value> out;
  for (Value value : values) {
    auto ty = value.getType().dyn_cast<MemRefType>();
    if (!ty || ty.getLayout().isIdentity()) {
      out.push_back(value);
    } else {
      auto default_layout_ty =
          MemRefType::get(ty.getShape(), ty.getElementType());
      auto alloc =
          out.emplace_back(b.create<memref::AllocOp>(default_layout_ty));
      b.create<memref::CopyOp>(value, alloc);
    }
  }
  return out;
}

// Replaces a DPS style collective op with a custom call.
func::CallOp CreateCallForDpsCollectiveOp(Operation* op,
                                          CustomCallDeclarations& custom_calls,
                                          StringRef call_target,
                                          PatternRewriter& rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  b.setInsertionPoint(op);

  // Subview ops result in strided Memrefs. The runtime can't deal with them,
  // so we copy everything that doesn't have the default layout.
  SmallVector<Value> new_operands = EnsureFlatMemrefs(op->getOperands(), b);

  func::FuncOp callee = custom_calls.GetOrCreate(
      b, call_target, TypeRange(ValueRange(new_operands)), TypeRange());
  auto call =
      b.create<func::CallOp>(callee.getName(), TypeRange(), new_operands);

  // Copy attributes from original op.
  for (auto& attr : op->getAttrs()) {
    call->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.eraseOp(op);
  return call;
}

//===----------------------------------------------------------------------===//

class CustomCallOpLowering : public OpRewritePattern<CustomCallOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.cpu.custom_call";

 public:
  CustomCallOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  // Rewrite custom call with `API_VERSION_TYPED_FFI` version into XLA runtime
  // custom calls bypassing custom call adaptor.
  LogicalResult rewriteTypedCustomCall(CustomCallOp op,
                                       PatternRewriter& rewriter) const {
    // TODO(ezhulenev): Support target arg mapping, or explain why we do not
    // need them for typed custom calls.
    if (op.getTargetArgMapping())
      return op.emitOpError(
          "API_VERSION_TYPED_FFI custom calls do not "
          "support target arg mapping");

    // Create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee =
        custom_calls_.GetOrCreate(b, op.getCallTargetName(), op);
    callee->setAttr("rt.dynamic", UnitAttr::get(b.getContext()));

    // Forward backend config to the custom call implementation.
    auto dict = op.getBackendConfig()
                    ? op.getBackendConfig()->cast<mlir::DictionaryAttr>()
                    : nullptr;
    llvm::SmallVector<NamedAttribute> backend_config(dict.begin(), dict.end());

    // Call the custom call function forwarding user-defined attributes.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, backend_config);

    return success();
  }

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    // Typed custom calls lowered directly to XLA runtime custom calls.
    if (op.getApiVersion() == mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI)
      return rewriteTypedCustomCall(op, rewriter);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // Get the number of outputs from operand_segment_sizes.
    int64_t num_results = op->getAttrOfType<DenseI32ArrayAttr>(
        op.getOperandSegmentSizesAttrName())[1];

    // If custom call has target arguments mapping, then we need to pass empty
    // memrefs in place of holes.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      num_results = mapping.getNumResults();

      // Always create an `alloca` in the parent function entry block.
      // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
      Value hole = [&]() -> Value {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(
            &op->getParentOfType<func::FuncOp>().front());
        return b.create<memref::AllocaOp>(MemRefType::get({0}, b.getI8Type()));
      }();

      // We represent holes as empty i8 memrefs.
      operands = llvm::SmallVector<Value>(num_args + num_results, hole);

      // Update operands to mapped custom call arguments.
      auto args = mapping.getArgsToTargetArgs();
      for (const auto& indexed : llvm::enumerate(args))
        operands[indexed.value()] = op.getArgs()[indexed.index()];

      // Update operands to mapped custom call results.
      auto res = mapping.getResultsToTargetResults();
      for (const auto& indexed : llvm::enumerate(res))
        operands[num_args + indexed.value()] = op.getOutput()[indexed.index()];
    }

    // TODO(jreiffers): This will break if an output has a non-default layout.
    operands = EnsureFlatMemrefs(operands, b);
    // Create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCustomCallTarget, TypeRange(ValueRange(operands)), TypeRange());

    // The ABI is different depending on whether the original op was outputting
    // a tuple or not. For multiple outputs this is trivial but for a single
    // output we rely on the xla_shape attribute to distinguish the ABIs.
    bool output_tuple = num_results > 1;
    if (auto xla_shape = op->getAttrOfType<StringAttr>("xla_shape"))
      output_tuple = ParseShape(xla_shape.strref())->IsTuple();

    // This is not equivalent to op.getApiVersionAttr() - that call returns null
    // if the attribute is absent. getApiVersion returns the default.
    Attribute api_version =
        mhlo::CustomCallApiVersionAttr::get(getContext(), op.getApiVersion());
    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("num_results"),
         b.getI32IntegerAttr(static_cast<int32_t>(num_results))},
        {b.getStringAttr("output_tuple"), b.getBoolAttr(output_tuple)},
        {b.getStringAttr("api_version"), api_version},
        {b.getStringAttr("call_target_name"), op.getCallTargetNameAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class InfeedOpLowering : public OpRewritePattern<InfeedOp> {
 private:
  static constexpr const char kCallTarget[] = "xla.cpu.infeed";

 public:
  InfeedOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(InfeedOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // By default all operands are passed to the custom call handler.
    llvm::SmallVector<Value> operands = op->getOperands();

    // Create a custom call function declaration.
    func::FuncOp callee =
        custom_calls_.GetOrCreate(b, StringRef(kCallTarget),
                                  TypeRange(ValueRange(operands)), TypeRange());

    // Call the runtime intrinsic with the original operands.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(), TypeRange(),
                                              operands);
    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

template <typename IdOp>
class IdOpLowering : public OpRewritePattern<IdOp> {
 public:
  IdOpLowering(MLIRContext* ctx, llvm::StringRef call_target,
               CustomCallDeclarations& custom_calls)
      : OpRewritePattern<IdOp>(ctx),
        call_target_(call_target),
        custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // Create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, call_target_, TypeRange(), TypeRange(rewriter.getI32Type()));

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(),
                                              TypeRange(rewriter.getI32Type()));
    return success();
  }

 private:
  llvm::StringRef call_target_;
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class AllReduceLowering : public OpRewritePattern<xla_cpu::AllReduceOp> {
 public:
  AllReduceLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getOperandTypes().front().isa<MemRefType>()) {
      return failure();
    }

    auto call = CreateCallForDpsCollectiveOp(op.getOperation(), custom_calls_,
                                             kCallTarget, rewriter);

    // Set default attributes.
    if (!call->hasAttr("use_global_device_ids")) {
      call->setAttr("use_global_device_ids", rewriter.getI32IntegerAttr(0));
    }
    if (!call->hasAttr("op_id")) {
      call->setAttr("op_id", rewriter.getI64IntegerAttr(0));
    }

    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.all_reduce";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class AllToAllLowering : public OpRewritePattern<xla_cpu::AllToAllOp> {
 public:
  AllToAllLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::AllToAllOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getSplitDimensionAttr()) {
      op.emitOpError("ArrayAllToAll is not supported");
      return failure();
    }
    CreateCallForDpsCollectiveOp(op.getOperation(), custom_calls_, kCallTarget,
                                 rewriter);
    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.tuple_all_to_all";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class CollectivePermuteLowering
    : public OpRewritePattern<xla_cpu::CollectivePermuteOp> {
 public:
  CollectivePermuteLowering(MLIRContext* ctx,
                            CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::CollectivePermuteOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getOperandTypes().front().isa<MemRefType>()) {
      return failure();
    }

    CreateCallForDpsCollectiveOp(op.getOperation(), custom_calls_, kCallTarget,
                                 rewriter);
    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.collective_permute";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class ConvolutionLowering : public OpRewritePattern<xla_cpu::ConvolutionOp> {
 public:
  ConvolutionLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    b.setInsertionPoint(op);

    // Subview ops result in strided Memrefs. The runtime can't deal with them,
    // so we copy everything that doesn't have the default layout.
    SmallVector<Value> new_operands = EnsureFlatMemrefs(op->getOperands(), b);

    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCallTarget, TypeRange(ValueRange(new_operands)), TypeRange());
    auto call =
        b.create<func::CallOp>(callee.getName(), TypeRange(), new_operands);

    // Copy attributes from original op.
    for (auto name :
         {"inputBatchDimension", "inputSpatialDimensions",
          "inputFeatureDimension", "kernelSpatialDimensions",
          "kernelInputFeatureDimension", "kernelOutputFeatureDimension",
          "outputSpatialDimensions", "window_strides", "padding",
          "lhs_dilation", "rhs_dilation", "feature_group_count"}) {
      call->setAttr(name, op->getAttr(name));
    }
    rewriter.eraseOp(op);
    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.convolution";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class RngBitGeneratorLowering
    : public OpRewritePattern<xla_cpu::RngBitGeneratorOp> {
 public:
  RngBitGeneratorLowering(MLIRContext* ctx,
                          CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::RngBitGeneratorOp op,
                                PatternRewriter& rewriter) const override {
    auto algorithm =
        op.getRngAlgorithmAttr().cast<mhlo::RngAlgorithmAttr>().getValue();
    op->removeAttr("rng_algorithm");

    CreateCallForDpsCollectiveOp(op.getOperation(), custom_calls_,
                                 algorithm == mhlo::RngAlgorithm::THREE_FRY
                                     ? kThreeFryTarget
                                     : kPhiloxTarget,
                                 rewriter);
    return success();
  }

 private:
  static constexpr const char kThreeFryTarget[] = "xla.cpu.rng.three_fry";
  static constexpr const char kPhiloxTarget[] = "xla.cpu.rng.philox";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class OutfeedLowering : public OpRewritePattern<xla_cpu::OutfeedOp> {
 public:
  OutfeedLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::OutfeedOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // By default all operands are passed to the custom call handler.
    llvm::SmallVector<Value> operands = EnsureFlatMemrefs(op->getOperands(), b);

    // Create a custom call function declaration.
    func::FuncOp callee =
        custom_calls_.GetOrCreate(b, StringRef(kCallTarget),
                                  TypeRange(ValueRange(operands)), TypeRange());

    llvm::SmallVector<NamedAttribute> custom_call_attrs;
    SmallVector<int32_t> types;
    for (int i = 0; i < op.getResultType().size(); ++i) {
      auto type_attr = cast<TypeAttr>(op.getResultType()[i]);
      auto status_or_primitive_type =
          xla::runtime::TypeConverter::ConvertElementType(type_attr.getValue());
      if (!status_or_primitive_type.ok()) {
        return rewriter.notifyMatchFailure(
            op,
            "is not provided with a supported primitive type in the result "
            "type attribute.");
      }
      types.push_back(status_or_primitive_type.value());
    }

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    call->setAttr("result_type", b.getI32ArrayAttr(types));

    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.outfeed";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class FftLowering : public OpRewritePattern<xla_cpu::FftOp> {
 public:
  FftLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::FftOp op,
                                PatternRewriter& rewriter) const override {
    CreateCallForDpsCollectiveOp(op.getOperation(), custom_calls_, kCallTarget,
                                 rewriter);
    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.fft";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

void ConvertLmhloToCpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA cpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns
      .insert<AllReduceLowering, AllToAllLowering, CollectivePermuteLowering,
              ConvolutionLowering, CustomCallOpLowering, FftLowering,
              InfeedOpLowering, OutfeedLowering, RngBitGeneratorLowering>(
          ctx, custom_calls);
  patterns.insert<IdOpLowering<PartitionIdOp>>(ctx, "xla.cpu.partition_id",
                                               custom_calls);
  patterns.insert<IdOpLowering<ReplicaIdOp>>(ctx, "xla.cpu.replica_id",
                                             custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToCpuRuntimePass() {
  return std::make_unique<ConvertLmhloToCpuRuntimePass>();
}

}  // namespace cpu
}  // namespace xla
