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

#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "google/protobuf/text_format.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_common.h"
#include "xla/codegen/emitters/transforms/lowering_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERTOLLVMGPUPASS
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h.inc"

namespace {

namespace se = ::stream_executor;

// ln(2), used to express log(x) = log2(x) * ln(2).
constexpr double kLn2 = 0.6931471805599453;

// log2(e), used to express exp(x) = exp2(x * log2(e)).
constexpr double kLog2e = 1.4426950408889634;

// Smallest normal f32; v_log_f32 flushes anything below it to zero.
constexpr double kMinNormalF32 = 0x1p-126;

// Scale applied to subnormal log arguments, and its correction in ln(2) units.
constexpr double kSubnormalScale = 0x1p64;
constexpr double kSubnormalScaleLn2 = 64 * kLn2;

// Lowers a scalar bf16 unary `math` op to the matching native gfx1250 bf16
// transcendental instruction (v_exp_bf16, v_sqrt_bf16, v_rsq_bf16, v_tanh_bf16,
// v_log_bf16, ...) via its `llvm.amdgcn.*` intrinsic, when the op maps 1:1 to
// the instruction. Without this, the default MathToROCDL lowering upcasts bf16
// to f32 and calls an `__ocml_*_f32` library function, never using the hardware
// bf16 transcendental unit. Vector ops are scalarized first by MathToROCDL's
// ScalarizeVectorOpLowering (lower benefit), so this pattern only needs to
// handle the scalar case.
template <typename OpTy>
struct TranscendentalBF16ToAMDGPU : public mlir::ConvertOpToLLVMPattern<OpTy> {
  TranscendentalBF16ToAMDGPU(const mlir::LLVMTypeConverter& converter,
                             llvm::StringRef intrinsic,
                             mlir::PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        intrinsic(intrinsic) {}

  mlir::LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 op");
    }
    mlir::Value operand = adaptor.getOperands().front();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/operand.getType(), rewriter.getStringAttr(intrinsic),
        mlir::ValueRange{operand});
    return mlir::success();
  }

  llvm::StringRef intrinsic;
};

// Lowers a scalar bf16 `math.log` by rewriting log(x) = log2(x) * ln(2) and
// computing log2 with the native `v_log_f32` transcendental (the
// `llvm.amdgcn.log` intrinsic) in f32.
//
// Everything is computed in f32 and rounded to bf16 once, which makes every
// bf16 input correctly rounded. Only f32 instructions are needed, so this
// applies to every AMD GPU, and it stays well under the cost of upcasting the
// op to f32 and calling `__ocml_log_f32`.
struct LogBF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::LogOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::LogOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 log");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperands().front();
    mlir::Type bf16 = operand.getType();
    mlir::Type f32 = rewriter.getF32Type();
    mlir::Value x_f32 =
        mlir::LLVM::FPExtOp::create(rewriter, loc, f32, operand);
    // v_log_f32 flushes subnormals to zero and a bf16 subnormal is still
    // subnormal in f32, so scale those up first and correct the result below.
    // The abs costs nothing; it folds into a source modifier on the compare.
    mlir::Value min_normal = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kMinNormalF32));
    mlir::Value abs_x = mlir::LLVM::FAbsOp::create(rewriter, loc, x_f32);
    mlir::Value is_subnormal = mlir::LLVM::FCmpOp::create(
        rewriter, loc, mlir::LLVM::FCmpPredicate::olt, abs_x, min_normal);
    mlir::Value scale = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kSubnormalScale));
    mlir::Value scaled_x =
        mlir::LLVM::FMulOp::create(rewriter, loc, x_f32, scale);
    mlir::Value log_arg = mlir::LLVM::SelectOp::create(
        rewriter, loc, is_subnormal, scaled_x, x_f32);
    mlir::Value log2x = mlir::LLVM::CallIntrinsicOp::create(
                            rewriter, loc, /*resultType=*/f32,
                            rewriter.getStringAttr("llvm.amdgcn.log"),
                            mlir::ValueRange{log_arg})
                            .getResults();
    mlir::Value ln2_f32 = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kLn2));
    mlir::Value logx_f32 =
        mlir::LLVM::FMulOp::create(rewriter, loc, log2x, ln2_f32);
    mlir::Value zero = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, 0.0));
    mlir::Value scale_ln2 = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kSubnormalScaleLn2));
    mlir::Value correction = mlir::LLVM::SelectOp::create(
        rewriter, loc, is_subnormal, scale_ln2, zero);
    // Multiply then subtract so the two contract into one fma.
    mlir::Value result =
        mlir::LLVM::FSubOp::create(rewriter, loc, logx_f32, correction);
    rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(op, bf16, result);
    return mlir::success();
  }
};

// Lowers a scalar bf16 `math.exp` by rewriting exp(x) = 2^(x * log2(e)) and
// computing exp2 with the native `v_exp_f32` transcendental (the
// `llvm.amdgcn.exp2` intrinsic) in f32.
//
// Everything is computed in f32 and rounded to bf16 once. Doing it in f32
// rather than with the native bf16 `v_exp_bf16` (which would round x * log2(e)
// to bf16 before exponentiating) keeps the error from growing with |x| and
// overflowing to inf, which is why the native bf16 exp path was removed. Only
// f32 instructions are needed, so this applies to every AMD GPU, and every
// bf16 input is correctly rounded apart from two near-ties.
struct ExpBF16ToAMDGPU
    : public mlir::ConvertOpToLLVMPattern<mlir::math::ExpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a scalar bf16 exp");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperands().front();
    mlir::Type bf16 = operand.getType();
    mlir::Type f32 = rewriter.getF32Type();
    mlir::Value x_f32 =
        mlir::LLVM::FPExtOp::create(rewriter, loc, f32, operand);
    // v_exp_f32 flushes subnormal results to zero. Halving the exponent keeps
    // exp2 normal, and squaring below reaches the subnormals because an
    // ordinary f32 multiply does round to them.
    mlir::Value half_log2e = mlir::LLVM::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, kLog2e / 2));
    mlir::Value scaled =
        mlir::LLVM::FMulOp::create(rewriter, loc, x_f32, half_log2e);
    mlir::Value root = mlir::LLVM::CallIntrinsicOp::create(
                           rewriter, loc, /*resultType=*/f32,
                           rewriter.getStringAttr("llvm.amdgcn.exp2"),
                           mlir::ValueRange{scaled})
                           .getResults();
    mlir::Value exp2x = mlir::LLVM::FMulOp::create(rewriter, loc, root, root);
    rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(op, bf16, exp2x);
    return mlir::success();
  }
};

class LowerToLLVMGPUPass
    : public impl::LowerToLLVMGPUPassBase<LowerToLLVMGPUPass> {
 public:
  LowerToLLVMGPUPass() = default;

  explicit LowerToLLVMGPUPass(const LowerToLLVMGPUPassOptions& options)
      : LowerToLLVMGPUPassBase(options) {}

  explicit LowerToLLVMGPUPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (!gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      absl::StatusOr<se::DeviceDescription> device_description =
          se::DeviceDescription::FromProto(device_info);
      CHECK_OK(device_description.status());
      *device_spec_.mutable_type() = *device_description;
    }

    auto populate_patterns =
        [&](mlir::LLVMTypeConverter& converter,
            mlir::RewritePatternSet& patterns,
            mlir::ConversionTarget& target) -> mlir::LogicalResult {
      if (device_spec_.IsAmdGpu()) {
        std::string chipset =
            device_spec_.gpu().rocm_compute_capability().gfx_version();
        llvm::FailureOr<mlir::amdgpu::Chipset> maybeChipset =
            mlir::amdgpu::Chipset::parse(chipset);
        if (mlir::failed(maybeChipset)) {
          mlir::emitError(mlir::UnknownLoc::get(&getContext()),
                          "Invalid chipset name: " + chipset);
          return mlir::failure();
        }
        mlir::populateGpuToROCDLConversionPatterns(
            converter, patterns, mlir::gpu::amd::Runtime::Unknown,
            *maybeChipset);
        mlir::configureGpuToROCDLConversionLegality(target);
        mlir::populateAMDGPUToROCDLConversionPatterns(converter, patterns,
                                                      *maybeChipset);
        // Higher benefit than the default MathToROCDL patterns so these win for
        // scalar bf16 ops.
        mlir::PatternBenefit benefit(2);
        // exp and log need only f32 transcendentals, so they apply everywhere;
        // MathToROCDL has no bf16 lowering for them at all.
        patterns.add<LogBF16ToAMDGPU>(converter, benefit);
        patterns.add<ExpBF16ToAMDGPU>(converter, benefit);
        // The rest map 1:1 onto native bf16 transcendentals, which only gfx1250
        // has; elsewhere MathToROCDL upcasts to f32 and calls __ocml_*_f32.
        if (device_spec_.gpu()
                .rocm_compute_capability()
                .has_bf16_transcendental_support()) {
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::Exp2Op>>(
              converter, "llvm.amdgcn.exp2", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::SqrtOp>>(
              converter, "llvm.amdgcn.sqrt", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::RsqrtOp>>(
              converter, "llvm.amdgcn.rsq", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::TanhOp>>(
              converter, "llvm.amdgcn.tanh", benefit);
          patterns.add<TranscendentalBF16ToAMDGPU<mlir::math::Log2Op>>(
              converter, "llvm.amdgcn.log", benefit);
        }
        target.addIllegalDialect<mlir::amdgpu::AMDGPUDialect>();
      } else if (device_spec_.IsIntelGpu()) {
        // Add sub-group-size attribute to functions.
        int32_t sub_group_size = device_spec_.gpu().threads_per_warp();
        if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(getOperation())) {
          module_op.walk([sub_group_size](mlir::func::FuncOp func) {
            if (!func.getBody().empty()) {
              mlir::OpBuilder b(func.getContext());
              auto sub_group_attr = b.getI32IntegerAttr(sub_group_size);
              func->setAttr("intel_reqd_sub_group_size", sub_group_attr);
            }
          });
        }
        populateGpuToLLVMSPVConversionPatterns(converter, patterns);
        spirv::populateMathToLLVMSPVConversionPatterns(spirv::getSPIRVMathOps(),
                                                       converter, patterns);
        populateGpuMemorySpaceAttributeConversions(converter);
      } else {
        mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
        mlir::configureGpuToNVVMConversionLegality(target);
      }
      return mlir::success();
    };

    if (mlir::failed(LowerToLLVM(getOperation(), populate_patterns))) {
      signalPassFailure();
      return;
    }

    if (device_spec_.IsAmdGpu()) {
      EnsureAMDGPUAllocasUseAS5(getOperation());
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> createLowerToLLVMGPUPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerToLLVMGPUPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
