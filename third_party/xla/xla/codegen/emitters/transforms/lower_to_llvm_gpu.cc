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

// log2(e), used to express exp(x) = exp2(x * log2(e)).
constexpr double kLog2E = 1.4426950408889634;

// ln(2), used to express log(x) = log2(x) * ln(2).
constexpr double kLn2 = 0.6931471805599453;

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

// Lowers a scalar bf16 `math.exp` to the native gfx1250 `v_exp_bf16`
// instruction by rewriting exp(x) = exp2(x * log2(e)) and emitting the
// `llvm.amdgcn.exp2` intrinsic. See TranscendentalBF16ToAMDGPU for the
// rationale.
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
    mlir::Value log2e = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, bf16, rewriter.getFloatAttr(bf16, kLog2E));
    mlir::Value scaled =
        rewriter.create<mlir::LLVM::FMulOp>(loc, operand, log2e);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, /*resultType=*/bf16, rewriter.getStringAttr("llvm.amdgcn.exp2"),
        mlir::ValueRange{scaled});
    return mlir::success();
  }
};

// Lowers a scalar bf16 `math.log` to the native gfx1250 `v_log_bf16`
// instruction by rewriting log(x) = log2(x) * ln(2) and emitting the
// `llvm.amdgcn.log` intrinsic. See TranscendentalBF16ToAMDGPU for the
// rationale.
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
    mlir::Value log2x = rewriter
                            .create<mlir::LLVM::CallIntrinsicOp>(
                                loc, /*resultType=*/bf16,
                                rewriter.getStringAttr("llvm.amdgcn.log"),
                                mlir::ValueRange{operand})
                            .getResults();
    mlir::Value ln2 = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, bf16, rewriter.getFloatAttr(bf16, kLn2));
    rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, log2x, ln2);
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
        // On gfx1250 emit native bf16 transcendentals (v_exp_bf16, v_sqrt_bf16,
        // v_rsq_bf16, v_tanh_bf16, v_log_bf16, ...) instead of upcasting to f32
        // and calling __ocml_*_f32. Higher benefit than the default MathToROCDL
        // patterns so it wins for scalar bf16 ops.
        if (device_spec_.gpu()
                .rocm_compute_capability()
                .has_bf16_transcendental_support()) {
          mlir::PatternBenefit benefit(2);
          patterns.add<ExpBF16ToAMDGPU>(converter, benefit);
          patterns.add<LogBF16ToAMDGPU>(converter, benefit);
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
