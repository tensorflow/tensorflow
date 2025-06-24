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
#include <string>
#include <utility>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {
namespace {

namespace se = ::stream_executor;

#define GEN_PASS_DEF_LOWERTOLLVMPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

class LowerToLLVMPass : public impl::LowerToLLVMPassBase<LowerToLLVMPass> {
 public:
  explicit LowerToLLVMPass(const LowerToLLVMPassOptions& options)
      : LowerToLLVMPassBase(options) {}

  explicit LowerToLLVMPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (target_type_ == "gpu" && !gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      *device_spec_.mutable_type() = se::DeviceDescription(device_info);
    } else if (target_type_ == "cpu") {
      CHECK(gpu_device_info_.empty());
      *device_spec_.mutable_type() = CpuDeviceSpec{};
    }
    // Populate type conversions.
    mlir::LowerToLLVMOptions llvm_opts(&getContext(),
                                       mlir::DataLayout(getOperation()));
    mlir::LLVMTypeConverter type_converter(getOperation().getContext(),
                                           llvm_opts);
    mlir::LLVMConversionTarget target(*getOperation().getContext());

    // Populate patterns.
    mlir::RewritePatternSet patterns(&getContext());
    mlir::arith::populateArithExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
    if (device_spec_.IsGpu()) {
      if (device_spec_.IsAmdGpu()) {
        std::string chipset =
            device_spec_.gpu().rocm_compute_capability().gfx_version();
        llvm::FailureOr<mlir::amdgpu::Chipset> maybeChipset =
            mlir::amdgpu::Chipset::parse(chipset);
        if (failed(maybeChipset)) {
          mlir::emitError(mlir::UnknownLoc::get(&getContext()),
                          "Invalid chipset name: " + chipset);
          return signalPassFailure();
        }
        mlir::populateGpuToROCDLConversionPatterns(
            type_converter, patterns, mlir::gpu::amd::Runtime::Unknown,
            *maybeChipset);
        mlir::configureGpuToROCDLConversionLegality(target);
      } else {
        mlir::populateGpuToNVVMConversionPatterns(type_converter, patterns);
        mlir::configureGpuToNVVMConversionLegality(target);
      }
    }
    mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);
    mlir::populateVectorToLLVMConversionPatterns(type_converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                          patterns);
    mlir::populateComplexToLLVMConversionPatterns(type_converter, patterns);

    //  Setup target.
    target.addIllegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                             mlir::complex::ComplexDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Cleanup any leftover math ops not handled NVVM or ROCDL lowering
    mlir::RewritePatternSet mathPatterns(&getContext());
    mlir::populateMathToLLVMConversionPatterns(type_converter, mathPatterns,
                                               /* approximateLog1p */ false);
    target.addIllegalDialect<mlir::math::MathDialect>();

    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(mathPatterns)))) {
      signalPassFailure();
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerToLLVMPass(
    const std::string& target_type, const std::string& gpu_device_info) {
  LowerToLLVMPassOptions options;
  options.gpu_device_info_ = gpu_device_info;
  options.target_type_ = target_type;
  return std::make_unique<LowerToLLVMPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerToLLVMPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerToLLVMPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
