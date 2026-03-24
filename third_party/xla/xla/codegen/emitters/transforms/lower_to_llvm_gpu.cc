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
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // IWYU pragma: keep
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
namespace {

namespace se = ::stream_executor;

#define GEN_PASS_DEF_LOWERTOLLVMGPUPASS
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h.inc"

class LowerToLLVMGPUPass
    : public impl::LowerToLLVMGPUPassBase<LowerToLLVMGPUPass> {
 public:
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
        populateGpuMemorySpaceAttributeConversions(converter);
      } else {
        mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
        mlir::configureGpuToNVVMConversionLegality(target);
      }
      return mlir::success();
    };

    // NVVM and ROCDL lower math.log1p directly via their GPU pattern sets, but
    // the SPIR-V pipeline does not. For Intel GPUs we therefore fall back to
    // the generic MathToLLVM conversion, hence enabling approximate log1p.
    bool lower_math_log1p = device_spec_.IsIntelGpu();
    if (mlir::failed(
            LowerToLLVM(getOperation(), populate_patterns, lower_math_log1p))) {
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

std::unique_ptr<::mlir::Pass> CreateLowerToLLVMGPUPass(
    const std::string& gpu_device_info) {
  LowerToLLVMGPUPassOptions options;
  options.gpu_device_info_ = gpu_device_info;
  return std::make_unique<LowerToLLVMGPUPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerToLLVMGPUPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerToLLVMGPUPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
