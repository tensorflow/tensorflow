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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"
#include "xla/codegen/emitters/transforms/pass_pipelines.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::DLTIDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::complex::ComplexDialect, mlir::func::FuncDialect,
      mlir::gpu::GPUDialect, mlir::math::MathDialect, mlir::mhlo::MhloDialect,
      mlir::mhlo::MhloDialect, mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect, mlir::vector::VectorDialect, xla::XlaDialect,
      xla::cpu::XlaCpuDialect, xla::gpu::XlaGpuDialect,
      xla::xtile::XTileDialect, mlir::stablehlo::StablehloDialect,
      mlir::ROCDL::ROCDLDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInliner();
  xla::emitters::registerTransformsPasses();
  xla::emitters::registerTransformsLLVMGPUPasses();
  xla::gpu::registerGpuFusionTransformsPasses();
  xla::cpu::registerXlaCpuTransformsPasses();
  xla::cpu::registerXTileCpuTransformsPasses();
  xla::xtile::registerXTileTransformsPasses();
  mlir::bufferization::registerBufferizationPasses();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::registerPassPipeline(
      "xla-test-optimize",
      "Test pipeline of passes up to inlining. No vectorization, also does not "
      "lower xla_gpu. Intended to simplify IR in tests.",
      [=](mlir::OpPassManager& pm, llvm::StringRef options,
          llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
              errorHandler) {
        if (!options.empty()) {
          return mlir::failure();
        }
        xla::emitters::RegisterOptimizationPasses(pm);
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});
  mlir::registerPassPipeline(
      "xla-gpu-test-transform-loops",
      "Test pipeline for vectorization. Should run after "
      "xla-gpu-test-to-inline.",
      [=](mlir::OpPassManager& pm, llvm::StringRef options,
          llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
              errorHandler) {
        if (!options.empty()) {
          return mlir::failure();
        }
        xla::gpu::AddLoopTransformationPasses(
            pm, xla::gpu::TestGpuDeviceInfo::RTXA6000DeviceInfo());
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});
  mlir::registerPassPipeline(
      "xla-gpu-test-to-llvm",
      "Test pipeline for the lowering to LLVM. Should run after "
      "xla-gpu-test-to-transform-loops.",
      [=](mlir::OpPassManager& pm, llvm::StringRef options,
          llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
              errorHandler) {
        if (!options.empty()) {
          return mlir::failure();
        }
        xla::gpu::AddLoweringPasses(
            pm, xla::gpu::TestGpuDeviceInfo::RTXA6000DeviceInfo());
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});

  return mlir::failed(
      MlirOptMain(argc, argv, "XLA Emitters Pass Driver\n", registry));
}
