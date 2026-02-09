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

#include <string>

#include "bin/RegisterTritonDialects.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace {

struct TritonPipelineOptions
    : public mlir::PassPipelineOptions<TritonPipelineOptions> {
  Option<std::string> target{*this, "target", llvm::cl::init("8.0")};
  Option<bool> rewrite_int4{*this, "rewrite-int4", llvm::cl::init(true)};
  Option<bool> allow_tma{*this, "allow-tma", llvm::cl::init(false)};
  Option<int> num_warps{*this, "num-warps", llvm::cl::init(4)};
  Option<int> num_ctas{*this, "num-ctas", llvm::cl::init(1)};
  Option<int> num_stages{*this, "num-stages", llvm::cl::init(3)};
};

mlir::PassPipelineRegistration<TritonPipelineOptions>
    register_triton_xla_pipeline(
        "triton-xla-pipeline",
        "Runs all Triton passes, including the ones from XLA.",
        [](mlir::OpPassManager& pm, const TritonPipelineOptions& options) {
          stream_executor::GpuComputeCapability gpu_cc;

          if (auto cuda_cc =
                  stream_executor::CudaComputeCapability().FromString(
                      options.target);
              cuda_cc.ok()) {
            gpu_cc = *cuda_cc;
          }
          if (stream_executor::RocmComputeCapability rocm_cc(options.target);
              rocm_cc.is_supported_gfx_version()) {
            gpu_cc = rocm_cc;
          }
          bool warp_specialization_allowed = true;
          xla::gpu::CreateTritonXlaPipeline(
              &pm, gpu_cc, options.rewrite_int4, options.allow_tma,
              options.num_stages, warp_specialization_allowed);

          xla::gpu::CreateTritonPipeline(&pm, gpu_cc, options.num_warps,
                                         options.num_ctas, options.num_stages);
        });

}  // namespace

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  registerTritonDialects(registry);  // This registers all passes as well.
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::triton::xla::XlaTritonDialect, xla::XlaDialect,
                  xla::xtile::XTileDialect, mlir::stablehlo::StablehloDialect,
                  mlir::memref::MemRefDialect>();
  mlir::triton::xla::registerTritonXlaTransformsPasses();
  xla::emitters::registerTransformsPasses();
  registry.addExtension(+[](mlir::MLIRContext* ctx, xla::XlaDialect* dialect) {
    xla::RegisterSymbolicExprStorage(ctx);
  });
  xla::gpu::registerGpuFusionTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "xla-opt modular optimizer driver\n", registry));
}
