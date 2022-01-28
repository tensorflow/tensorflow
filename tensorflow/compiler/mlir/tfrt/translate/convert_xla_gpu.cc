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
#include "tensorflow/compiler/mlir/tfrt/translate/convert_xla_gpu.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pass_utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/core/platform/errors.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

namespace tensorflow {

StatusOr<tfrt::gpu::Program> ConvertXlaGpuToGpuProgram(
    std::unique_ptr<xla::HloModule> hlo_module, tfrt::HostContext* host,
    llvm::StringRef platform_name) {
  // TODO(fishx): Register dialect in each lowering passes instead.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::lmhlo_gpu::LmhloGpuDialect,
                  tfrt::gpu::GpuDialect,
                  tfrt::gpu::conversion::GpuConversionDialect>();
  tfrt::RegisterTFRTDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  std::string entry_name = hlo_module->entry_computation()->name();

  // XLA HLO -> LMHLO
  TF_RETURN_IF_ERROR(mlir::OptimizeAndConvertHloToLmhlo(
      std::move(hlo_module), *module, platform_name,
      /*optimize_xla_hlo=*/true));

  // LMHLO -> TFRT Dialect (gpu kernels)
  TF_RETURN_IF_ERROR(
      tensorflow::ConvertLmhloToTfrtGpuWithBinary(*module, entry_name, {}));

  // TFRT Dialect -> BEF
  std::string bef;
  llvm::raw_string_ostream bef_ostream(bef);
  if (tfrt::MLIRToBEFTranslate(*module, bef_ostream).failed()) {
    return errors::Internal("Failed to lower TFRT Dialect to BEF.");
  }

  return tfrt::gpu::Program(
      tfrt::BefBuffer(bef.data(), bef.data() + bef.size()), entry_name, host);
}

}  // namespace tensorflow
