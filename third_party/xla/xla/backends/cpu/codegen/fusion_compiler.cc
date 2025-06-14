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

#include "xla/backends/cpu/codegen/fusion_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_attrs.h.inc"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

static absl::Status RunPassPipeline(
    mlir::ModuleOp module, mlir::PassManager& pm,
    mlir::interpreter::MlirCompilationTrace* trace) {
  if (VLOG_IS_ON(5)) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }

  tsl::StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.consumeStatus(pm.run(module));
}

static void AddXlaOpsOptimizationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createCSEPass());
}

static void AddLoopTransformationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<mlir::func::FuncOp>(CreateLowerXlaSharedPass());
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateLowerXlaToScfPass());
  pm.addPass(mlir::createInlinerPass({}, [&](mlir::OpPassManager& pm) {
    // CSE after inlining because inlining can introduce duplicates.
    pm.addPass(mlir::createCSEPass());
  }));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateLowerXlaLoopsToScfPass());
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  pm.addPass(emitters::CreatePropagateSliceIndicesPass());
  pm.addPass(emitters::CreateFlattenTensorsPass());
  // We need LICM before unswitching loops, because our loop unswitcher only
  // detects for loops with a single if inside them.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateUnswitchLoopsPass());
  // We need LICM again after unswitching, because that can introduce new
  // opportunities for LICM. This would not be necessary if LICM also moved
  // instructions over ifs.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateVectorizeLoadsAndStoresPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

static void AddLoweringPasses(mlir::OpPassManager& pm, int32_t vector_width) {
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateConvertPureCallOpsPass());
  pm.addPass(cpu::createLowerToLLVMPass(
      cpu::LowerToLLVMPassOptions{/*prefer_vector_width =*/vector_width}));
  pm.addPass(emitters::CreateLowerTensorsPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(emitters::CreateMergePointersToSameSlicePass());

  // LowerTensors creates new affine.apply ops. Fold and CSE them so
  // simplify-affine has maximally folded expressions to work with.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());

  pm.addNestedPass<mlir::func::FuncOp>(cpu::CreateExpandFloatOpsPass());
  pm.addPass(emitters::CreateExpandFloatOpsPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
  pm.addPass(emitters::CreateLowerToLLVMPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

static int GetLlvmFunctionDefCount(mlir::ModuleOp m) {
  int count = 0;
  m.walk([&count](mlir::LLVM::LLVMFuncOp func) {
    if (!func.getBody().empty()) {
      count++;
    }
    return mlir::WalkResult::advance();
  });
  return count;
};

absl::StatusOr<std::unique_ptr<llvm::Module>> FusionCompiler::Compile(
    llvm::LLVMContext& llvm_context, mlir::ModuleOp mlir_module) {
  mlir::PassManager optimization_pass_manager(mlir_module.getContext());

  if (hooks_.pre_optimization) {
    hooks_.pre_optimization(mlir_module);
  }

  AddXlaOpsOptimizationPasses(optimization_pass_manager);
  AddLoopTransformationPasses(optimization_pass_manager);

  TF_RETURN_IF_ERROR(
      RunPassPipeline(mlir_module, optimization_pass_manager, nullptr));

  if (hooks_.post_optimization) {
    hooks_.post_optimization(mlir_module);
  }

  mlir::PassManager lowering_pass_manager(mlir_module.getContext());

  AddLoweringPasses(lowering_pass_manager, options_.vector_width);

  TF_RETURN_IF_ERROR(
      RunPassPipeline(mlir_module, lowering_pass_manager, nullptr));

  if (hooks_.post_lowering) {
    hooks_.post_lowering(mlir_module);
  }

  // At the end of the MLIR pipeline we must have just one function definition.
  // This helps later compilation stages, where each thunk is assumed to be a
  // standalone function.
  if (int func_count = GetLlvmFunctionDefCount(mlir_module); func_count != 1) {
    return Internal("The module must have just one function definition; has %d",
                    func_count);
  }

  constexpr absl::string_view kXlaModuleIdentifier = "__compute_module";
  absl::string_view module_name =
      mlir_module.getName() ? *mlir_module.getName() : "UnknownFusionModule";
  std::unique_ptr<llvm::Module> llvm_module = mlir::translateModuleToLLVMIR(
      mlir_module, llvm_context,
      absl::StrCat(kXlaModuleIdentifier, "_", module_name));

  if (mlir::Attribute options =
          mlir_module->getAttr(xla::ExtraBackendOptionsAttr::name)) {
    const auto formatter = [](std::string* out, const mlir::StringAttr& attr) {
      absl::StrAppend(out, attr.str());
    };
    std::string options_csv = absl::StrJoin(
        mlir::cast<xla::ExtraBackendOptionsAttr>(options), ",", formatter);
    llvm::MDString* options_mdstring =
        llvm::MDString::get(llvm_context, options_csv);
    llvm_module->addModuleFlag(llvm::Module::Error, "xla_backend_extra_options",
                               options_mdstring);
  }

  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";

  llvm_module->setDataLayout(llvm_module->getDataLayout());

  return llvm_module;
}

// Compile a MLIR kernel source to a LLVM kernel source.
absl::StatusOr<LlvmIrKernelSource> FusionCompiler::Compile(
    MlirKernelSource mlir_kernel_source) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> llvm_module,
                      Compile(*llvm_context, mlir_kernel_source.module()));
  return LlvmIrKernelSource(std::move(llvm_context), std::move(llvm_module));
}

std::unique_ptr<mlir::MLIRContext> FusionCompiler::CreateContext() {
  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::DLTIDialect, mlir::affine::AffineDialect,
                       mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                       mlir::func::FuncDialect, mlir::math::MathDialect,
                       xla::cpu::XlaCpuDialect, mlir::mhlo::MhloDialect,
                       mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                       mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                       xla::XlaDialect>();

  mlir::DialectRegistry registry;
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  context->appendDialectRegistry(registry);

  return context;
}

}  // namespace xla::cpu
