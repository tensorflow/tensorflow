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

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_attrs.h.inc"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/pass_pipelines.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/trace_pass_instrumentation.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

class ModuleCallbackPass
    : public mlir::PassWrapper<ModuleCallbackPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit ModuleCallbackPass(absl::FunctionRef<void(mlir::ModuleOp)> callback)
      : callback_(callback) {}

  void runOnOperation() override { callback_(getOperation()); }

 private:
  absl::FunctionRef<void(mlir::ModuleOp)> callback_;
};

static absl::Status RunPassPipeline(
    mlir::ModuleOp module, mlir::PassManager& pm,
    mlir::interpreter::MlirCompilationTrace* trace,
    int32_t verification_level) {
  if (VLOG_IS_ON(5)) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }

#if NDEBUG
  pm.enableVerifier(verification_level > 0);
  module.getContext()->printOpOnDiagnostic(verification_level > 0);
#endif

  tsl::StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  return diagnostic_handler.consumeStatus(pm.run(module));
}

static std::unique_ptr<::mlir::Pass> CreateConvertMathToLLVMPass() {
  mlir::ConvertMathToLLVMPassOptions options;
  options.approximateLog1p = false;
  return mlir::createConvertMathToLLVMPass(options);
}

// The final lowering passes common to both scalar and tiled kernels.
// These passes are primarily responsible for lowering individual ops to
// their LLVM equivalent.
static void AddGenericLoweringPasses(mlir::OpPassManager& pm) {
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());

  pm.addNestedPass<mlir::func::FuncOp>(cpu::CreateExpandFloatOpsPass());
  pm.addPass(emitters::CreateExpandFloatOpsPass(/*aproximate_tanh=*/false));
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(emitters::CreateLowerXlaIntrinsicLibPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateConvertMathToLLVMPass());
  pm.addPass(emitters::CreateLowerToLLVMPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

static std::unique_ptr<::mlir::Pass> CreateInlinerAndCsePass() {
  return mlir::createCompositeFixedPointPass(
      "Inliner", [](mlir::OpPassManager& pm) {
        pm.addPass(mlir::createInlinerPass({}, [](mlir::OpPassManager& pm) {
          // CSE after inlining because inlining can introduce duplicates.
          pm.addPass(mlir::createCSEPass());
        }));
      });
}

// Optimizations passes for the "hero" emitters, e.g. loop emitter.
// It is expected that the input has a simple nested loop structure that works
// on scalar instructions extracted/inserted from tensor types.
static void AddScalarOptimizationPasses(mlir::OpPassManager& pm,
                                        int32_t vector_width) {
  emitters::RegisterOptimizationPasses(pm);
  pm.addPass(CreateAddReductionFastMathFlagsPass());
  pm.addPass(CreateInlinerAndCsePass());
  pm.addNestedPass<mlir::func::FuncOp>(CreatePeelWorkgroupLoopPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateLowerXlaSharedPass());
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateLowerXlaToScfPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateLowerXlaLoopsToScfPass());
  pm.addPass(mlir::stablehlo::createStablehloConvertToSignlessPass());
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
  // TODO(willfroom): Re-enable vectorization once b/431961172 is fixed.
  // pm.addNestedPass<mlir::func::FuncOp>(
  //     emitters::CreateVectorizeLoadsAndStoresPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateAddLoopUnrollFlagsPass());
}

// Lowering passes for the "hero" emitters, e.g. loop emitter.
// It is expected that the input has a simple nested loop structure that works
// on scalar instructions extracted/inserted from tensor types.
// The resulting IR can then be translated to native LLVM.
static void AddScalarLoweringPasses(mlir::OpPassManager& pm,
                                    int32_t vector_width, bool fast_min_max) {
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
  pm.addNestedPass<mlir::func::FuncOp>(
      emitters::CreateSimplifyArithPass(fast_min_max));
  AddGenericLoweringPasses(pm);
}

// Optimizations passes for the tiled emitter.
// This is currently very simple but will grow to include tiled optimizations
// such as transpose hoisting and dimension reduction.
static void AddTiledOptimizationPasses(mlir::OpPassManager& pm) {
  emitters::RegisterOptimizationPasses(pm);
}

// Lowering passes for the tiled emitter.
// The input IR is from the xtile dialect which uses tensors that are converted
// first to the vector dialect and then to LLVM.
static void AddTiledLoweringPasses(mlir::OpPassManager& pm) {
  pm.addPass(CreateShloToVectorPass());
  pm.addPass(CreateXTileToVectorPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(CreateRewriteDynamicVectorExtractPass());
  pm.addPass(CreateElementalTensorToVectorPass());
  pm.addPass(CreateLowerXTileEntryPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::vector::createLowerVectorMultiReductionPass(
          mlir::vector::VectorMultiReductionLowering::InnerParallel));
  pm.addPass(CreateTensorOpsToVectorPass());
  pm.addPass(cpu::createLowerToLLVMPass());
  pm.addPass(mlir::createConvertVectorToSCFPass(
      mlir::VectorTransferToSCFOptions().enableFullUnroll(false)));
  mlir::ConvertVectorToLLVMPassOptions options;
  options.vectorTransposeLowering =
      mlir::vector::VectorTransposeLowering::Shuffle1D;
  pm.addPass(mlir::createConvertVectorToLLVMPass(options));

  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  AddGenericLoweringPasses(pm);
}

static int GetLlvmFunctionDefCount(mlir::ModuleOp m) {
  int count = 0;
  m.walk([&count](mlir::LLVM::LLVMFuncOp func) {
    if (!func.getBody().empty() &&
        func.getLinkage() != mlir::LLVM::Linkage::Internal) {
      count++;
    }
    return mlir::WalkResult::advance();
  });
  return count;
};

static void ApplyFastMathFlags(llvm::Module& llvm_module,
                               const llvm::FastMathFlags& fast_math_flags) {
  for (llvm::Function& function : llvm_module) {
    for (llvm::BasicBlock& basic_block : function) {
      for (llvm::Instruction& instruction : basic_block) {
        if (llvm::isa<llvm::FPMathOperator>(instruction)) {
          instruction.setFastMathFlags(fast_math_flags);
        }
      }
    }
  }
}

FusionCompiler::FusionCompiler(mlir::MLIRContext* context, Options options,
                               CompilationHooks hooks)
    : options_(std::move(options)),
      hooks_(std::move(hooks)),
      scalar_pass_manager_(mlir::PassManager::on<mlir::ModuleOp>(context)),
      tiled_pass_manager_(mlir::PassManager::on<mlir::ModuleOp>(context)) {
  // Scalar passes.
  AddScalarOptimizationPasses(scalar_pass_manager_, options_.vector_width);
  if (hooks_.post_optimization) {
    scalar_pass_manager_.addPass(
        std::make_unique<ModuleCallbackPass>(hooks_.post_optimization));
  }
  AddScalarLoweringPasses(scalar_pass_manager_, options_.vector_width,
                          options_.fast_min_max);

  // Tiled passes.
  AddTiledOptimizationPasses(tiled_pass_manager_);
  if (hooks_.post_optimization) {
    tiled_pass_manager_.addPass(
        std::make_unique<ModuleCallbackPass>(hooks_.post_optimization));
  }
  AddTiledLoweringPasses(tiled_pass_manager_);

  scalar_pass_manager_.addInstrumentation(
      std::make_unique<TraceInstrumentation>());
  tiled_pass_manager_.addInstrumentation(
      std::make_unique<TraceInstrumentation>());
}

absl::StatusOr<std::unique_ptr<llvm::Module>> FusionCompiler::Compile(
    llvm::LLVMContext& llvm_context, mlir::ModuleOp mlir_module) {
  absl::string_view module_name =
      mlir_module.getName() ? *mlir_module.getName() : "UnknownFusionModule";
  auto get_module_op_count = [&mlir_module]() {
    // Count the number of leaf ops, i.e those without a sub-region.
    int64_t count = 0;
    mlir_module.walk([&count](mlir::Operation* op) {
      if (op->getNumRegions() == 0) {
        count++;
      }
    });
    return count;
  };

  bool is_tiled = !mlir_module.getBody()->getOps<xtile::EntryFuncOp>().empty();
  mlir::PassManager& pm = is_tiled ? tiled_pass_manager_ : scalar_pass_manager_;

  VLOG(1) << "Compiling MLIR module: " << module_name << ", with "
          << get_module_op_count() << " operations.";
  XLA_SCOPED_LOGGING_TIMER_LEVEL(
      absl::StrCat("Compiled MLIR module: ", module_name), 1);

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "FusionCompiler::Compile",
        {{"module", module_name}, {"op_count", get_module_op_count()}});
  });

  if (hooks_.pre_optimization) {
    hooks_.pre_optimization(mlir_module);
  }
  TF_RETURN_IF_ERROR(
      RunPassPipeline(mlir_module, pm, nullptr, options_.verification_level));

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

  if (mlir::Attribute options =
          mlir_module->getAttr(xla::CpuMemoryRegionNameAttr::name)) {
    SetModuleMemoryRegionName(*llvm_module,
                              mlir::cast<mlir::StringAttr>(options).str());
  }

  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";

  llvm_module->setDataLayout(llvm_module->getDataLayout());

  if (options_.fast_math_flags.any()) {
    ApplyFastMathFlags(*llvm_module, options_.fast_math_flags);
  }

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
  // MLIR uses std::thread, which means we will easily oversubscribe, disable it
  // for now.
  // TODO(willfroom): Look into implementing llvm::ThreadPoolInterface using an
  // underlying tsl::thread::ThreadPool (b/437348148).
  auto context = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);

  context->loadDialect<mlir::DLTIDialect, mlir::affine::AffineDialect,
                       mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                       mlir::func::FuncDialect, mlir::math::MathDialect,
                       xla::cpu::XlaCpuDialect, mlir::mhlo::MhloDialect,
                       mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                       mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                       xla::XlaDialect, xla::xtile::XTileDialect>();

  mlir::DialectRegistry registry;
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  context->appendDialectRegistry(registry);

  return context;
}

}  // namespace xla::cpu
