/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

//===- cubin_creator.cc -----------------------------------------*- C++ -*-===//
//
// This file implements the function to compile a TF kernel function to a cubin.
//
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/tools/kernel_gen/cubin_creator.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/NVVMIR.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#endif

namespace {
using tensorflow::Status;
using xla::InternalError;
using xla::StatusOr;

StatusOr<std::string> GetLibdeviceDir(
    const xla::HloModuleConfig& hlo_module_config) {
  for (const std::string& cuda_root : tensorflow::CandidateCudaRoots(
           hlo_module_config.debug_options().xla_gpu_cuda_data_dir())) {
    std::string libdevice_dir =
        tensorflow::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  return InternalError(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice");
}

struct MaterializeBroadcastsPass
    : public mlir::PassWrapper<MaterializeBroadcastsPass, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::ConversionTarget conversionTarget(getContext());
    mlir::OwningRewritePatternList conversionPatterns;

    // Consider the xla_hlo dialect legal for tests.
    conversionTarget.addLegalDialect<mlir::xla_hlo::XlaHloDialect>();
    // The conversion uses helpers from the Standard dialect.
    conversionTarget.addLegalDialect<mlir::StandardOpsDialect>();

    mlir::xla_hlo::SetupMaterializeBroadcastsLegality(&getContext(),
                                                      &conversionTarget);
    mlir::xla_hlo::PopulateMaterializeBroadcastsPatterns(&getContext(),
                                                         &conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

struct UnfuseBatchNormPass
    : public mlir::PassWrapper<UnfuseBatchNormPass, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    mlir::xla_hlo::PopulateUnfuseBatchNormPatterns(&getContext(), &patterns);
    mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

Status LowerTfOpToLhloWithDynamicShapes(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  auto enable_if_vlog_is_on = [](mlir::Pass* pass, mlir::Operation* op) {
    return VLOG_IS_ON(1);
  };
  pm.enableIRPrinting(/*shouldPrintBeforePass=*/{},
                      /*shouldPrintAfterPass=*/enable_if_vlog_is_on,
                      /*printModuleScope=*/false,
                      /*printAfterOnlyOnChange=*/false, llvm::dbgs());
  pm.addNestedPass<mlir::FuncOp>(mlir::xla_hlo::createLegalizeTFPass(false));
  pm.addNestedPass<mlir::FuncOp>(
      absl::make_unique<MaterializeBroadcastsPass>());
  pm.addNestedPass<mlir::FuncOp>(absl::make_unique<UnfuseBatchNormPass>());
  pm.addPass(mlir::xla_hlo::createLegalizeToLhloPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::xla_lhlo::createLhloCopyRemovalPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering TF to LHLO failed.");
  }
  return Status::OK();
}

struct PropagateStaticKnowledge
    : public mlir::PassWrapper<PropagateStaticKnowledge,
                               mlir::OperationPass<mlir::LLVM::LLVMFuncOp>> {
  explicit PropagateStaticKnowledge(mlir::FunctionType type,
                                    llvm::ArrayRef<uint32_t> same_shape_)
      : func_type(type), same_shape(same_shape_) {}

  void runOnOperation() override {
    // We know due to tensorflow ABI that the offset is always 0 and that the
    // innermost stride is always 1. To make this visible to the compiler,
    // we insert constants into the code and replace usages accordingly.
    // We do not change the signature so that we keep a somewhat stable ABI
    // that is easy to undertand by tools.
    mlir::LLVM::LLVMFuncOp func = getOperation();
    mlir::OpBuilder b(func.getBody());
    auto index_type = func.getArgument(3).getType();
    mlir::Value one = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 1));
    mlir::Value zero = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 0));
    uint32_t arg_pos = 0;
    std::vector<uint32_t> positions;
    for (mlir::Type arg_type : func_type.getInputs()) {
      positions.push_back(arg_pos);
      func.getArgument(arg_pos + 2).replaceAllUsesWith(zero);
      arg_pos += 3 + arg_type.cast<mlir::ShapedType>().getRank() * 2;
      func.getArgument(arg_pos - 1).replaceAllUsesWith(one);
    }

    // If we have knowledge that some arguments have the same shape, we
    // can use that here. Simply replace usages of the shape parameters within
    // the function body to a single shape parameter.
    if (!same_shape.empty()) {
      auto first = same_shape.front();
      auto first_offset = positions.at(first);
      mlir::ShapedType first_type =
          func_type.getInput(first).cast<mlir::ShapedType>();
      uint32_t rank = first_type.getRank();
      for (auto same : same_shape.drop_front(1)) {
        uint32_t same_offset = positions.at(same);
        auto same_type = func_type.getInput(same).cast<mlir::ShapedType>();
        if (same_type.getRank() != rank) {
          func.emitOpError() << "same shape constraints on arguments with "
                                "non-matching shapes: #"
                             << first << " and #" << same;
          signalPassFailure();
        }

        for (uint32_t i = 0; i < 2 * rank; ++i) {
          // Replace uses for second arg data with first arg.
          auto same_arg = func.getArgument(same_offset + 3 + i);
          auto first_arg = func.getArgument(first_offset + 3 + i);
          same_arg.replaceAllUsesWith(first_arg);
        }
      }
    }
  }

  mlir::FunctionType func_type;
  llvm::ArrayRef<uint32_t> same_shape;
};

Status PropagateStaticShapeKnowledgeToKernel(
    mlir::ModuleOp module, llvm::ArrayRef<uint32_t> same_shape) {
  // Grab the original signature from the single function.
  auto func = *module.getBody()->op_begin<mlir::FuncOp>();

  mlir::PassManager pm(module.getContext());
  auto enable_if_vlog_is_on = [](mlir::Pass*, mlir::Operation*) {
    return VLOG_IS_ON(1);
  };
  pm.enableIRPrinting(/*shouldPrintBeforePass=*/{},
                      /*shouldPrintAfterPass=*/enable_if_vlog_is_on,
                      /*printModuleScope=*/false,
                      /*printAfterOnlyOnChange=*/false, llvm::dbgs());
  auto& kernel_pm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernel_pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      absl::make_unique<PropagateStaticKnowledge>(func.getType(), same_shape));

  if (failed(pm.run(module))) {
    return InternalError("Static knowledge propagation failed.");
  }
  return Status::OK();
}
}  // namespace

StatusOr<std::vector<uint8_t>> tensorflow::kernel_gen::GenerateCubinForTfCode(
    llvm::StringRef tf_code, std::pair<int32_t, int32_t> compute_capability,
    llvm::ArrayRef<uint32_t> tile_sizes, llvm::ArrayRef<uint32_t> same_shape,
    llvm::ArrayRef<uint32_t> unroll_factors) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();  // TODO(b/152572127)
  mlir::OwningModuleRef module = mlir::parseSourceString(tf_code, &context);

  TF_RETURN_IF_ERROR(LowerTfOpToLhloWithDynamicShapes(module.get()));
  TF_RETURN_IF_ERROR(
      xla::mlir_gpu::LowerLHLOToGPU(module.get(), tile_sizes, unroll_factors,
                                    /*collapseParallelLoops=*/false));
  TF_RETURN_IF_ERROR(xla::mlir_gpu::LowerKernelBodiesToNVVM(module.get()));
  TF_RETURN_IF_ERROR(
      PropagateStaticShapeKnowledgeToKernel(module.get(), same_shape));

  mlir::OwningModuleRef kernel_module =
      xla::mlir_gpu::ExtractKernelModule(*module).ValueOrDie();
  auto llvmModule = mlir::translateModuleToNVVMIR(*kernel_module);
  if (!llvmModule) {
    return InternalError("Could not translate MLIR module to NVVM");
  }

  llvmModule->setModuleIdentifier("acme");
  llvmModule->setDataLayout(xla::gpu::nvptx::kDataLayout);

  xla::HloModuleConfig config;
  config.set_debug_options(xla::GetDebugOptionsFromFlags());

  TF_ASSIGN_OR_RETURN(std::string libdevice_dir, GetLibdeviceDir(config));
  TF_ASSIGN_OR_RETURN(std::string ptx, xla::gpu::nvptx::CompileToPtx(
                                           llvmModule.get(), compute_capability,
                                           config, libdevice_dir));
  VLOG(1) << ptx;

#if GOOGLE_CUDA
  return tensorflow::se::CompileGpuAsm(
      std::get<0>(compute_capability), std::get<1>(compute_capability),
      ptx.c_str(), xla::gpu::PtxOptsFromConfig(config));
#else
  return InternalError(
      "GOOGLE_CUDA not defined. Did you specify --config=cuda ?");
#endif
}
