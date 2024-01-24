/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/mlir_to_hlo.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/util.h"

namespace xla {

namespace {

static mlir::Attribute ArrayToElements(mlir::Attribute attr) {
  if (!attr) return attr;  // Handle optional attrs
  if (auto array = attr.dyn_cast<mlir::DenseI64ArrayAttr>()) {
    return mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(array.size(), array.getElementType()),
        array.asArrayRef());
  }
  if (auto array = attr.dyn_cast<mlir::DenseBoolArrayAttr>()) {
    return mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(array.size(), array.getElementType()),
        array.asArrayRef());
  }
  return attr;
}

static mlir::Attribute ElementsToArray(mlir::Attribute attr) {
  if (!attr) return attr;  // Handle optional attrs
  if (auto elements = llvm::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
    if (elements.getElementType().isInteger(64)) {
      return mlir::DenseI64ArrayAttr::get(
          attr.getContext(), llvm::to_vector(elements.getValues<int64_t>()));
    }
    return mlir::DenseBoolArrayAttr::get(
        attr.getContext(), llvm::to_vector(elements.getValues<bool>()));
  }
  return attr;
}

// Convert attrs that use DenseI64ArrayAttr (or DenseBoolArrayAttr) to use a
// different type of Attribute. For backwards compatibility purposes, arrays
// should be converted to DenseIntElementsAttr right before serialization, and
// converted back right after serialization. Deserialization checks the IR is
// valid by default, so you will need to disable that and do the verification
// explicitly after parsing.
void ConvertStablehloDenseAttributes(
    mlir::Operation* root_op,
    llvm::function_ref<mlir::Attribute(mlir::Attribute)> convert) {
  llvm::TypeSwitch<mlir::Operation*>(root_op)
      .Case([&](mlir::stablehlo::BroadcastOp op) {
        op->setAttr("broadcast_sizes", convert(op->getAttr("broadcast_sizes")));
      })
      .Case([&](mlir::stablehlo::BroadcastInDimOp op) {
        op->setAttr("broadcast_dimensions",
                    convert(op->getAttr("broadcast_dimensions")));
      })
      .Case([&](mlir::stablehlo::ConvolutionOp op) {
        op->setAttr("window_strides", convert(op->getAttr("window_strides")));
        op->setAttr("lhs_dilation", convert(op->getAttr("lhs_dilation")));
        op->setAttr("rhs_dilation", convert(op->getAttr("rhs_dilation")));
        op->setAttr("window_reversal", convert(op->getAttr("window_reversal")));
      })
      .Case([&](mlir::stablehlo::DynamicBroadcastInDimOp op) {
        op->setAttr("broadcast_dimensions",
                    convert(op->getAttr("broadcast_dimensions")));
        op->setAttr("known_expanding_dimensions",
                    convert(op->getAttr("known_expanding_dimensions")));
        op->setAttr("known_nonexpanding_dimensions",
                    convert(op->getAttr("known_nonexpanding_dimensions")));
      })
      .Case([&](mlir::stablehlo::DynamicConvOp op) {
        op->setAttr("window_strides", convert(op->getAttr("window_strides")));
        op->setAttr("lhs_dilation", convert(op->getAttr("lhs_dilation")));
        op->setAttr("rhs_dilation", convert(op->getAttr("rhs_dilation")));
        op->setAttr("window_reversal", convert(op->getAttr("window_reversal")));
      })
      .Case([&](mlir::stablehlo::DynamicSliceOp op) {
        op->setAttr("slice_sizes", convert(op->getAttr("slice_sizes")));
      })
      .Case([&](mlir::stablehlo::FftOp op) {
        op->setAttr("fft_length", convert(op->getAttr("fft_length")));
      })
      .Case([&](mlir::stablehlo::GatherOp op) {
        op->setAttr("slice_sizes", convert(op->getAttr("slice_sizes")));
      })
      .Case([&](mlir::stablehlo::MapOp op) {
        op->setAttr("dimensions", convert(op->getAttr("dimensions")));
      })
      .Case([&](mlir::stablehlo::PadOp op) {
        op->setAttr("edge_padding_low",
                    convert(op->getAttr("edge_padding_low")));
        op->setAttr("edge_padding_high",
                    convert(op->getAttr("edge_padding_high")));
        op->setAttr("interior_padding",
                    convert(op->getAttr("interior_padding")));
      })
      .Case([&](mlir::stablehlo::ReduceOp op) {
        op->setAttr("dimensions", convert(op->getAttr("dimensions")));
      })
      .Case([&](mlir::stablehlo::ReduceWindowOp op) {
        op->setAttr("window_dimensions",
                    convert(op->getAttr("window_dimensions")));
        op->setAttr("window_strides", convert(op->getAttr("window_strides")));
        op->setAttr("base_dilations", convert(op->getAttr("base_dilations")));
        op->setAttr("window_dilations",
                    convert(op->getAttr("window_dilations")));
      })
      .Case([&](mlir::stablehlo::ReverseOp op) {
        op->setAttr("dimensions", convert(op->getAttr("dimensions")));
      })
      .Case([&](mlir::stablehlo::SelectAndScatterOp op) {
        op->setAttr("window_dimensions",
                    convert(op->getAttr("window_dimensions")));
        op->setAttr("window_strides", convert(op->getAttr("window_strides")));
      })
      .Case([&](mlir::stablehlo::SliceOp op) {
        op->setAttr("start_indices", convert(op->getAttr("start_indices")));
        op->setAttr("limit_indices", convert(op->getAttr("limit_indices")));
        op->setAttr("strides", convert(op->getAttr("strides")));
      })
      .Case([&](mlir::stablehlo::TransposeOp op) {
        op->setAttr("permutation", convert(op->getAttr("permutation")));
      });
}

void DowngradeStablehlo(mlir::ModuleOp module) {
  module->walk([](mlir::Operation* op) {
    ConvertStablehloDenseAttributes(op, ArrayToElements);
  });
}
void UpgradeStablehlo(mlir::ModuleOp module) {
  module->walk([](mlir::Operation* op) {
    ConvertStablehloDenseAttributes(op, ElementsToArray);
  });
}

}  // namespace

Status MlirToXlaComputation(mlir::ModuleOp module,
                            XlaComputation& xla_computation,
                            bool use_tuple_args, bool return_tuple,
                            bool legalize_sparse_ops) {
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(module->getContext());
  {
    mlir::PassManager pm(module->getContext());
    if (legalize_sparse_ops) {
      // Convert sparse operations to custom_calls in order to translate sparse
      // operations into XLA HLO.
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::mhlo::createLegalizeSparseOperationsPass(
              /*legalizeToCustomCalls=*/true));
    }
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHloPass(
            /*legalizeBroadcasts=*/true, /*expandCompositions=*/true));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSinkConstantsToControlFlowPass());
    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      module->dump();
      return diagnostic_handler.ConsumeStatus();
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  HloProto proto;
  mlir::MlirToHloConversionOptions options;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(module, &proto, use_tuple_args,
                                         return_tuple, options));

  xla_computation = XlaComputation(std::move(*proto.mutable_hlo_module()));
  return OkStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_module_str.data(), mlir_module_str.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/false});
  if (!module) {
    return diagnostic_handler.ConsumeStatus();
  }

  // In
  // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
  // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
  // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
  // dense elements, not dense arrays, so when serializing we always convert the
  // arrays to elements. The elements need to be converted back to arrays when
  // deserializing.
  // TODO: b/320507168 - Remove the conversion code, and verifyAfterParse.
  TF_RETURN_IF_ERROR(UpgradeVersionedStablehlo(*module));
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return diagnostic_handler.ConsumeStatus();
  }
  return std::move(module);
}

Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));
  return xla::MlirToXlaComputation(*module, xla_computation, use_tuple_args,
                                   return_tuple);
}

StatusOr<std::string> SerializeUsingNativeBytecode(mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  // Pin bytecode version to 1 until transition to stable.
  // TODO: b/285913864 - Remove post enabling frameworks to set it.
  config.setDesiredBytecodeVersion(1);
  // In
  // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
  // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
  // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
  // dense elements, not dense arrays, so convert the arrays to elements before
  // serializing. The elements need to be converted back to arrays when
  // deserializing.
  // TODO: b/320507168 - Remove this conversion code.
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  DowngradeStablehlo(*cloned);
  if (mlir::failed(mlir::writeBytecodeToFile(*cloned, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view target, bool inplace) {
  // Legalize CHLO -> [MHLO+Shape] -> StableHLO
  mlir::PassManager pm(mlir_module->getContext());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createChloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createShapeLegalizeToHloPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(mlir_module))) {
    return xla::InvalidArgument("CHLO => [MHLO+Shape] => StableHLO failed");
  }

  // Avoid mutating the original module if it will be reused elsewhere
  mlir::OwningOpRef<mlir::ModuleOp> cloned = mlir_module.clone();
  if (!inplace) {
    cloned = mlir_module.clone();
    mlir_module = *cloned;
  }

  // Serialize portable artifact
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (failed(
          mlir::stablehlo::serializePortableArtifact(mlir_module, target, os)))
    return xla::InvalidArgument("Failed to serialize StableHLO");
  return buffer;
}

Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module) {
  // Apply StableHLO bytecode patch
  UpgradeStablehlo(mlir_module);

  // Upgrade if VHLO
  mlir::PassManager pm(mlir_module->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(mlir_module)))
    return xla::InvalidArgument("Failed to upgrade versioned StableHLO.");
  return OkStatus();
}

}  // namespace xla
