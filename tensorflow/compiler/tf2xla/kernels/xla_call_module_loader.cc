/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/client/xla_computation.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

namespace {

// When adding a new version, write when it was added. Also change the default
// version in the constructor in xla.py.
// See
// https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
// for a description of the different versions.

constexpr int kVersionStartStableHloCompatibility = 4;
constexpr int kVersionStartSupportCallTFGraph = 5;
constexpr int kVersionStartSupportDisabledChecks = 6;
constexpr int kVersionStartSupportShapeAssertions = 7;
constexpr int kVersionStartSupportUsesShapePolymorphismAttr = 8;
constexpr int kVersionMinimumSupported = kVersionStartStableHloCompatibility;

// This should match xla.py:call_module_maximum_supported_version
constexpr int kVersionMaximumSupported =
    kVersionStartSupportUsesShapePolymorphismAttr;

constexpr llvm::StringRef kDisabledCheckPlatform = "platform";

bool IsPlatformCheckDisabled(absl::Span<const std::string> disabled_checks) {
  return llvm::is_contained(disabled_checks, kDisabledCheckPlatform);
}

constexpr llvm::StringRef kDisabledCheckShapeAssertions = "shape_assertions";

bool IsShapeAssertionsCheckDisabled(
    absl::Span<const std::string> loading_disabled_checks) {
  return llvm::is_contained(loading_disabled_checks,
                            kDisabledCheckShapeAssertions);
}

constexpr llvm::StringRef kUsesShapePolymorphismAttr =
    "jax.uses_shape_polymorphism";

// For a module whose "main" takes a first platform_index argument, sets
// the argument to a constant value, and erases the argument.
tsl::Status SetPlatformIndex(mlir::func::FuncOp main, int platform_index) {
  mlir::Block &main_body = main.front();

  if (main.getNumArguments() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The module should have a platform index argument but it has no ",
        "arguments"));
  }
  mlir::OpBuilder op_builder(main);
  op_builder.setInsertionPointToStart(&main_body);
  mlir::BlockArgument platform_index_arg = main_body.getArgument(0);
  mlir::RankedTensorType arg_ranked_type =
      platform_index_arg.getType().dyn_cast<mlir::RankedTensorType>();
  if (!arg_ranked_type || arg_ranked_type.getRank() != 0 ||
      !(arg_ranked_type.getElementType().isSignlessInteger(32) ||
        arg_ranked_type.getElementType().isSignlessInteger(64))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module argument at index 0 should be a 0-dimensional "
                     "32-bit or 64-bit integer-tensor platform index argument "
                     "but has type ",
                     mlir::debugString(platform_index_arg.getType())));
  }
  bool is_32_bit = arg_ranked_type.getElementType().isSignlessInteger(32);
  auto const_attr = is_32_bit ? op_builder.getI32IntegerAttr(platform_index)
                              : op_builder.getI64IntegerAttr(platform_index);
  auto platform_index_op = op_builder.create<mlir::stablehlo::ConstantOp>(
      platform_index_arg.getLoc(), const_attr);
  platform_index_arg.replaceAllUsesWith(platform_index_op);

  main.eraseArgument(0);

  return tsl::OkStatus();
}

}  // namespace

tsl::StatusOr<std::unique_ptr<XlaCallModuleLoader>> XlaCallModuleLoader::Create(
    mlir::MLIRContext *context, int version, std::string module_str,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, std::string loading_platform,
    int num_invocation_args, bool main_has_token_input_output) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadAndPreprocessModule(
      context, version, std::move(module_str), std::move(disabled_checks),
      std::move(platforms), std::move(loading_platform), num_invocation_args,
      main_has_token_input_output));
  return loader;
}

tsl::Status XlaCallModuleLoader::RefineDynamicShapes(
    llvm::ArrayRef<xla::Shape> input_shapes) {
  // Skip shape refinement for new versions if USES_SHAPE_POLYMORPHISM_ATTR=1
  if (version_ >= kVersionStartSupportUsesShapePolymorphismAttr) {
    if (mlir::Attribute uses_shape_poly_attr =
            (*module_)->getAttr(kUsesShapePolymorphismAttr)) {
      mlir::BoolAttr uses_shape_poly_bool_attr =
          llvm::dyn_cast<mlir::BoolAttr>(uses_shape_poly_attr);

      if (!uses_shape_poly_bool_attr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "jax.uses_shape_polymorphism is not a boolean attribute: ",
            mlir::debugString(uses_shape_poly_attr)));
      }
      if (!uses_shape_poly_bool_attr.getValue()) {
        VLOG(3) << "XlaCallModule skipping shape refinement due to module "
                << " attribute " << kUsesShapePolymorphismAttr.str() << "="
                << mlir::debugString(uses_shape_poly_attr);
        return tsl::OkStatus();
      }
    } else {
      VLOG(3) << "XlaCallModule skipping shape refinement due to module "
              << " attribute " << kUsesShapePolymorphismAttr.str()
              << " missing";
      return tsl::OkStatus();
    }
  }

  mlir::Block &main_body = main_.front();
  int non_dimension_arguments = input_shapes.size();

  mlir::Builder builder(module_->getContext());
  std::vector<mlir::Type> static_array_input_types(non_dimension_arguments);
  for (int i = 0, end = non_dimension_arguments; i < end; ++i) {
    const xla::Shape &xla_shape = input_shapes[i];
    if (xla_shape.IsToken()) {
      static_array_input_types[i] = mlir::stablehlo::TokenType::get(context_);
    } else {
      std::vector<int64_t> xla_dimensions(xla_shape.dimensions().begin(),
                                          xla_shape.dimensions().end());
      TF_ASSIGN_OR_RETURN(
          mlir::Type element_type,
          ConvertPrimitiveTypeToMLIRType(xla_shape.element_type(), builder));
      mlir::RankedTensorType type =
          mlir::RankedTensorType::get(xla_dimensions, element_type);
      // TODO(burmako): This fails with an obscure compilation error.
      // TF_ASSIGN_OR_RETURN(
      //     mlir::Type type,
      //     ConvertShapeToType<mlir::RankedTensorType>(xla_shape, builder));
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(type);
      mlir::TensorType arg_type =
          main_body.getArgument(i).getType().dyn_cast<mlir::TensorType>();
      if (arg_type == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Argument ", i, " passed to XlaCallModule is not a tensor"));
      }

      if (arg_type.getElementType() != type.getElementType()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Element type mismatch for argument ", i,
            " passed to XlaCallModule: ", "expecting ",
            mlir::debugString(arg_type), ", got ", mlir::debugString(type)));
      }

      if (auto ranked_arg_type = arg_type.dyn_cast<mlir::RankedTensorType>()) {
        if (mlir::failed(mlir::verifyCompatibleShape(ranked_arg_type.getShape(),
                                                     type.getShape()))) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Shape mismatch for argument ", i,
              " passed to XlaCallModule: ", "expecting ",
              mlir::debugString(arg_type), ", got ", mlir::debugString(type)));
        }
      }

      static_array_input_types[i] = type;
    }
  }

  // Refine 'main' argument types to use static input types instead. The main
  // arguments may occur as return values, or as inputs to called functions,
  // and changing their types may invalidate the module. To prevent this
  // we insert dummy conversion ops as the sole uses of the main arguments.
  // If we use stablehlo.convert, we end up with "convert 3xf32 -> *xf32"
  // after we set the static shapes for the main arguments. The "convert"
  // op does not support unranked result for ranked inputs. So, we use
  // "bitcast_convert", which is more flexible in the relationship between
  // the input and the result.
  mlir::OpBuilder op_builder(module_->getBodyRegion());
  op_builder.setInsertionPointToStart(&main_body);
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    mlir::BlockArgument arg = main_body.getArgument(i);
    auto convert_op = op_builder.create<mlir::stablehlo::BitcastConvertOp>(
        arg.getLoc(), arg.getType(), arg);
    arg.replaceAllUsesExcept(convert_op, convert_op);
  }

  auto static_array_output_types = llvm::to_vector(main_.getResultTypes());
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    auto arg = main_body.getArgument(i);
    arg.setType(static_array_input_types[i]);
  }
  main_.setType(builder.getFunctionType(static_array_input_types,
                                        static_array_output_types));
  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_refined_input_types", *module_);
  }
  bool enable_shape_assertions =
      (version_ >= kVersionStartSupportShapeAssertions &&
       !IsShapeAssertionsCheckDisabled(loading_disabled_checks_));
  TF_RETURN_IF_ERROR(
      xla::RefinePolymorphicShapes(*module_, enable_shape_assertions));

  if (VLOG_IS_ON(3)) {
    DumpMlirOpToFile("xla_call_module.after_shape_refinement", *module_);
  }
  return tsl::OkStatus();
}

tsl::Status XlaCallModuleLoader::LoadAndPreprocessModule(
    mlir::MLIRContext *context, int version, std::string module_str,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, std::string loading_platform,
    int num_invocation_args, bool main_has_token_input_output) {
  context_ = context;
  version_ = version;

  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::stablehlo::StablehloDialect>();
  context_->loadDialect<mlir::mhlo::MhloDialect>();
  context_->loadDialect<mlir::chlo::ChloDialect>();
  context_->loadDialect<mlir::vhlo::VhloDialect>();

  if (version >= kVersionStartSupportDisabledChecks && platforms.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " must have non-empty platforms."));
  }

  // Parses both IR text and bytecode.
  if (version >= kVersionStartStableHloCompatibility) {
    module_ =
        mlir::stablehlo::deserializePortableArtifact(module_str, context_);
  } else {
    module_ = mlir::parseSourceString<mlir::ModuleOp>(module_str, context_);
  }

  loading_disabled_checks_ = disabled_checks;
  loading_disabled_checks_.insert(
      loading_disabled_checks_.end(),
      GetXlaCallModuleFlags()->disabled_checks.begin(),
      GetXlaCallModuleFlags()->disabled_checks.end());
  if (!module_) {
    return absl::InvalidArgumentError("Cannot deserialize computation");
  }

  VLOG(3) << "Parsed serialized module (version = " << version
          << ", platforms = [" << absl::StrJoin(platforms, ", ")
          << "], loading_platform = " << loading_platform
          << ", main_has_token_input_output = " << main_has_token_input_output
          << ", disabled_checks = [" << absl::StrJoin(disabled_checks, ", ")
          << "], loading_disabled_checks = ["
          << absl::StrJoin(loading_disabled_checks_, ", ") << "]), module = "
          << DumpMlirOpToFile("xla_call_module.parsed", *module_);

  if (version < kVersionMinimumSupported) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModuleOp with version ", version,
        " is not supported anymore. Must be >= ", kVersionMinimumSupported));
  }
  if (version > kVersionMaximumSupported) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " is not supported by this build. Must be <= ",
                     kVersionMaximumSupported));
  }

  platform_index_ = -1;
  if (!platforms.empty()) {
    auto found_platform =
        std::find(platforms.begin(), platforms.end(), loading_platform);
    if (found_platform == platforms.end()) {
      if (!IsPlatformCheckDisabled(loading_disabled_checks_)) {
        return absl::NotFoundError(absl::StrCat(
            "The current platform ", loading_platform,
            " is not among the platforms required by the module: [",
            absl::StrJoin(platforms, ", "), "]"));
      } else {
        if (platforms.size() > 1) {
          platform_index_ = 0;
        }
      }
    } else {
      // We only use a platform index arguments if we support at least 2
      // platforms.
      if (platforms.size() > 1) {
        platform_index_ = found_platform - platforms.begin();
      }
    }
  }

  {
    mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());
    if (mlir::failed(mlir::verify(*module_))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Error verifying module: ", diag_handler.ConsumeStatus().ToString()));
    }
  }
  main_ = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_) {
    return absl::InvalidArgumentError("Cannot find 'main' in module");
  }

  if (platforms.size() > 1) {
    VLOG(3) << "XlaCallModule setting the platform_index to "
            << platform_index_;
    TF_RETURN_IF_ERROR(SetPlatformIndex(main_, platform_index_));
  }

  mlir::Block &main_body = main_.front();
  int nr_token_arguments = main_has_token_input_output ? 1 : 0;
  int nr_platform_args = (platform_index_ >= 0 ? 1 : 0);
  if (num_invocation_args != main_body.getNumArguments() - nr_token_arguments) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments passed to XlaCallModule = ",
        num_invocation_args, ". The module main function takes ",
        main_body.getNumArguments() + nr_platform_args + nr_token_arguments,
        " arguments of which ", nr_platform_args, " platform index arguments, ",
        "and ", nr_token_arguments, " token arguments. It must be called with ",
        main_body.getNumArguments() - nr_token_arguments, " arguments."));
  }
  return tsl::OkStatus();
}

tsl::Status XlaCallModuleLoader::ValidateDialect() {
  mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());
  bool moduleHasUnsupportedDialects = false;

  module_->walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func and StableHLO dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>(
            op->getDialect())) {
      moduleHasUnsupportedDialects = true;
      op->emitOpError() << "is an op from an unsupported dialect";
    }
  });

  if (moduleHasUnsupportedDialects) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module has unsupported dialects: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return tsl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateStaticShapes() {
  return xla::ValidateStaticShapes(*module_);
}

absl::Status XlaCallModuleLoader::LowerModuleToMhlo() {
  mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());

  mlir::PassManager pm(module_->getContext());
  applyTensorflowAndCLOptions(pm);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createChloLegalizeToHloPass(
      /*legalizeBroadcasts=*/true, /*expandCompositions=*/true));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // In order to export to XLA, we must sink constants to control flow
  // regions, since XLA uses functional control flow.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());
  if (failed(pm.run(*module_))) {
    return absl::InternalError(
        absl::StrCat("MHLO->HLO lowering passes failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }

  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_mhlo_lowering", *module_);
  }

  return absl::OkStatus();
}

tsl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::HloProto proto;
  mlir::MlirToHloConversionOptions options;
  TF_RETURN_IF_ERROR(
      mlir::ConvertMlirHloToHlo(*module_, &proto, /*use_tuple_args=*/false,
                                /*return_tuple=false*/ false, options));
  return xla::XlaComputation(std::move(*proto.mutable_hlo_module()));
}

}  // namespace tensorflow
