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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
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
#include "stablehlo/transforms/StablehloRefineShapes.h"  // from @stablehlo
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

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
constexpr int kVersionStartSupportEffects = 9;
constexpr int kVersionStartSupportShardyPartitioner = 10;
constexpr int kVersionMinimumSupported = kVersionStartStableHloCompatibility;

// This should match xla.py:call_module_maximum_supported_version
constexpr int kVersionMaximumSupported = kVersionStartSupportShardyPartitioner;

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

}  // namespace

bool IsTokenType(mlir::Type type) {
  return mlir::isa<mlir::stablehlo::TokenType>(type);
}

absl::StatusOr<std::unique_ptr<XlaCallModuleLoader>>
XlaCallModuleLoader::Create(mlir::MLIRContext *context, int version,
                            mlir::StringRef module_str,
                            std::vector<std::string> disabled_checks,
                            std::vector<std::string> platforms,
                            int num_invocation_args,
                            bool main_has_token_input_output) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadModule(
      context, version, module_str, std::move(disabled_checks),
      std::move(platforms), num_invocation_args, main_has_token_input_output));
  return loader;
}

absl::Status XlaCallModuleLoader::SetPlatformIndex(
    absl::string_view compilation_platform) {
  int platform_index = -1;
  if (!platforms_.empty()) {
    auto found_platform =
        std::find(platforms_.begin(), platforms_.end(), compilation_platform);
    if (found_platform == platforms_.end()) {
      if (!IsPlatformCheckDisabled(loading_disabled_checks_)) {
        return absl::NotFoundError(absl::StrCat(
            "The current platform ", compilation_platform,
            " is not among the platforms required by the module: [",
            absl::StrJoin(platforms_, ", "), "]"));
      } else {
        if (platforms_.size() > 1) {
          platform_index = 0;
        }
      }
    } else {
      // We only use a platform index argument if we support at least 2
      // platforms.
      if (platforms_.size() > 1) {
        platform_index = found_platform - platforms_.begin();
      }
    }
  }

  if (platform_index < 0) return absl::OkStatus();
  VLOG(3) << "XlaCallModule setting the platform_index to " << platform_index
          << " for platform " << compilation_platform << ".";
  mlir::Block &main_body = main_.front();

  if (main_.getNumArguments() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The module should have a platform index argument but it has no ",
        "arguments"));
  }
  mlir::OpBuilder op_builder(main_);
  op_builder.setInsertionPointToStart(&main_body);
  mlir::BlockArgument platform_index_arg = main_body.getArgument(0);
  mlir::RankedTensorType arg_ranked_type =
      mlir::dyn_cast<mlir::RankedTensorType>(platform_index_arg.getType());
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

  CHECK(llvm::succeeded(main_.eraseArgument(0)));
  platform_index_arg_set_ = true;
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::RefineDynamicShapes(
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
        return absl::OkStatus();
      }
    } else {
      VLOG(3) << "XlaCallModule skipping shape refinement due to module "
              << " attribute " << kUsesShapePolymorphismAttr.str()
              << " missing";
      return absl::OkStatus();
    }
  }
  // Add the tokens to the input_shapes. Starting with version 9, the main
  // function may take token arguments that do not correspond with op inputs.
  int nr_inputs = NrInputs();
  int nr_expected_tokens = llvm::count_if(InputTypes(), IsTokenType);
  bool has_platform_index_arg =
      platforms_.size() > 1 && !platform_index_arg_set_;
  int nr_expected_platform_index_args = has_platform_index_arg ? 1 : 0;
  if (input_shapes.size() !=
      nr_inputs - nr_expected_tokens - nr_expected_platform_index_args) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModule RefineDynamicShapes called with ", input_shapes.size(),
        " input shapes, but the main function takes ",
        nr_inputs - nr_expected_tokens - nr_expected_platform_index_args,
        " non-token and non-platform-index arguments. The input ",
        "shapes are (",
        absl::StrJoin(input_shapes, ", ",
                      [](std::string *out, const xla::Shape &s) {
                        absl::StrAppend(out, s.ToString());
                      }),
        ") and the main function argument types are ",
        absl::StrJoin(InputTypes(), ", ",
                      [](std::string *out, const mlir::Type &t) {
                        absl::StrAppend(out, mlir::debugString(t));
                      }),
        ")"));
  }

  // Derive static input types to use for main.
  mlir::Block &main_body = main_.front();
  mlir::Builder builder(module_->getContext());
  std::vector<mlir::Type> static_array_input_types(nr_inputs);
  int next_actual_input = 0;
  for (int i = 0, end = nr_inputs; i < end; ++i) {
    mlir::Type arg_type = main_body.getArgument(i).getType();
    if (i == 0 && has_platform_index_arg) {
      static_array_input_types[i] = arg_type;
      continue;
    }
    if (IsTokenType(arg_type)) {
      static_array_input_types[i] = arg_type;
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(static_array_input_types[i])
              << " for argument type " << mlir::debugString(arg_type);
      continue;
    }

    // Get static MLIR Type from xla Shape.
    const xla::Shape &xla_shape = input_shapes[next_actual_input++];
    std::vector<int64_t> xla_dimensions;
    if (xla_shape.IsArray()) {
      xla_dimensions = std::vector<int64_t>(xla_shape.dimensions().begin(),
                                            xla_shape.dimensions().end());
    }
    TF_ASSIGN_OR_RETURN(
        mlir::Type element_type,
        ConvertPrimitiveTypeToMlirType(xla_shape.element_type(), builder));
    mlir::RankedTensorType type =
        mlir::RankedTensorType::get(xla_dimensions, element_type);

    VLOG(3) << "XlaCallModule static array input type #" << i << ": "
            << mlir::debugString(type) << " for argument type "
            << mlir::debugString(arg_type);
    static_array_input_types[i] = type;
  }

  // Insert custom_call ops as shims to maintain the validity of the module when
  // main's input types are changed later. This is a workaround to allow shape
  // refinement to be applied; the custom_calls are removed before returning.
  // Arguments to main may occur as return values, or as inputs to called
  // functions, and changing their types may invalidate the module due to type
  // mismatches. To prevent this, for each argument that is a dynamically-shaped
  // tensor, we insert a custom_call op that takes the argument as an input and
  // replace uses of the argument with the custom_call's result. custom_call
  // is used as it allows its inputs and outputs to be unranked.
  //
  // Example:
  //
  // The below main function returns its argument directly:
  //
  // func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Changing the argument's type invalidates the IR (type mismatch):
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Inserting a custom_call allows the IR to remain valid:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   %0 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
  //   %1 = stablehlo.custom_call
  //   @stablehlo.shape_refinement_operand_wrapper(%arg0, %0)
  //   {indices_of_shape_operands = dense<1> : tensor<1xi64>} :
  //   (tensor<2x3xf32>, tensor<2xi64>) -> tensor<*xf32>
  //   return %1 : tensor<*xf32>
  // }
  //
  // After shapes are refined and the custom_calls are removed, we get:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  //   return %arg0 : tensor<2x3xf32>
  // }
  //
  {
    mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());
    if (failed(mlir::stablehlo::refineArguments(main_,
                                                static_array_input_types))) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error refining argument shapes: ",
                       diag_handler.ConsumeStatus().ToString()));
    }
  }

  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_refined_input_types", *module_);
  }
  bool enable_shape_assertions =
      (version_ >= kVersionStartSupportShapeAssertions &&
       !IsShapeAssertionsCheckDisabled(loading_disabled_checks_));

  // Store the original output types before shape refinement.
  mlir::TypeRange original_output_types = OutputTypes();

  // RefinePolymorphicShapes will refine using the new static types and clean up
  // the shape_refinement_operand_wrapper custom calls.
  TF_RETURN_IF_ERROR(
      xla::RefinePolymorphicShapes(*module_, enable_shape_assertions));

  if (VLOG_IS_ON(3)) {
    DumpMlirOpToFile("xla_call_module.after_shape_refinement", *module_);
  }

  // Mark the output types as refined if they are different from the original
  // output types.
  if (OutputTypes() != original_output_types) {
    output_types_refined_ = true;
  }

  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::LoadModule(
    mlir::MLIRContext *context, int version, mlir::StringRef module_str,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, int num_invocation_args,
    bool main_has_token_input_output) {
  context_ = context;
  version_ = version;
  platforms_ = platforms;
  loading_disabled_checks_ = disabled_checks;
  loading_disabled_checks_.insert(
      loading_disabled_checks_.end(),
      GetXlaCallModuleFlags()->disabled_checks.begin(),
      GetXlaCallModuleFlags()->disabled_checks.end());

  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::stablehlo::StablehloDialect>();
  context_->loadDialect<mlir::chlo::ChloDialect>();
  context_->loadDialect<mlir::vhlo::VhloDialect>();

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
  if (version >= kVersionStartSupportDisabledChecks && platforms.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " must have non-empty platforms."));
  }

  // Parse the StableHLO/VHLO bytecode
  {
    mlir::StatusScopedDiagnosticHandler diag_handler(context_);
    module_ =
        mlir::stablehlo::deserializePortableArtifact(module_str, context_);
    if (!module_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot deserialize computation: ",
                       diag_handler.ConsumeStatus().ToString()));
    }
  }
  VLOG(3) << "Parsed serialized module (version = " << version
          << ", platforms = [" << absl::StrJoin(platforms, ", ")
          << "], main_has_token_input_output = " << main_has_token_input_output
          << ", disabled_checks = [" << absl::StrJoin(disabled_checks, ", ")
          << "], loading_disabled_checks = ["
          << absl::StrJoin(loading_disabled_checks_, ", ") << "]), module = "
          << DumpMlirOpToFile("xla_call_module.parsed", *module_);

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

  mlir::Block &main_body = main_.front();

  int nr_token_arguments = llvm::count_if(InputTypes(), IsTokenType);
  if (version < kVersionStartSupportEffects) {
    bool has_token_at_start = (nr_token_arguments == 1 &&
                               IsTokenType(main_.getArgument(0).getType()));
    if (main_has_token_input_output != has_token_at_start) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected a token at start iff main_has_token_input_output. ",
          "Found main function type ",
          mlir::debugString(main_.getFunctionType()),
          " and main_has_token_input_output = ", main_has_token_input_output));
    }
  }
  int nr_platform_args = (platforms.size() > 1 ? 1 : 0);
  if (num_invocation_args !=
      main_body.getNumArguments() - nr_platform_args - nr_token_arguments) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments passed to XlaCallModule = ",
        num_invocation_args, ". It must be called with ",
        main_body.getNumArguments() - nr_platform_args - nr_token_arguments,
        " because the module main function takes ", main_body.getNumArguments(),
        " arguments of which ", nr_platform_args, " platform index arguments, ",
        "and ", nr_token_arguments, " token arguments."));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateXlaCallModuleInvariants() {
  mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());
  bool moduleValidationFailed = false;

  module_->walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func and StableHLO dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>(
            op->getDialect())) {
      op->emitOpError() << "is an op from an unsupported dialect";
      moduleValidationFailed = true;
    }
    // `shape_assertion` custom calls must have side effects. We check this here
    // because a pure `shape_assertion` is likely to be removed by MLIR's
    // dead-code elimination, preventing us from detecting the issue later.
    if (auto customCallOp = llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (!customCallOp.getHasSideEffect() &&
          customCallOp.getCallTargetName() == "shape_assertion") {
        op->emitOpError() << "`shape_assertion` custom calls must set "
                             "`has_side_effect = true`.";
        moduleValidationFailed = true;
      }
    }
  });

  if (moduleValidationFailed) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModule failed validation: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateStaticShapes() {
  return xla::ValidateStaticShapes(*module_);
}

absl::Status XlaCallModuleLoader::PrepareStablehloForLowering() {
  mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());

  // TODO (b/410057228): Replace MHLO canonicalization with StableHLO.
  // This code requires MHLO CaseOp canonicalization to remove unreachable
  // branches, else `tf.call_tf_function` inlining can fail.
  mlir::PassManager pm(module_->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

  if (failed(pm.run(*module_))) {
    return absl::InternalError(
        absl::StrCat("MHLO->HLO lowering passes failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }

  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_canonicalization", *module_);
  }

  return absl::OkStatus();
}

absl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::HloProto proto;
  TF_RETURN_IF_ERROR(xla::ConvertStablehloToHloProto(*module_, &proto));
  return xla::XlaComputation(std::move(*proto.mutable_hlo_module()));
}

}  // namespace tensorflow
