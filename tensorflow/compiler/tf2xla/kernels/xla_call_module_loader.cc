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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
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
#include "xla/mlir/utils/type_util.h"
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
constexpr int kVersionStartSupportEffects = 9;
constexpr int kVersionMinimumSupported = kVersionStartStableHloCompatibility;

// This should match xla.py:call_module_maximum_supported_version
constexpr int kVersionMaximumSupported = kVersionStartSupportEffects;

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

constexpr llvm::StringRef kCustomCallShimTarget =
    "stablehlo.shape_refinement_operand_wrapper";

}  // namespace

bool IsTokenType(mlir::Type type) {
  return mlir::isa<mlir::stablehlo::TokenType>(type) ||
         mlir::isa<mlir::mhlo::TokenType>(type);
}

absl::StatusOr<std::unique_ptr<XlaCallModuleLoader>>
XlaCallModuleLoader::Create(mlir::MLIRContext *context, int version,
                            std::string module_str,
                            std::vector<std::string> disabled_checks,
                            std::vector<std::string> platforms,
                            int num_invocation_args,
                            bool main_has_token_input_output) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadModule(
      context, version, std::move(module_str), std::move(disabled_checks),
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

  main_.eraseArgument(0);
  platform_index_arg_set_ = true;
  return absl::OkStatus();
}

static mlir::stablehlo::CustomCallOp MakeShapeRefinementOperandWrapper(
    mlir::OpBuilder op_builder, mlir::Value operand,
    llvm::ArrayRef<int64_t> shape) {
  auto constant = op_builder.create<mlir::stablehlo::ConstantOp>(
      operand.getLoc(), op_builder.getI64TensorAttr(shape));
  return op_builder.create<mlir::stablehlo::CustomCallOp>(
      operand.getLoc(), operand.getType(), mlir::ValueRange{operand, constant},
      llvm::SmallVector<mlir::NamedAttribute>{
          op_builder.getNamedAttr(
              "call_target_name",
              op_builder.getStringAttr(kCustomCallShimTarget)),
          op_builder.getNamedAttr("indices_of_shape_operands",
                                  op_builder.getI64TensorAttr({1})),
          op_builder.getNamedAttr("has_side_effect",
                                  op_builder.getBoolAttr(false)),
      });
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
      static_array_input_types[i] = mlir::stablehlo::TokenType::get(context_);
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(static_array_input_types[i])
              << " for argument type " << mlir::debugString(arg_type);
    } else {
      const xla::Shape &xla_shape = input_shapes[next_actual_input++];
      std::vector<int64_t> xla_dimensions(xla_shape.dimensions().begin(),
                                          xla_shape.dimensions().end());
      TF_ASSIGN_OR_RETURN(
          mlir::Type element_type,
          ConvertPrimitiveTypeToMlirType(xla_shape.element_type(), builder));
      mlir::RankedTensorType type =
          mlir::RankedTensorType::get(xla_dimensions, element_type);
      // TODO(burmako): This fails with an obscure compilation error on Windows.
      // TF_ASSIGN_OR_RETURN(
      //     mlir::Type type,
      //     ConvertShapeToType<mlir::RankedTensorType>(xla_shape, builder));
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(type) << " for argument type "
              << mlir::debugString(arg_type);
      mlir::TensorType arg_type =
          mlir::dyn_cast<mlir::TensorType>(main_body.getArgument(i).getType());
      if (arg_type == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Argument ", i, " passed to XlaCallModule is not a tensor, ",
            "has type ",
            mlir::debugString(main_body.getArgument(i).getType())));
      }

      if (arg_type.getElementType() != type.getElementType()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Element type mismatch for argument ", i,
            " passed to XlaCallModule: ", "expecting ",
            mlir::debugString(arg_type), ", got ", mlir::debugString(type)));
      }

      if (auto ranked_arg_type =
              mlir::dyn_cast<mlir::RankedTensorType>(arg_type)) {
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
  mlir::OpBuilder op_builder(module_->getBodyRegion());
  op_builder.setInsertionPointToStart(&main_body);
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    mlir::BlockArgument arg = main_body.getArgument(i);
    mlir::Type arg_type = arg.getType();
    bool is_input_refined = arg_type == static_array_input_types[i];
    if (IsTokenType(arg_type) || is_input_refined) {
      continue;
    }
    auto ranked_arg_type = mlir::dyn_cast<mlir::RankedTensorType>(arg_type);
    if (!ranked_arg_type || !ranked_arg_type.hasStaticShape()) {
      auto type =
          mlir::cast<mlir::RankedTensorType>(static_array_input_types[i]);
      auto custom_call =
          MakeShapeRefinementOperandWrapper(op_builder, arg, type.getShape());
      auto call_result = custom_call.getResult(0);
      arg.replaceAllUsesExcept(call_result, custom_call);
    }
  }

  // Actually update main's input types.
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    auto arg = main_body.getArgument(i);
    arg.setType(static_array_input_types[i]);
  }
  main_.setType(builder.getFunctionType(static_array_input_types,
                                        main_.getResultTypes()));
  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_refined_input_types", *module_);
  }
  bool enable_shape_assertions =
      (version_ >= kVersionStartSupportShapeAssertions &&
       !IsShapeAssertionsCheckDisabled(loading_disabled_checks_));
  TF_RETURN_IF_ERROR(
      xla::RefinePolymorphicShapes(*module_, enable_shape_assertions));

  // Clean up custom_call shims.
  for (auto call : llvm::make_early_inc_range(
           main_body.getOps<mlir::stablehlo::CustomCallOp>())) {
    if (mlir::cast<mlir::StringAttr>(call->getAttr("call_target_name"))
            .strref() == kCustomCallShimTarget) {
      auto operand = call->getOperand(0);
      auto result = call->getResult(0);
      if (operand.getType() != result.getType()) {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "custom_call shim shape refinement failed, input type does not "
              "match output type: "
           << operand.getType() << " != " << result.getType();
        return absl::InvalidArgumentError(os.str());
      }
      call->getResult(0).replaceAllUsesExcept(call->getOperand(0), call);
      call.erase();
    }
  }

  if (VLOG_IS_ON(3)) {
    DumpMlirOpToFile("xla_call_module.after_shape_refinement", *module_);
  }

  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::LoadModule(
    mlir::MLIRContext *context, int version, std::string module_str,
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
  if (!module_) {
    return absl::InvalidArgumentError("Cannot deserialize computation");
  }
  VLOG(3) << "Parsed serialized module (version = " << version
          << ", platforms = [" << absl::StrJoin(platforms, ", ")
          << "], main_has_token_input_output = " << main_has_token_input_output
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

absl::Status XlaCallModuleLoader::ValidateDialect() {
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
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateStaticShapes() {
  return xla::ValidateStaticShapes(*module_);
}

absl::Status XlaCallModuleLoader::LowerModuleToMhlo() {
  mlir::StatusScopedDiagnosticHandler diag_handler(module_->getContext());

  mlir::PassManager pm(module_->getContext());
  applyTensorflowAndCLOptions(pm);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createChloLegalizeToHloPass());
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

absl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::HloProto proto;
  mlir::MlirToHloConversionOptions options;
  TF_RETURN_IF_ERROR(
      mlir::ConvertMlirHloToHlo(*module_, &proto, /*use_tuple_args=*/false,
                                /*return_tuple=false*/ false, options));
  return xla::XlaComputation(std::move(*proto.mutable_hlo_module()));
}

}  // namespace tensorflow
