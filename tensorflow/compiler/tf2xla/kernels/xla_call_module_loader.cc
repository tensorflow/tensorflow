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
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/python/refine_polymorphic_shapes.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/regexp.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

namespace {

// When adding a new version, write when it was added. Also change the default
// version in the constructor in xla.py.
// See
// https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
// for a description of the different versions.

// TODO(b/283439649): Remove support for dim_args_spec.
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

// Computes a dimension value from the dim_arg specification.
// The specification is of the form "<arg_idx>.<arg_axis_idx>".
// TODO(b/283439649): Remove support for dim_args_spec.
tsl::StatusOr<mlir::Value> ComputeDimensionValue(
    int version, std::string dim_arg_spec, std::vector<mlir::Value> arguments,
    mlir::OpBuilder op_builder, mlir::Type dim_arg_type) {
  static const LazyRE2 dim_arg_spec_re = {R"((\d+).(\d+))"};
  int arg_idx, arg_axis_idx;
  if (!RE2::FullMatch(dim_arg_spec, *dim_arg_spec_re, &arg_idx,
                      &arg_axis_idx)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Syntax error in dim_args_spec '", dim_arg_spec, "'"));
  }
  if (arg_idx < 0 || arg_idx >= arguments.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid argument index ", arg_idx,
        " when the number of non-dimension arguments is ", arguments.size(),
        " in dim_arg_spec '", dim_arg_spec, "'"));
  }
  mlir::RankedTensorType arg_type =
      arguments[arg_idx].getType().dyn_cast<mlir::RankedTensorType>();
  if (!arg_type) {
    return absl::InvalidArgumentError(
        absl::StrCat("Argument ", arg_idx, " referenced in dim_arg_spec '",
                     dim_arg_spec, "' does not have a RankedTensorType"));
  }
  if (arg_axis_idx < 0 || arg_axis_idx >= arg_type.getShape().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid axis index ", arg_axis_idx,
        " when the rank of non-dimension argument ", arg_idx, " is ",
        arg_type.getShape().size(), " in dim_arg_spec '", dim_arg_spec, "'"));
  }
  mlir::Value val;
  mlir::Type get_dim_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  val = op_builder.create<mlir::stablehlo::GetDimensionSizeOp>(
      arguments[arg_idx].getLoc(), get_dim_type, arguments[arg_idx],
      op_builder.getI64IntegerAttr(arg_axis_idx));
  if (dim_arg_type != get_dim_type) {
    val = op_builder.create<mlir::stablehlo::ConvertOp>(
        arguments[arg_idx].getLoc(), dim_arg_type, val);
  }
  return val;
}

}  // namespace

tsl::StatusOr<std::unique_ptr<XlaCallModuleLoader>> XlaCallModuleLoader::Create(
    mlir::MLIRContext *context, int version, std::string module_str,
    std::vector<std::string> dim_args_spec,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, std::string loading_platform,
    int num_invocation_args, bool main_has_token_input_output) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadAndPreprocessModule(
      context, version, std::move(module_str), std::move(dim_args_spec),
      std::move(disabled_checks), std::move(platforms),
      std::move(loading_platform), num_invocation_args,
      main_has_token_input_output));
  return loader;
}

// Adds a wrapper for the "main" function to compute the platform index and the
// dimension arguments.
//
// The input module has the following structure:
//
//    func public main(%arg_platform_index: i32, %arg_dim0: i32, %arg_dim1: i32,
//                     %arg0: f32[?, ?, 8]) { ... }
//
// where %arg_platform_index is the index of the current compilation platform
// among the declared `platforms` (missing if version < 3 or if platforms has
// fewer than 2 elements), %arg_dim0 and %arg_dim1 are dimension arguments
// (missing if dim_args_spec is empty). The value of the dimension arguments
// are computed based on the static shapes of the actual arguments
// (%arg0 and following).
// In the above example, the dim_args_spec array would have two elements, one
// for %arg_dim0 and one for %arg_dim1. E.g., ['0.0', '0.1'] specifies that
// %arg_dim0 should be set to the size of axis 0 or array argument 0 (%arg0),
// while %arg_dim1 should be set to the size of axis 1.
// The platform index argument must be a 0-dimensional 32-bit integer, and the
// dimension arguments must be 0-dimensional tensors of integer type.
//
// We create a new "main" function as follows:
//   func public main(%arg0: f32[?, ?, 8]) {
//      %arg_platform_index = stablehlo.constant <platform_index>
//      %arg_dim0 = stablehlo.get_dimension_size(%arg0) dimension=0
//      %arg_dim1 = stablehlo.get_dimension_size(%arg0) dimension=1
//      %res = func.call _wrapped_main(%arg_platform_index,
//                                     %arg_dim0, %arg_dim1, %arg0)
//      return %res
//   }
//   func private _wrapped_main(%arg_platform_index: i32,
//                              %arg_dim0: i32, %arg_dim1: i32,
//                              %arg0: f32[?, ?, 8]) {
//      ... the original main function ...
//   }
//
// and then we run the inliner. This is important because in the
// RefineDynamicShapes method called in Compile we refine the shape of the
// array arguments. This would create a type error at the call to _wrapped_main
// with the expected type of %arg0.
tsl::Status XlaCallModuleLoader::AddMainWrapper() {
  int nr_dim_args = dim_args_spec_.size();
  // Locate the 'main' function.
  // This is the convention used by MlirToXlaComputation.
  mlir::func::FuncOp orig_main =
      module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!orig_main) {
    return absl::InvalidArgumentError("Cannot find 'main' in module");
  }
  int nr_platform_args = 0;
  if (platform_index_ >= 0) {
    nr_platform_args = 1;
  }
  if (orig_main.getNumArguments() <= nr_platform_args + nr_dim_args) {
    return absl::InvalidArgumentError(
        absl::StrCat("The module should have ", nr_platform_args,
                     " platform index arguments and ", nr_dim_args,
                     " dimension arguments, but it ", "has only ",
                     orig_main.getNumArguments(), " total arguments"));
  }
  mlir::Block &orig_main_body = orig_main.front();

  mlir::SymbolTable::setSymbolVisibility(
      orig_main, mlir::SymbolTable::Visibility::Private);
  mlir::OpBuilder op_builder(module_->getBodyRegion());
  orig_main.setName(op_builder.getStringAttr("_wrapped_main"));
  mlir::Location loc = module_->getLoc();
  std::vector<mlir::Type> new_main_arg_types(
      orig_main.getArgumentTypes().begin() + nr_platform_args + nr_dim_args,
      orig_main.getArgumentTypes().end());
  mlir::func::FuncOp new_main = op_builder.create<mlir::func::FuncOp>(
      loc, "main",
      mlir::FunctionType::get(module_->getContext(),
                              /*inputs=*/new_main_arg_types,
                              /*results=*/orig_main.getResultTypes()));
  mlir::SymbolTable::setSymbolVisibility(new_main,
                                         mlir::SymbolTable::Visibility::Public);
  mlir::Block *new_main_block = new_main.addEntryBlock();
  std::vector<mlir::Value> block_args(new_main_block->getArguments().begin(),
                                      new_main_block->getArguments().end());
  op_builder.setInsertionPointToStart(new_main_block);

  std::vector<mlir::Value> call_args(orig_main_body.getNumArguments());
  for (int i = 0; i < orig_main_body.getNumArguments(); ++i) {
    if (i < nr_platform_args + nr_dim_args) {
      mlir::Type arg_type = orig_main.getArgument(i).getType();
      mlir::RankedTensorType arg_ranked_type =
          arg_type.dyn_cast<mlir::RankedTensorType>();
      if (!arg_ranked_type ||
          !arg_ranked_type.getElementType().dyn_cast<mlir::IntegerType>() ||
          !arg_ranked_type.getShape().empty()) {
        std::string argument_type =
            (i < nr_platform_args) ? "platform index" : "dimension";
        return absl::InvalidArgumentError(absl::StrCat(
            "Module argument at index ", i,
            " should be a 0-dimensional integer-tensor ", argument_type,
            " argument but has type ", mlir::debugString(arg_type)));
      }
      if (i < nr_platform_args) {
        if (arg_ranked_type.getElementTypeBitWidth() != 32) {
          return absl::InvalidArgumentError(
              absl::StrCat("Module argument at index ", i,
                           " should be a 0-dimensional 32-bit integer-tensor"
                           " platform index argument but has type ",
                           mlir::debugString(arg_type)));
        }
        call_args[i] = op_builder.create<mlir::stablehlo::ConstantOp>(
            block_args[0].getLoc(),
            op_builder.getI32IntegerAttr(platform_index_));
      } else {
        TF_ASSIGN_OR_RETURN(
            call_args[i],
            ComputeDimensionValue(
                version_, dim_args_spec_[i - nr_platform_args], block_args,
                op_builder, orig_main.getArgument(i).getType()));
      }
    } else {
      call_args[i] =
          new_main_block->getArgument(i - nr_platform_args - nr_dim_args);
    }
  }
  mlir::func::CallOp call_op = op_builder.create<mlir::func::CallOp>(
      loc, orig_main.getResultTypes(), orig_main.getSymName(), call_args);
  op_builder.create<mlir::func::ReturnOp>(loc, call_op.getResults());

  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.after_add_main_wrapper", *module_);
  }

  return tsl::OkStatus();
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
    std::vector<std::string> dim_args_spec,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, std::string loading_platform,
    int num_invocation_args, bool main_has_token_input_output) {
  context_ = context;
  version_ = version;
  dim_args_spec_ = std::move(dim_args_spec);

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
          << ", dim_args_spec = [" << absl::StrJoin(dim_args_spec_, ", ")
          << "], disabled_checks = [" << absl::StrJoin(disabled_checks, ", ")
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

  if (version >= kVersionStartSupportCallTFGraph && !dim_args_spec_.empty()) {
    return absl::InvalidArgumentError(
        "dim_args_spec not supported in this version");
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

  if (!dim_args_spec_.empty() || platform_index_ >= 0) {
    TF_RETURN_IF_ERROR(AddMainWrapper());
    main_ = module_->lookupSymbol<mlir::func::FuncOp>("main");
  }

  mlir::Block &main_body = main_.front();
  int nr_platform_args = (platform_index_ >= 0 ? 1 : 0);
  int nr_dim_args = dim_args_spec_.size();
  int nr_token_arguments = main_has_token_input_output ? 1 : 0;
  if (num_invocation_args != main_body.getNumArguments() - nr_token_arguments) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments passed to XlaCallModule: ",
        num_invocation_args, ". The module takes ",
        main_body.getNumArguments() + nr_platform_args + nr_dim_args +
            nr_token_arguments,
        " arguments of which ", nr_platform_args, " platform index arguments, ",
        nr_dim_args, " dimension arguments and ", nr_token_arguments,
        " token arguments. It must be called with ",
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
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeSparseChloToLinalgPass());
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
