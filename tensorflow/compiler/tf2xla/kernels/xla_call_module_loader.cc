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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
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
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/regexp.h"

namespace tensorflow {

namespace {

// When adding a new version, write when it was added. Also change the default
// version in the constructor in xla.py.
// Version 1 used MHLO & CHLO, not supported anymore.
// Version 2 supports StableHLO & CHLO. From 10/2022.
const int VERSION_START_STABLE_HLO = 2;
// Version 3 supports platform checking and multiple platforms. From 02/2023.
const int VERSION_START_PLATFORMS = 3;
// Version 4 supports StableHLO with compatibility guarantees.
// Used from 03/2023.
const int VERSION_START_STABLE_HLO_COMPATIBILITY = 4;
const int VERSION_MINIMUM_SUPPORTED = VERSION_START_STABLE_HLO;
const int VERSION_MAXIMUM_SUPPORTED = VERSION_START_STABLE_HLO_COMPATIBILITY;

// Computes a dimension value from the dim_arg specification.
// The specification is of the form "<arg_idx>.<arg_axis_idx>".
tsl::StatusOr<mlir::Value> ComputeDimensionValue(
    int version, std::string dim_arg_spec, std::vector<mlir::Value> arguments,
    mlir::OpBuilder op_builder, mlir::Type dim_arg_type) {
  static const LazyRE2 dim_arg_spec_re = {R"((\d+).(\d+))"};
  int arg_idx, arg_axis_idx;
  if (!RE2::FullMatch(dim_arg_spec, *dim_arg_spec_re, &arg_idx,
                      &arg_axis_idx)) {
    return tsl::errors::InvalidArgument("Syntax error in dim_args_spec '",
                                        dim_arg_spec, "'");
  }
  if (arg_idx < 0 || arg_idx >= arguments.size()) {
    return tsl::errors::InvalidArgument(
        "Invalid argument index ", arg_idx,
        " when the number of non-dimension arguments is ", arguments.size(),
        " in dim_arg_spec '", dim_arg_spec, "'");
  }
  mlir::RankedTensorType arg_type =
      arguments[arg_idx].getType().dyn_cast<mlir::RankedTensorType>();
  if (!arg_type) {
    return tsl::errors::InvalidArgument(
        "Argument ", arg_idx, " referenced in dim_arg_spec '", dim_arg_spec,
        "' does not have a RankedTensorType");
  }
  if (arg_axis_idx < 0 || arg_axis_idx >= arg_type.getShape().size()) {
    return tsl::errors::InvalidArgument(
        "Invalid axis index ", arg_axis_idx,
        " when the rank of non-dimension argument ", arg_idx, " is ",
        arg_type.getShape().size(), " in dim_arg_spec '", dim_arg_spec, "'");
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
    int version, std::string module_str, std::vector<std::string> dim_args_spec,
    int platform_index) {
  if (version < VERSION_MINIMUM_SUPPORTED) {
    return tsl::errors::InvalidArgument(
        "XlaCallModuleOp with version ", version,
        " is not supported anymore. Must be >= ", VERSION_MINIMUM_SUPPORTED);
  }
  if (version > VERSION_MAXIMUM_SUPPORTED) {
    return tsl::errors::InvalidArgument(
        "XlaCallModuleOp with version ", version,
        " is not supported by this build. Must be <= ",
        VERSION_MAXIMUM_SUPPORTED);
  }

  if (version < VERSION_START_PLATFORMS) {
    platform_index = -1;
  }

  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadAndPreprocessModule(
      version, std::move(module_str), std::move(dim_args_spec),
      platform_index));
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
    return tsl::errors::InvalidArgument("Cannot find 'main' in module");
  }
  int nr_platform_args = 0;
  if (platform_index_ >= 0) {
    nr_platform_args = 1;
  }
  if (orig_main.getNumArguments() <= nr_platform_args + nr_dim_args) {
    return tsl::errors::InvalidArgument(
        "The module should have ", nr_platform_args,
        " platform index arguments and ", nr_dim_args,
        " dimension arguments, but it ", "has only ",
        orig_main.getNumArguments(), " total arguments");
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
        return tsl::errors::InvalidArgument(
            "Module argument at index ", i,
            " should be a 0-dimensional integer-tensor ", argument_type,
            " argument but has type ", mlir::debugString(arg_type));
      }
      if (i < nr_platform_args) {
        if (arg_ranked_type.getElementTypeBitWidth() != 32) {
          return tsl::errors::InvalidArgument(
              "Module argument at index ", i,
              " should be a 0-dimensional 32-bit integer-tensor"
              " platform index argument but has type ",
              mlir::debugString(arg_type));
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
  VLOG(3) << "XlaCallModule module with wrapper: "
          << mlir::debugString(*module_);

  return tsl::OkStatus();
}

tsl::Status XlaCallModuleLoader::RefineDynamicShapes(
    absl::Span<const xla::Shape> input_shapes) {
  // Locate the (wrapped) 'main' function.
  // This is the convention used by MlirToXlaComputation.
  mlir::func::FuncOp main = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return tsl::errors::InvalidArgument("Cannot find 'main' in module");
  }
  mlir::Block &main_body = main.front();
  int nr_platform_args = (platform_index_ >= 0 ? 1 : 0);
  int nr_dim_args = dim_args_spec_.size();
  int non_dimension_arguments = input_shapes.size();
  if (non_dimension_arguments != main_body.getNumArguments()) {
    return tsl::errors::InvalidArgument(
        "Incorrect number of arguments passed to XlaCallModule: ",
        non_dimension_arguments, ". The module takes ",
        main_body.getNumArguments() + nr_platform_args + nr_dim_args,
        " arguments of which ", nr_platform_args,
        " platform index arguments and ", nr_dim_args,
        " dimension arguments. It must be called with ",
        main_body.getNumArguments(), " arguments.");
  }

  mlir::Builder builder(module_->getContext());
  std::vector<mlir::Type> static_array_input_types(non_dimension_arguments);
  for (int i = 0, end = non_dimension_arguments; i < end; ++i) {
    const xla::Shape &xla_shape = input_shapes[i];
    std::vector<int64_t> xla_dimensions(xla_shape.dimensions().begin(),
                                        xla_shape.dimensions().end());
    TF_ASSIGN_OR_RETURN(
        mlir::Type element_type,
        ConvertPrimitiveTypeToMLIRType(xla_shape.element_type(), builder));
    mlir::Type type = mlir::RankedTensorType::get(xla_dimensions, element_type);
    // TODO(burmako): This fails with an obscure compilation error.
    // TF_ASSIGN_OR_RETURN(
    //     mlir::Type type,
    //     ConvertShapeToType<mlir::RankedTensorType>(xla_shape, builder));
    VLOG(3) << "XlaCallModule static array input type #" << i << ": "
            << mlir::debugString(type);
    static_array_input_types[i] = type;
  }

  // Refine 'main' argument types to use static input types instead.
  // This will only change the argument types and will not propagate the
  // additional type information further. For that, we'll need to run
  // shape refinement as explained below.
  // Before refining the argument types it is useful to run the inliner to
  // remove calls that may be called with the input arguments.
  mlir::PassManager pm_inline(module_->getContext());
  pm_inline.addPass(mlir::createInlinerPass());
  if (!mlir::succeeded(pm_inline.run(*module_))) {
    return tsl::errors::InvalidArgument("Module inlining failed");
  }
  VLOG(3) << "XlaCallModule module after inlining: "
          << mlir::debugString(*module_);

  auto static_array_output_types = llvm::to_vector(main.getResultTypes());
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    auto arg = main_body.getArgument(i);
    arg.setType(static_array_input_types[i]);
    // If the argument is used by `func.return`, then we also need to
    // update function result types. It's not great that we need this hack,
    // but in the future when we have stablehlo.func, stablehlo.return, etc,
    // this will not be needed.
    // TODO(burmako): Once https://github.com/openxla/stablehlo/issues/425 is
    // fixed, clean this up.
    for (mlir::OpOperand &use : arg.getUses()) {
      if (auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(use.getOwner())) {
        static_array_output_types[use.getOperandNumber()] = arg.getType();
      }
    }
  }
  main.setType(builder.getFunctionType(static_array_input_types,
                                       static_array_output_types));

  // Verify the module before running passes on it.
  // If the module doesn't pass verification, all sorts of weirdness might
  // happen if we run the pass manager.
  if (failed(verify(*module_))) {
    VLOG(3) << "XlaCallModule module with verification failed: "
            << mlir::debugString(*module_);
    return tsl::errors::InvalidArgument("Module verification failed");
  }
  mlir::PassManager pm(module_->getContext());
  if (VLOG_IS_ON(3)) {
    auto print_before = [](mlir::Pass *, mlir::Operation *) { return true; };
    auto print_after = [](mlir::Pass *, mlir::Operation *) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false);
  }
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  if (!mlir::succeeded(pm.run(*module_))) {
    return tsl::errors::InvalidArgument("Module shape refinement failed");
  }

  VLOG(3) << "XlaCallModule module with refined shapes: "
          << mlir::debugString(*module_);
  return tsl::OkStatus();
}

tsl::Status XlaCallModuleLoader::LoadAndPreprocessModule(
    int version, std::string module_str, std::vector<std::string> dim_args_spec,
    int platform_index) {
  version_ = version;
  dim_args_spec_ = std::move(dim_args_spec);
  platform_index_ = platform_index;

  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_.loadDialect<mlir::func::FuncDialect>();
  context_.loadDialect<mlir::stablehlo::StablehloDialect>();
  context_.loadDialect<mlir::mhlo::MhloDialect>();
  context_.loadDialect<mlir::chlo::ChloDialect>();
  context_.loadDialect<mlir::vhlo::VhloDialect>();
  // Parses both IR text and bytecode.
  if (version >= VERSION_START_STABLE_HLO_COMPATIBILITY) {
    module_ =
        mlir::stablehlo::deserializePortableArtifact(module_str, &context_);
  } else {
    module_ = mlir::parseSourceString<mlir::ModuleOp>(module_str, &context_);
  }

  if (!module_) {
    return tsl::errors::InvalidArgument("Cannot deserialize computation");
  }
  VLOG(3) << "Parsed serialized module (version " << version
          << ", platform_index = " << platform_index_ << ", dim_args_spec = ["
          << absl::StrJoin(dim_args_spec_, ", ") << "])\n"
          << mlir::debugString(*module_);

  if (failed(module_->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module_->dump();
    return tsl::errors::InvalidArgument("Error verifying module");
  }
  mlir::func::FuncOp main = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return tsl::errors::InvalidArgument("Cannot find 'main' in module");
  }

  if (!dim_args_spec_.empty() || platform_index_ >= 0) {
    TF_RETURN_IF_ERROR(AddMainWrapper());
    main = module_->lookupSymbol<mlir::func::FuncOp>("main");
  }
  nr_outputs_ = main.getNumResults();
  return tsl::OkStatus();
}

tsl::Status XlaCallModuleLoader::ValidateModule() {
  bool moduleHasUnsupportedDialects = false;
  bool moduleHasDynamicShapes = false;

  module_->walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func and StableHLO dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>(
            op->getDialect())) {
      moduleHasUnsupportedDialects = true;
      VLOG(3) << "Operation has unsupported dialects: "
              << mlir::debugString(op);
    }

    // It's sufficient to only check results because operands either come from
    // results or from block arguments which are checked below.
    auto hasDynamicShape = [](mlir::Value value) {
      auto shaped_type = value.getType().dyn_cast<mlir::ShapedType>();
      return shaped_type ? !shaped_type.hasStaticShape() : false;
    };
    bool opHasDynamicShapes = false;
    opHasDynamicShapes |= llvm::any_of(op->getResults(), hasDynamicShape);
    for (mlir::Region &region : op->getRegions()) {
      opHasDynamicShapes |=
          llvm::any_of(region.getArguments(), hasDynamicShape);
    }
    if (opHasDynamicShapes) {
      moduleHasDynamicShapes = true;
      VLOG(3) << "Operation has dynamic shapes: " << mlir::debugString(op);
    }
  });

  if (moduleHasUnsupportedDialects)
    return tsl::errors::InvalidArgument("Module has unsupported dialects");
  if (moduleHasDynamicShapes)
    return tsl::errors::InvalidArgument("Module has dynamic shapes");
  return tsl::OkStatus();
}

tsl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(
      MlirToXlaComputation(*module_, xla_computation, false, false));
  return xla_computation;
}

}  // namespace tensorflow
