/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/tsl/platform/regexp.h"

namespace tensorflow {
namespace {

// Version 1 uses MHLO, starting with version 2 use StableHLO.
const int VERSION_START_STABLE_HLO = 2;
// Version 3 supports platform checking and multiple platforms
const int VERSION_START_PLATFORMS = 3;

// Computes a dimension value from the dim_arg specification.
// The specification is of the form "<arg_idx>.<arg_axis_idx>".
StatusOr<mlir::Value> ComputeDimensionValue(int version, string dim_arg_spec,
                                            std::vector<mlir::Value> arguments,
                                            mlir::OpBuilder op_builder,
                                            mlir::Type dim_arg_type) {
  static const LazyRE2 dim_arg_spec_re = {R"((\d+).(\d+))"};
  int arg_idx, arg_axis_idx;
  if (!RE2::FullMatch(dim_arg_spec, *dim_arg_spec_re, &arg_idx,
                      &arg_axis_idx)) {
    return errors::InvalidArgument("Syntax error in dim_args_spec '",
                                   dim_arg_spec, "'");
  }
  if (arg_idx < 0 || arg_idx >= arguments.size()) {
    return errors::InvalidArgument(
        "Invalid argument index ", arg_idx,
        " when the number of non-dimension arguments is ", arguments.size(),
        " in dim_arg_spec '", dim_arg_spec, "'");
  }
  mlir::RankedTensorType arg_type =
      arguments[arg_idx].getType().dyn_cast<mlir::RankedTensorType>();
  if (!arg_type) {
    return errors::InvalidArgument(
        "Argument ", arg_idx, " referenced in dim_arg_spec '", dim_arg_spec,
        "' does not have a RankedTensorType");
  }
  if (arg_axis_idx < 0 || arg_axis_idx >= arg_type.getShape().size()) {
    return errors::InvalidArgument("Invalid axis index ", arg_axis_idx,
                                   " when the rank of non-dimension argument ",
                                   arg_idx, " is ", arg_type.getShape().size(),
                                   " in dim_arg_spec '", dim_arg_spec, "'");
  }
  mlir::Value val;
  mlir::Type get_dim_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  if (version >= VERSION_START_STABLE_HLO) {
    val = op_builder.create<mlir::stablehlo::GetDimensionSizeOp>(
        arguments[arg_idx].getLoc(), get_dim_type, arguments[arg_idx],
        op_builder.getI64IntegerAttr(arg_axis_idx));
    if (dim_arg_type != get_dim_type) {
      val = op_builder.create<mlir::stablehlo::ConvertOp>(
          arguments[arg_idx].getLoc(), dim_arg_type, val);
    }
  } else {
    val = op_builder.create<mlir::mhlo::GetDimensionSizeOp>(
        arguments[arg_idx].getLoc(), get_dim_type, arguments[arg_idx],
        op_builder.getI64IntegerAttr(arg_axis_idx));
    if (dim_arg_type != get_dim_type) {
      val = op_builder.create<mlir::mhlo::ConvertOp>(
          arguments[arg_idx].getLoc(), dim_arg_type, val);
    }
  }
  return val;
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
Status AddMainWrapper(int version, mlir::ModuleOp module, int platform_index,
                      std::vector<string> dim_args_spec) {
  int nr_dim_args = dim_args_spec.size();
  // Locate the 'main' function.
  // This is the convention used by MlirToXlaComputation.
  mlir::func::FuncOp orig_main =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!orig_main) {
    return errors::InvalidArgument("Cannot find 'main' in module");
  }
  int nr_platform_args = 0;
  if (platform_index >= 0) {
    nr_platform_args = 1;
  }
  if (orig_main.getNumArguments() <= nr_platform_args + nr_dim_args) {
    return errors::InvalidArgument("The module should have ", nr_platform_args,
                                   " platform index arguments and ",
                                   nr_dim_args, " dimension arguments, but it ",
                                   "has only ", orig_main.getNumArguments(),
                                   " total arguments");
  }
  mlir::Block &orig_main_body = orig_main.front();

  mlir::SymbolTable::setSymbolVisibility(
      orig_main, mlir::SymbolTable::Visibility::Private);
  mlir::OpBuilder op_builder(module.getBodyRegion());
  orig_main.setName(op_builder.getStringAttr("_wrapped_main"));
  mlir::Location loc = module.getLoc();
  std::vector<mlir::Type> new_main_arg_types(
      orig_main.getArgumentTypes().begin() + nr_platform_args + nr_dim_args,
      orig_main.getArgumentTypes().end());
  mlir::func::FuncOp new_main = op_builder.create<mlir::func::FuncOp>(
      loc, "main",
      mlir::FunctionType::get(module.getContext(),
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
        string argument_type =
            (i < nr_platform_args) ? "platform index" : "dimension";
        return errors::InvalidArgument(
            "Module argument at index ", i,
            " should be a 0-dimensional integer-tensor ", argument_type,
            " argument but has type ", debugString(arg_type));
      }
      if (i < nr_platform_args) {
        if (arg_ranked_type.getElementTypeBitWidth() != 32) {
          return errors::InvalidArgument(
              "Module argument at index ", i,
              " should be a 0-dimensional 32-bit integer-tensor"
              " platform index argument but has type ",
              debugString(arg_type));
        }
        call_args[i] = op_builder.create<mlir::stablehlo::ConstantOp>(
            block_args[0].getLoc(),
            op_builder.getI32IntegerAttr(platform_index));
      } else {
        TF_ASSIGN_OR_RETURN(
            call_args[i],
            ComputeDimensionValue(version, dim_args_spec[i - nr_platform_args],
                                  block_args, op_builder,
                                  orig_main.getArgument(i).getType()));
      }
    } else {
      call_args[i] =
          new_main_block->getArgument(i - nr_platform_args - nr_dim_args);
    }
  }
  mlir::func::CallOp call_op = op_builder.create<mlir::func::CallOp>(
      loc, orig_main.getResultTypes(), orig_main.getSymName(), call_args);
  op_builder.create<mlir::func::ReturnOp>(loc, call_op.getResults());
  VLOG(3) << "XlaCallModule module with wrapper: " << debugString(module);

  mlir::PassManager pm(module.getContext());
  // Inliner will merge main and _wrapped_main, making subsequent passes
  // like constant propagation and shape refinement work better.
  pm.addPass(mlir::createInlinerPass());
  if (!mlir::succeeded(pm.run(module))) {
    return errors::InvalidArgument("Module inlining failed");
  }
  VLOG(3) << "XlaCallModule module with inlined wrapper: "
          << debugString(module);

  return OkStatus();
}

// Refines the dynamic module arguments based on the static argument shapes.
// This assumes that the module has a "main" function without dimension args,
// but possibly with dynamic shapes. We read the static shapes of the inputs,
// then set them as the types of the function parameters, and run StableHLO
// shape refinement to specialize all dynamic shapes in the StableHLO program
// to static shapes.
Status RefineDynamicShapes(XlaOpKernelContext *ctx,
                           mlir::OwningOpRef<mlir::ModuleOp> *module,
                           int nr_platform_args, int nr_dim_args) {
  // Locate the (wrapped) 'main' function.
  // This is the convention used by MlirToXlaComputation.
  mlir::func::FuncOp main = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return errors::InvalidArgument("Cannot find 'main' in module");
  }
  mlir::Block &main_body = main.front();
  int non_dimension_arguments = ctx->num_inputs();
  if (non_dimension_arguments != main_body.getNumArguments()) {
    return errors::InvalidArgument(
        "Incorrect number of arguments for XlaCallModule: ",
        non_dimension_arguments, ". The module has ",
        main_body.getNumArguments() + nr_platform_args + nr_dim_args,
        " of which ", nr_platform_args, " platform index arguments and ",
        nr_dim_args, " dimension arguments. It must be called with ",
        main_body.getNumArguments(), " arguments.");
  }

  mlir::Builder builder((*module)->getContext());
  std::vector<mlir::Type> static_array_input_types(non_dimension_arguments);
  for (int i = 0, end = non_dimension_arguments; i < end; ++i) {
    TF_ASSIGN_OR_RETURN(xla::Shape xla_shape, ctx->InputXlaShape(i));
    std::vector<int64_t> xla_dimensions(xla_shape.dimensions().begin(),
                                        xla_shape.dimensions().end());
    TF_ASSIGN_OR_RETURN(
        mlir::Type element_type,
        ConvertPrimitiveTypeToMLIRType(xla_shape.element_type(), builder));
    mlir::Type type = mlir::RankedTensorType::get(xla_dimensions, element_type);
    // TODO(burmako): This fails with an obscure compilation error.
    // OP_REQUIRES_VALUE(
    //     mlir::Type type, ctx,
    //     ConvertShapeToType<mlir::RankedTensorType>(xla_shape, builder));
    VLOG(3) << "XlaCallModule static array input type #" << i << ": "
            << debugString(type);
    static_array_input_types[i] = type;
  }
  // Refine 'main' argument types to use static input types instead.
  // This will only change the argument types and will not propagate the
  // additional type information further. For that, we'll need to run
  // shape refinement as explained below.
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
  if (failed(verify(**module))) {
    VLOG(3) << "XlaCallModule module with verification failed: "
            << debugString(**module);
    return errors::InvalidArgument("Module verification failed");
  }
  mlir::PassManager pm((*module)->getContext());
  if (VLOG_IS_ON(3)) {
    auto print_before = [](mlir::Pass *, mlir::Operation *) { return true; };
    auto print_after = [](mlir::Pass *, mlir::Operation *) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false);
  }
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  if (!mlir::succeeded(pm.run(**module))) {
    return errors::InvalidArgument("Module shape refinement failed");
  }

  VLOG(3) << "XlaCallModule module with refined shapes: "
          << debugString(**module);
  return OkStatus();
}

Status LoadAndPreprocessModule(int version,
                               mlir::OwningOpRef<mlir::ModuleOp> *module,
                               mlir::MLIRContext *context, string module_str,
                               std::vector<string> dim_args_spec,
                               std::vector<string> platforms,
                               int platform_index, bool *has_dynamic_shapes,
                               int *nr_outputs) {
  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();
  context->loadDialect<mlir::mhlo::MhloDialect>();
  context->loadDialect<mlir::chlo::ChloDialect>();
  // Parses both IR text and bytecode.
  *module = mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(module_str),
                                                    context);
  if (!*module) {
    return errors::InvalidArgument("Cannot deserialize computation");
  }
  VLOG(3) << "Parsed serialized module (version " << version
          << ", platforms = [" << absl::StrJoin(platforms, ", ") << "]"
          << ", platform_index = " << platform_index << ", dim_args_spec = ["
          << absl::StrJoin(dim_args_spec, ", ") << "])\n"
          << debugString(**module);

  if (failed((*module)->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    (*module)->dump();
    return errors::InvalidArgument("Error verifying module");
  }
  mlir::func::FuncOp main = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return errors::InvalidArgument("Cannot find 'main' in module");
  }
  *has_dynamic_shapes = false;
  for (const mlir::Type arg_type : main.getArgumentTypes()) {
    mlir::RankedTensorType arg_ranked_type =
        arg_type.dyn_cast<mlir::RankedTensorType>();
    if (!arg_ranked_type) {
      return errors::InvalidArgument("Module main has unranked arguments");
    }
    for (const int64_t arg_dim_size : arg_ranked_type.getShape()) {
      if (arg_dim_size < 0) {
        *has_dynamic_shapes = true;
      }
    }
  }

  if (*has_dynamic_shapes && dim_args_spec.empty()) {
    return errors::InvalidArgument(
        "Module main has dynamic shapes but no dim_args_spec was given");
  }
  if (!dim_args_spec.empty() || platform_index >= 0) {
    TF_RETURN_IF_ERROR(
        AddMainWrapper(version, **module, platform_index, dim_args_spec));
    main = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
  }
  *nr_outputs = main.getNumResults();
  return OkStatus();
}

// Validate that the module represents a statically-shaped StableHLO program,
// otherwise all sorts of weirdness might happen in the HLO exporter which
// is much easier to detect here.
Status ValidateModule(mlir::ModuleOp module) {
  bool moduleHasUnsupportedDialects = false;
  bool moduleHasDynamicShapes = false;

  module.walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func and StableHLO dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>(
            op->getDialect())) {
      moduleHasUnsupportedDialects = true;
      VLOG(3) << "Operation has unsupported dialects: " << debugString(op);
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
      VLOG(3) << "Operation has dynamic shapes: " << debugString(op);
    }
  });

  if (moduleHasUnsupportedDialects)
    return errors::InvalidArgument("Module has unsupported dialects");
  if (moduleHasDynamicShapes)
    return errors::InvalidArgument("Module has dynamic shapes");
  return OkStatus();
}

class XlaCallModuleOp : public XlaOpKernel {
 public:
  explicit XlaCallModuleOp(OpKernelConstruction *ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("version", &version_));
    string module_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("module", &module_str));
    std::vector<PartialTensorShape> expected_output_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Sout", &expected_output_shapes));
    std::vector<DataType> expected_output_dtypes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &expected_output_dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_args_spec", &dim_args_spec_));
    OP_REQUIRES(ctx,
                expected_output_shapes.size() == expected_output_dtypes.size(),
                errors::InvalidArgument("The size of Sout (",
                                        expected_output_shapes.size(),
                                        ") must match the size of Tout (",
                                        expected_output_dtypes.size(), ")"));
    std::vector<string> platforms;
    platform_index_ = -1;
    if (version_ >= VERSION_START_PLATFORMS) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("platforms", &platforms));
      if (!platforms.empty()) {
        std::string current_device_type = ctx->device_type().type_string();
        std::string current_platform = "";
        if (current_device_type == DEVICE_CPU_XLA_JIT) {
          current_platform = "CPU";
        } else if (current_device_type == DEVICE_GPU_XLA_JIT) {
          current_platform = "GPU";
        } else if (current_device_type == DEVICE_TPU_XLA_JIT) {
          current_platform = "TPU";
        } else {
          OP_REQUIRES(ctx, false,
                      errors::Unimplemented("Unexpected device type ",
                                            current_device_type));
        }
        auto found_platform =
            std::find(platforms.begin(), platforms.end(), current_platform);
        OP_REQUIRES(ctx, found_platform != platforms.end(),
                    errors::NotFound(
                        "The current platform ", current_platform,
                        " is not among the platforms required by the module: [",
                        absl::StrJoin(platforms, ", "), "]"));
        // We only use a platform index arguments if we support at least 2
        // platforms.
        if (platforms.size() > 1) {
          platform_index_ = found_platform - platforms.begin();
        }
      }
    }
    OP_REQUIRES_OK(
        ctx, LoadAndPreprocessModule(version_, &module_, &context_, module_str,
                                     dim_args_spec_, platforms, platform_index_,
                                     &has_dynamic_shapes_, &nr_outputs_));
  }

  void Compile(XlaOpKernelContext *ctx) override {
    if (has_dynamic_shapes_) {
      OP_REQUIRES_OK(ctx, RefineDynamicShapes(ctx, &module_,
                                              (platform_index_ >= 0 ? 1 : 0),
                                              dim_args_spec_.size()));
    }
    OP_REQUIRES_OK(ctx, ValidateModule(*module_));

    std::vector<xla::XlaOp> inputs(ctx->num_inputs());
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      inputs[i] = ctx->Input(i);
    }

    xla::XlaComputation xla_computation;
    OP_REQUIRES_OK(
        ctx, MlirToXlaComputation(*module_, xla_computation, false, false));

    if (VLOG_IS_ON(3)) {
      OP_REQUIRES_VALUE(
          const xla::HloModuleConfig module_config, ctx,
          xla::HloModule::CreateModuleConfigFromProto(
              xla_computation.proto(), xla::GetDebugOptionsFromFlags()));
      OP_REQUIRES_VALUE(std::unique_ptr<xla::HloModule> hlo_module, ctx,
                        xla::HloModule::CreateFromProto(xla_computation.proto(),
                                                        module_config));
      xla::HloPrintOptions options;
      options = xla::HloPrintOptions::ShortParsable();
      VLOG(3) << "XlaCallModule converted to HLO module "
              << hlo_module->ToString(options);
    }

    xla::XlaOp output = xla::Call(ctx->builder(), xla_computation, inputs);

    // Check that the resulting computation returns the expected shape
    OP_REQUIRES_VALUE(xla::Shape found_output_shape, ctx,
                      ctx->builder()->GetShape(output));
    VLOG(3) << "XlaCallModule compiled output shape : "
            << xla::ShapeUtil::HumanString(found_output_shape);

    if (nr_outputs_ == 1) {
      ctx->SetOutput(0, output);
    } else {
      for (int i = 0; i < nr_outputs_; ++i) {
        ctx->SetOutput(i, xla::GetTupleElement(output, i));
      }
    }
  }

 private:
  int version_;
  int nr_outputs_;
  std::vector<string> dim_args_spec_;
  bool has_dynamic_shapes_;
  int platform_index_;  // Index in platforms of the current platform, or -1
                        // if module does not take a platform index arg.
  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

REGISTER_XLA_OP(Name("XlaCallModule"), XlaCallModuleOp);
}  // namespace
}  // namespace tensorflow
