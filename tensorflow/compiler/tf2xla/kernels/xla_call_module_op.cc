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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

absl::Status ImportShardingsAndInlineMeshes(mlir::ModuleOp module) {
  mlir::PassManager sdy_roundtrip(module->getContext());
  sdy_roundtrip.addPass(xla::sdy::createImportShardingsPass(
      /*allowPropagationToArgs=*/{}, /*allowPropagationToResults=*/{},
      /*inlineMesh=*/true));

  tsl::StatusScopedDiagnosticHandler diagnosticHandler(module->getContext());
  absl::Status status =
      diagnosticHandler.consumeStatus(sdy_roundtrip.run(module));
  if (status.ok() && VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.imported_tf_func_after_shardy_import",
                     module);
  }
  return status;
}

// Imports the given `XlaComputation` into StableHLO functions the MLIR module.
// Returns the MLIR function in the imported module that represents the entry
// function of the imported computation.
absl::StatusOr<mlir::func::FuncOp> ImportXlaComputation(
    mlir::SymbolTableCollection& symbol_table_collection, mlir::ModuleOp module,
    const xla::XlaComputation& computation, bool use_shardy_partitioner) {
  mlir::MLIRContext *context = module.getContext();
  mlir::SymbolTable &symbol_table =
      symbol_table_collection.getSymbolTable(module);

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> imported,
      xla::ConvertHloToStablehlo(*context, &computation.proto()));
  if (VLOG_IS_ON(5)) {
    DumpMlirOpToFile("xla_call_module.imported_tf_func", *imported);
  }

  if (use_shardy_partitioner) {
    // Shardings for the tf function calls are not added in
    // XlaCompiler::CompileGraph. Therefore we need to add shardy shardings
    // separately here.
    TF_RETURN_IF_ERROR(ImportShardingsAndInlineMeshes(imported.get()));
  }

  // Rename all functions beforehand in order to avoid conflicts.
  mlir::StringAttr main_func_name;
  for (auto func : imported->getOps<mlir::func::FuncOp>()) {
    mlir::StringAttr name = func.getSymNameAttr();
    mlir::StringAttr new_name = name;
    for (int i = 0; symbol_table.lookup(new_name) != nullptr; ++i) {
      new_name = mlir::StringAttr::get(
          context, absl::StrCat(absl::string_view(name.getValue()), i));
    }
    if (new_name != name) {
      if (failed(mlir::SymbolTable::replaceAllSymbolUses(func, new_name,
                                                         *imported))) {
        return absl::InternalError(
            absl::StrCat("Failed to replace all symbol uses of function '",
                         absl::string_view(func.getName()), "'"));
      }
      func.setSymNameAttr(new_name);
    }
    if (name.getValue() == "main") {
      main_func_name = new_name;
    }
  }
  if (!main_func_name) {
    return absl::InternalError(
        "HLO module lowered from TF function is missing a main function");
  }

  mlir::func::FuncOp main_func;
  for (auto func : imported->getOps<mlir::func::FuncOp>()) {
    auto cloned = func.clone();
    cloned.setPrivate();
    symbol_table.insert(cloned);
    if (func.getSymNameAttr() == main_func_name) {
      main_func = cloned;
    }
  }

  return main_func;
}

class XlaCallModuleOp : public XlaOpKernel {
 public:
  explicit XlaCallModuleOp(OpKernelConstruction *ctx) : XlaOpKernel(ctx) {
    int version;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("version", &version));
    string module_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("module", &module_str));
    std::vector<PartialTensorShape> expected_output_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Sout", &expected_output_shapes));
    std::vector<DataType> expected_output_dtypes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &expected_output_dtypes));
    std::vector<string> dim_args_spec;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_args_spec", &dim_args_spec));
    OP_REQUIRES(ctx, dim_args_spec.empty(),
                absl::UnimplementedError(
                    "dim_args_spec attribute is no longer supported"));
    OP_REQUIRES(ctx,
                expected_output_shapes.size() == expected_output_dtypes.size(),
                absl::InvalidArgumentError(absl::StrCat(
                    "The size of Sout (", expected_output_shapes.size(),
                    ") must match the size of Tout (",
                    expected_output_dtypes.size(), ")")));
    std::vector<string> disabled_checks;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("disabled_checks", &disabled_checks));
    std::vector<string> platforms;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("platforms", &platforms));
    // TODO(necula): change this to OP_REQUIRES_OK when 6 months have passed
    // since we added the function_list and has_token_input_output
    // attributes (May 25, 2023).
    bool main_has_token_input_output = false;
    if (!ctx->GetAttr("has_token_input_output", &main_has_token_input_output)
             .ok()) {
      // Whether the StableHLO module's main function has token input/output as
      // the first argument and the first result.
      // This is used only prior to version 9; afterwards, we just look for
      // tokens among the types of the arguments and results, and we support
      // multiple tokens, not necessarily at the start.
      main_has_token_input_output = false;
    }
    if (!ctx->GetAttr("function_list", &function_list_).ok()) {
      function_list_.clear();
    }

    bool use_shardy_partitioner = false;
    if (!ctx->GetAttr("use_shardy_partitioner", &use_shardy_partitioner).ok()) {
      use_shardy_partitioner = false;
    }
    if (VLOG_IS_ON(3)) {
      VLOG(3) << "Initializing XlaCallModuleOp (version = " << version
              << ", platforms = [" << absl::StrJoin(platforms, ", ")
              << "], has_token_input_output = " << main_has_token_input_output
              << ", use_shardy_partitioner = " << use_shardy_partitioner
              << ", disabled_checks = [" << absl::StrJoin(disabled_checks, ", ")
              << "], "
              << "function_list = ["
              << absl::StrJoin(function_list_, ",",
                               [](std::string *out, NameAttrList x) {
                                 absl::StrAppend(out, x.name());
                               })
              << "])";
    }
    string compilation_device_type = ctx->device_type().type_string();
    compilation_platform_ = "";
    if (compilation_device_type == DEVICE_CPU_XLA_JIT) {
      compilation_platform_ = "CPU";
    } else if (compilation_device_type == DEVICE_GPU_XLA_JIT) {
#if GOOGLE_CUDA
      compilation_platform_ = "CUDA";
#elif TENSORFLOW_USE_ROCM
      compilation_platform_ = "ROCM";
#else
      OP_REQUIRES(ctx, false,
                  absl::UnimplementedError("CUDA or ROCM build required"));
#endif
    } else if (compilation_device_type == DEVICE_TPU_XLA_JIT) {
      compilation_platform_ = "TPU";
    } else {
      OP_REQUIRES(ctx, false,
                  absl::UnimplementedError(absl::StrCat(
                      "Unexpected device type ", compilation_device_type)));
    }
    VLOG(3) << "Initializing XlaCallModuleOp on " << compilation_platform_;
    {
      auto loader = XlaCallModuleLoader::Create(
          &context_, version, module_str, std::move(disabled_checks),
          std::move(platforms),
          /*num_invocation_args=*/ctx->num_inputs(),
          main_has_token_input_output, use_shardy_partitioner);
      OP_REQUIRES_OK(ctx, loader.status());
      loader_ = *std::move(loader);
    }
    OP_REQUIRES_OK(ctx, loader_->ValidateXlaCallModuleInvariants());

    if (!ctx->GetAttr(kXlaTokenInputNodesAttrName, &op_token_input_nodes_)
             .ok()) {
      op_token_input_nodes_.clear();
    }
    if (!ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                      &original_node_name_)
             .ok()) {
      original_node_name_ = name();
    }

    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    context_.appendDialectRegistry(registry);
  }

  void Compile(XlaOpKernelContext *ctx) override {
    XlaCompiler *const compiler = ctx->compiler();
    xla::XlaBuilder *const b = ctx->builder();

    std::vector<xla::Shape> input_shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      auto shape = ctx->InputXlaShape(i);
      OP_REQUIRES_OK(ctx, shape.status());
      input_shapes.push_back(*std::move(shape));
    }
    OP_REQUIRES_OK(ctx, loader_->SetPlatformIndex(compilation_platform_));
    OP_REQUIRES_OK(ctx, loader_->RefineDynamicShapes(input_shapes));
    OP_REQUIRES_OK(ctx, loader_->ValidateStaticShapes());
    // Lowering tf function calls before SdyRoundTripExport which is a part of
    // PrepareStablehloForLowering, as we run ImportShardy pass while lowering
    // tf function calls.
    if (!function_list_.empty()) {
      OP_REQUIRES_OK(ctx, LowerTfFunctionCalls(ctx));
    }
    OP_REQUIRES_OK(ctx, loader_->PrepareStablehloForLowering());

    xla::XlaOp token_input;
    if (!op_token_input_nodes_.empty()) {
      std::vector<xla::XlaOp> token_inputs;
      for (const string &node_name : op_token_input_nodes_) {
        auto token = compiler->GetNodeToken(node_name);
        OP_REQUIRES_OK(ctx, token.status());
        token_inputs.push_back(token.value());
      }
      token_input = xla::AfterAll(b, token_inputs);
    }

    // Set input and output types after removing platform args, but before
    // lowering the XLA computation since that will mutate the module in place
    // to use MHLO types.
    auto input_types = loader_->InputTypes();
    auto output_types = loader_->OutputTypes();

    std::vector<xla::XlaOp> inputs;
    int next_actual_input = 0;
    for (mlir::Type inputType : input_types) {
      if (IsTokenType(inputType)) {
        if (token_input.IsUninitialized()) {
          // Generate a dummy token if the XlaCallModule doesn't take one.
          token_input = xla::CreateToken(b);
        }
        inputs.push_back(token_input);
      } else {
        inputs.push_back(ctx->Input(next_actual_input++));
      }
    }
    auto xla_computation = loader_->ToXlaComputation();
    OP_REQUIRES_OK(ctx, xla_computation.status());

    if (VLOG_IS_ON(3)) {
      OP_REQUIRES_VALUE(
          const xla::HloModuleConfig module_config, ctx,
          xla::HloModule::CreateModuleConfigFromProto(
              xla_computation->proto(), xla::GetDebugOptionsFromFlags()));
      OP_REQUIRES_VALUE(std::unique_ptr<xla::HloModule> hlo_module, ctx,
                        xla::HloModule::CreateFromProto(
                            xla_computation->proto(), module_config));
      xla::HloPrintOptions options;
      options = xla::HloPrintOptions::ShortParsable();
      XLA_VLOG_LINES(3, absl::StrCat("XlaCallModule converted to HLO module ",
                                     hlo_module->ToString(options)));
    }

    xla::XlaOp computation_output = xla::Call(b, *xla_computation, inputs);

    // Check that the resulting computation returns the expected shape
    OP_REQUIRES_VALUE(xla::Shape found_output_shape, ctx,
                      b->GetShape(computation_output));
    VLOG(3) << "XlaCallModule compiled output shape : "
            << xla::ShapeUtil::HumanString(found_output_shape);
    std::vector<xla::XlaOp> computation_outputs;
    if (loader_->NrOutputs() == 1) {
      computation_outputs.push_back(computation_output);
    } else {
      for (int i = 0; i < loader_->NrOutputs(); ++i) {
        computation_outputs.push_back(
            xla::GetTupleElement(computation_output, i));
      }
    }

    // Collect the token outputs and set the non-token outputs
    std::vector<xla::XlaOp> token_outputs;
    int next_actual_output = 0;
    for (auto it : llvm::enumerate(output_types)) {
      int i = it.index();
      mlir::Type output_type = it.value();
      auto shape = b->GetShape(computation_outputs[i]);
      OP_REQUIRES_OK(ctx, shape.status());
      if (IsTokenType(output_type)) {
        OP_REQUIRES(ctx, shape->IsToken(),
                    absl::FailedPreconditionError(absl::StrCat(
                        "Token output at index ", i, " (",
                        mlir::debugString(output_type), ") is not token type: ",
                        xla::ShapeUtil::HumanString(*shape))));
        token_outputs.push_back(computation_outputs[i]);
      } else {
        OP_REQUIRES(ctx, !shape->IsToken(),
                    absl::FailedPreconditionError(absl::StrCat(
                        "Non-token output at index ", i, " (",
                        mlir::debugString(output_type), ") is not token type: ",
                        xla::ShapeUtil::HumanString(*shape))));
        ctx->SetOutput(next_actual_output++, computation_outputs[i]);
      }
    }

    if (!op_token_input_nodes_.empty()) {
      xla::XlaOp token_output = token_input;
      if (!token_outputs.empty()) {
        token_output = xla::AfterAll(b, token_outputs);
      } else {
        if (token_output.IsUninitialized()) {
          token_output = xla::CreateToken(b);
        }
      }
      OP_REQUIRES_OK(ctx,
                     compiler->SetNodeToken(original_node_name_, token_output));
    }
  }

 private:
  // Lowers `stablehlo.CustomCall` ops representing TF function calls into
  // nested XLA computation. The called TF functions are lowered into StableHLO
  // and inserted as function calls in the main module.
  //
  // This is implemented here instead of in xla_call_module_loader.cc in order
  // to prevent cyclic dependency with TF MLIR passes.
  absl::Status LowerTfFunctionCalls(XlaOpKernelContext *ctx) {
    mlir::ModuleOp module = loader_->module();
    mlir::SymbolTableCollection symbol_table_collection;

    llvm::SmallDenseSet<mlir::func::FuncOp> updated_funcs;

    auto lower =
        [&](mlir::stablehlo::CustomCallOp custom_call) -> absl::Status {
      if (custom_call.getCallTargetName() != "tf.call_tf_function") {
        return absl::OkStatus();
      }

      NameAttrList f;
      bool custom_call_has_token_input_output = false;
      {
        auto backend_config = custom_call->getAttrOfType<mlir::DictionaryAttr>(
            "tf.backend_config");
        if (!backend_config) {
          return absl::InternalError(
              "TF function custom call must have 'tf.backend_config' "
              "attribute");
        }

        auto called_index =
            backend_config.getAs<mlir::IntegerAttr>("called_index");
        if (!called_index) {
          return absl::InternalError(
              "TF function custom call must have 'called_index' in the "
              "'tf.backend_config' attribute");
        }

        int index = called_index.getInt();
        if (index < 0 || index >= function_list_.size()) {
          return absl::OutOfRangeError(absl::StrCat(
              "XlaCallModule has function_list of size ", function_list_.size(),
              " but TF function custom call references function #", index));
        }
        f = function_list_[index];

        // Whether the custom call takes a token argument and returns another
        // token. Used to model side effects.
        if (auto attr =
                backend_config.getAs<mlir::BoolAttr>("has_token_input_output");
            attr != nullptr) {
          custom_call_has_token_input_output = attr.getValue();
        }
      }

      // Lower the called TF function into an HLO module.
      std::vector<XlaCompiler::Argument> arguments;
      {
        mlir::TypeRange input_types(custom_call->getOperandTypes());
        if (custom_call_has_token_input_output) {
          if (input_types.empty() || !IsTokenType(input_types.front())) {
            return absl::InvalidArgumentError(absl::StrCat(
                "stablehlo.custom_call with has_token_input_output = true is "
                "expected to take !stablehlo.token as the first argument, but "
                "got ",
                mlir::debugString(custom_call)));
          }
          input_types = input_types.drop_front();
        }
        for (mlir::Type input_type : input_types) {
          XlaCompiler::Argument &argument = arguments.emplace_back();
          argument.kind = XlaCompiler::Argument::kParameter;
          TF_RETURN_IF_ERROR(ConvertToDataType(input_type, &argument.type));
          argument.shape = xla::TypeToShape(input_type);
        }

        mlir::TypeRange result_types(custom_call->getResultTypes());
        if (custom_call_has_token_input_output) {
          if (result_types.empty() || !IsTokenType(result_types.front())) {
            return absl::InvalidArgumentError(absl::StrCat(
                "stablehlo.custom_call with has_token_input_output = true is "
                "expected to return !stablehlo.token as the first result, but "
                "got ",
                mlir::debugString(custom_call)));
          }
        }
      }

      // There are times when TPU prefers tuple arguments, i.e. when there are
      // 2k+ input arguments, but for the most part we want to avoid tuples.
      XlaCompiler::CompileOptions options;
      options.use_tuple_arg = false;
      options.always_return_tuple = false;
      options.is_entry_computation = false;
      // Propagate tokens from XlaCallModule to inner computation.
      options.add_token_input_output = !op_token_input_nodes_.empty();

      XlaCompiler::CompilationResult result;
      TF_RETURN_IF_ERROR(
          ctx->compiler()->CompileFunction(options, f, arguments, &result));

      // Import the lowered HLO module into StableHLO functions in `module`.
      // The main function accepts variadic arguments and returns variadic
      // results.
      TF_ASSIGN_OR_RETURN(
          mlir::func::FuncOp main_func,
          ImportXlaComputation(
              symbol_table_collection, module, *result.computation,
              ctx->compiler()->options().use_shardy_partitioner));

      // Replace the custom call with ops that call the imported main function.
      mlir::OpBuilder builder(custom_call);
      auto loc = custom_call.getLoc();

      // If `has_tuple_input_output` is true, the first argument is a token type
      llvm::SmallVector<mlir::Value> input_args;
      {
        llvm::SmallVector<mlir::Value> args(custom_call->getOperands());
        if (custom_call_has_token_input_output) {
          // Adjust the indexes since custom calls with `has_token_input_output`
          // takes a token as the first argument, but TF2XLA'ed computation
          // expects the token to be the last argument.
          std::rotate(args.begin(), args.begin() + 1, args.end());
        } else if (options.add_token_input_output) {
          // Add a dummy token if the inner computation takes a token but the
          // custom call doesn't have a token argument.
          args.push_back(builder.create<mlir::stablehlo::CreateTokenOp>(loc));
        }

        input_args.reserve(result.input_mapping.size());
        for (int index : result.input_mapping) {
          input_args.push_back(args[index]);
        }
      }

      // Call the lowered function.
      auto call =
          builder.create<mlir::func::CallOp>(loc, main_func, input_args);

      // Unpack the result tuple (`options.always_return_tuple` is true). If
      // `has_tuple_input_output` is true, the first result is a token type.
      {
        llvm::SmallVector<mlir::Value> results(custom_call->getResults());
        if (custom_call_has_token_input_output) {
          // Adjust the indexes since custom calls with `has_token_input_output`
          // returns a token as the first result, but TF2XLA'ed computation
          // returns the token as the last result.
          std::rotate(results.begin(), results.begin() + 1, results.end());

          if (!options.add_token_input_output) {
            // If the custom call returns a token but the inner computation
            // doesn't, replace the token result with a dummy token.
            mlir::Value token = results.back();
            if (!token.use_empty()) {
              token.replaceAllUsesWith(
                  builder.create<mlir::stablehlo::CreateTokenOp>(loc));
            }
            results.pop_back();
          }
        }

        for (const auto &it : llvm::enumerate(results)) {
          if (!it.value().use_empty()) {
            it.value().replaceAllUsesWith(call.getResult(it.index()));
          }
        }
      }

      updated_funcs.insert(call->getParentOfType<mlir::func::FuncOp>());
      custom_call->erase();

      return absl::OkStatus();
    };

    absl::Status status;
    mlir::WalkResult result =
        module->walk([&](mlir::stablehlo::CustomCallOp op) {
          status.Update(lower(op));
          if (!status.ok()) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return status;
    }

    // If the call results are used by `func.return`, then we may need to update
    // function result types.
    for (auto func : updated_funcs) {
      auto ret = llvm::cast<mlir::func::ReturnOp>(
          func.getFunctionBody().front().getTerminator());
      func.setFunctionType(mlir::FunctionType::get(
          &context_, func.getArgumentTypes(), ret.getOperandTypes()));
    }

    if (VLOG_IS_ON(5)) {
      DumpMlirOpToFile("xla_call_module.after_tf_func_call_import", module);
    }
    return absl::OkStatus();
  }

  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};
  std::unique_ptr<XlaCallModuleLoader> loader_;
  std::vector<NameAttrList> function_list_;
  std::string compilation_platform_;

  // Whether the XlaCallModule op has token input/output.
  std::vector<std::string> op_token_input_nodes_;
  std::string original_node_name_;
};

REGISTER_XLA_OP(Name("XlaCallModule"), XlaCallModuleOp);

}  // namespace
}  // namespace tensorflow
