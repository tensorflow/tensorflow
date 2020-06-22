/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"

#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

// Parses the MLIR module from the mlir_module_string.
Status ParseMlirModule(llvm::StringRef mlir_module_string,
                       mlir::MLIRContext* mlir_context,
                       mlir::OwningModuleRef* mlir_module) {
  TF_RET_CHECK(!mlir_module_string.empty())
      << "unexpected empty serialized MLIR module string";
  TF_RET_CHECK(mlir_module) << "unexpected null MLIR module pointer";

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system.
  mlir::StatusScopedDiagnosticHandler error_handler(mlir_context);

  // Parse the module.
  *mlir_module = mlir::parseSourceString(mlir_module_string, mlir_context);
  if (!*mlir_module) {
    return error_handler.Combine(
        errors::InvalidArgument("could not parse MLIR module"));
  }

  return Status::OK();
}

// Converts arg_shapes to xla::Shape's and store into xla_input_shapes.
Status GetXlaInputShapes(
    mlir::ModuleOp module, llvm::ArrayRef<TensorShape> arg_shapes,
    bool use_tuple_args,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    std::vector<xla::Shape>* xla_input_shapes) {
  xla_input_shapes->clear();

  mlir::FuncOp main_func = module.lookupSymbol<mlir::FuncOp>("main");
  TF_RET_CHECK(main_func != nullptr) << "No main function found";
  mlir::FunctionType func_type = main_func.getType();

  int num_args = func_type.getNumInputs();
  xla_input_shapes->reserve(num_args);

  std::vector<xla::Shape> individual_arg_shapes;
  individual_arg_shapes.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    individual_arg_shapes.emplace_back();
    xla::Shape& xla_shape = individual_arg_shapes.back();

    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(func_type.getInput(i), &dtype));
    TF_ASSIGN_OR_RETURN(xla_shape,
                        shape_representation_fn(arg_shapes[i], dtype,
                                                /*use_fast_memory=*/false));

    // Rewrite layout with sharding, if sharding is set.
    auto sharding =
        main_func.getArgAttrOfType<mlir::StringAttr>(i, "xla_hlo.sharding");
    if (!sharding) continue;

    absl::optional<xla::HloSharding> arg_sharding;
    xla::OpSharding op_sharding;
    if (!op_sharding.ParseFromString(sharding.getValue().str()))
      return errors::InvalidArgument("failed to parse argument sharding ", i,
                                     " '", sharding.getValue().str(), "'");

    TF_ASSIGN_OR_RETURN(arg_sharding, xla::HloSharding::FromProto(op_sharding));
    TF_RETURN_IF_ERROR(
        RewriteLayoutWithShardedShape(arg_sharding, /*use_fast_memory=*/false,
                                      shape_representation_fn, &xla_shape));
  }
  if (use_tuple_args) {
    xla_input_shapes->push_back(
        xla::ShapeUtil::MakeTupleShape(individual_arg_shapes));
  } else {
    *xla_input_shapes = individual_arg_shapes;
  }
  return Status::OK();
}

// Calculates computation output shape and build OutputDescription for each
// output based on static shapes in MLIR module
Status GetOutputInfo(
    mlir::ModuleOp module,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    xla::Shape* xla_output_shape,
    std::vector<XlaCompiler::OutputDescription>* outputs) {
  auto shape_representation_fn_no_fast_memory =
      [shape_representation_fn](const TensorShape& shape, DataType dtype) {
        return shape_representation_fn(shape, dtype, /*use_fast_memory=*/false);
      };

  mlir::FuncOp main_func = module.lookupSymbol<mlir::FuncOp>("main");
  mlir::FunctionType func_type = main_func.getType();

  outputs->clear();
  outputs->reserve(func_type.getNumResults());

  std::vector<xla::Shape> shapes;
  shapes.reserve(func_type.getNumResults());

  for (mlir::Type type : func_type.getResults()) {
    TF_ASSIGN_OR_RETURN(
        xla::Shape shape,
        xla::TypeToShape(type, shape_representation_fn_no_fast_memory));
    auto tensor_type = type.dyn_cast<mlir::RankedTensorType>();
    shapes.push_back(shape);

    // Construct OutputDescription for result.
    outputs->emplace_back();
    XlaCompiler::OutputDescription& out_desc = outputs->back();
    TF_RETURN_IF_ERROR(ConvertToDataType(tensor_type, &out_desc.type));
    // TODO(ycao): Support constant output.
    out_desc.is_constant = false;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(shape, &out_desc.shape));
    // Input_index is only meaningful for resource output. Since MLIR-based
    // TF-Compiler bridge doesn't support resource output yet. Setting it to
    // meaningless value -1.
    // TODO(ycao): Support resource-type output.
    out_desc.input_index = -1;
    // MLIR-based TF-Compiler bridge doesn't support tensorlist output yet.
    // TODO(ycao): Support tensorlist-type output.
    out_desc.is_tensor_list = false;
  }

  // XLA computation always uses Tuple shape.
  *xla_output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  return Status::OK();
}

// Gets information about how computation updates Tensorflow resources.
// TODO(ycao): Implement logic to compute resource updates when we need to
// support graphs with resource updates in MLIR-based TF compiler bridge.
void GetResourceUpdatesForMlir(
    std::vector<XlaCompiler::ResourceUpdate>* resource_updates) {
  resource_updates->clear();
}

// Creates a vector that maps from the parameters of the XLA computation to
// their original argument positions.
// MLIR-based TF-Compiler bridge doesn't have constant analysis yet, thus no
// inputs are known constants. Therefore, the input mapping between input to
// computation arguments is a trivial in-order 1-1 mapping.
// TODO(ycao): Support computation with compile-time constant, which requires
// non-trivial input mapping as implemented now.
void GetInputMappingForMlir(int num_inputs, std::vector<int>* input_mapping) {
  input_mapping->resize(num_inputs, 0);
  std::iota(input_mapping->begin(), input_mapping->end(), 0);
}

// Refine MLIR types based on new shape information.
Status RefineShapes(llvm::ArrayRef<TensorShape> arg_shapes,
                    mlir::ModuleOp module) {
  auto producer_or = GetTfGraphProducerVersion(module);
  if (!producer_or.ok()) return producer_or.status();
  int64_t producer_version = producer_or.ValueOrDie();

  llvm::SmallVector<int64_t, 16> shape_backing;
  llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> arg_shapes_copy;
  {
    // Convert arg_shapes to a mlir friendly format.
    size_t count = 0;
    for (const TensorShape& shape : arg_shapes) {
      count += shape.dims();
    }
    shape_backing.resize(count);
    arg_shapes_copy.reserve(arg_shapes.size());
    size_t offset = 0;
    for (const TensorShape& shape : arg_shapes) {
      size_t start = offset;
      for (tensorflow::TensorShapeDim dim : shape) {
        shape_backing[offset] = dim.size;
        ++offset;
      }
      if (offset == start) {
        arg_shapes_copy.push_back(llvm::ArrayRef<int64_t>());
      } else {
        arg_shapes_copy.push_back(
            llvm::ArrayRef<int64_t>(&shape_backing[start], offset - start));
      }
    }
  }

  auto main_func = module.lookupSymbol<mlir::FuncOp>("main");

  mlir::StatusScopedDiagnosticHandler error_handler(module.getContext());
  mlir::LogicalResult result = mlir::TF::InferShapeForFunction(
      main_func, arg_shapes_copy, producer_version);

  if (failed(result)) {
    return error_handler.Combine(
        errors::Internal("MLIR Shape refinement failed"));
  }
  return Status::OK();
}

static void RegisterDialects() {
  static bool init_once = []() {
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::TF::TensorFlowDialect>();
    mlir::registerDialect<mlir::shape::ShapeDialect>();
    mlir::registerDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
    mlir::registerDialect<mlir::xla_hlo::XlaHloDialect>();
    return true;
  }();
  (void)init_once;
}

}  //  namespace

Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool return_tuple,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes) {
  mlir::PassManager tf2xla(module_op.getContext());
  tf2xla.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  tf2xla.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
  tf2xla.addPass(mlir::TF::CreateStackOpsDecompositionPass());
  tf2xla.addPass(mlir::TF::CreateTensorArrayOpsDecompositionPass());
  tf2xla.addPass(mlir::TFDevice::CreateDecomposeResourceOpsPass());
  tf2xla.addPass(mlir::TF::CreatePromoteResourcesToArgsPass());
  tf2xla.addPass(mlir::createSymbolDCEPass());
  tf2xla.addPass(mlir::TF::CreateTFShapeInferencePass());
  // LegalizeTFControlFlow encapsulates arguments for control flow operations
  // with a tuple argument which break the assumption of resource lifting
  // inside PromoteResourcesToArgs.
  tf2xla.addPass(mlir::xla_hlo::createLegalizeTFControlFlowPass());

  tf2xla.addNestedPass<mlir::FuncOp>(mlir::xla_hlo::createLegalizeTFPass(true));
  for (auto& target_pass : custom_legalization_passes) {
    tf2xla.addNestedPass<mlir::FuncOp>(std::move(target_pass));
  }
  tf2xla.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  tf2xla.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Leverage tf2xla kernels for ops that didn't get lowered in the previous
  // legalization pass.
  tf2xla.addPass(mlir::xla_hlo::createLegalizeTfWithTf2XlaPass(device_type));
  tf2xla.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());

  // Run shape inference pass to propagate shapes through tensor_cast operations
  // from static to dynamic shapes. This could be generated if the shape
  // inference was originally missing in a TF op but the corresponding HLO op
  // had static shape after lowering.
  tf2xla.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Run LegalizeTFPass again because the previous legalization passes can
  // expose more graph pruning and canonicalization opportunities that are
  // necessary for the second LegalizeTFPass(allow_partial_conversion=false)
  // invocation.
  tf2xla.addNestedPass<mlir::FuncOp>(
      mlir::xla_hlo::createLegalizeTFPass(false));
  // In order to export to XLA, we must sink constants to control flow regions,
  // since XLA uses functional control flow.
  tf2xla.addNestedPass<mlir::FuncOp>(
      mlir::xla_hlo::createSinkConstantsToControlFlowPass());

  if (VLOG_IS_ON(1)) {
    // Print the whole module after each pass which requires disabling
    // multi-threading as well.
    module_op.getContext()->disableMultithreading();
    tf2xla.enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>(
        /*print_module_scope=*/true));
  }

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system. Report a generic error if pass manager failed
  // without emitting a diagnostic.
  mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());

  if (failed(tf2xla.run(module_op))) {
    return error_handler.Combine(
        errors::Internal("MLIR TF to XLA legalization failed"));
  }

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_legalize_hlo", module_op);

  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(mlir::ConvertMlirHloToHlo(module_op, &hlo_proto,
                                               use_tuple_args, return_tuple,
                                               shape_representation_fn));
  *xla_computation = xla::XlaComputation(hlo_proto.hlo_module());
  return Status::OK();
}

static Status CompileMlirToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes) {
  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_before", module_op);

  // Use arg_shapes to improve the mlir type information of `main` in module_op.
  TF_RETURN_IF_ERROR(RefineShapes(arg_shapes, module_op));

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_shape_refiner", module_op);

  if (!shape_representation_fn)
    shape_representation_fn = IdentityShapeRepresentationFn();

  // Convert MLIR module to XLA HLO proto contained in XlaComputation.
  compilation_result->computation = std::make_shared<xla::XlaComputation>();
  TF_RETURN_IF_ERROR(ConvertMLIRToXlaComputation(
      module_op, device_type, compilation_result->computation.get(),
      use_tuple_args,
      /*return_tuple=*/true, shape_representation_fn,
      std::move(custom_legalization_passes)));

  // Construct mapping from XlaComputation's arg to input edges of execute
  // node.
  GetInputMappingForMlir(arg_shapes.size(), &compilation_result->input_mapping);

  // Compute all input shapes.
  TF_RETURN_IF_ERROR(GetXlaInputShapes(module_op, arg_shapes, use_tuple_args,
                                       shape_representation_fn,
                                       &compilation_result->xla_input_shapes));

  // Compute all output descriptions.
  TF_RETURN_IF_ERROR(GetOutputInfo(module_op, shape_representation_fn,
                                   &compilation_result->xla_output_shape,
                                   &compilation_result->outputs));

  // Compute what resource variables need to be updated after XlaComputation's
  // execution.
  GetResourceUpdatesForMlir(&compilation_result->resource_updates);

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_after", module_op);

  return Status::OK();
}

Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes) {
  RegisterDialects();
  mlir::MLIRContext mlir_context;
  mlir::OwningModuleRef mlir_module;

  TF_RETURN_IF_ERROR(
      ParseMlirModule(mlir_module_string, &mlir_context, &mlir_module));
  return CompileMlirToXlaHlo(mlir_module.get(), arg_shapes, device_type,
                             use_tuple_args, shape_representation_fn,
                             compilation_result,
                             std::move(custom_legalization_passes));
}

// Rewrites the given module with specified args. For each of the constant args,
// it gets inlined in the "main' function and the corresponding argument is
// removed from the signature.
// Returns the original indices for the other arguments on success.
static StatusOr<std::vector<int>> RewriteWithArgs(
    mlir::ModuleOp module, llvm::ArrayRef<const XlaCompiler::Argument> args) {
  mlir::FuncOp main_fn = module.lookupSymbol<mlir::FuncOp>("main");
  std::vector<int> params;

  auto builder = mlir::OpBuilder(main_fn.getBody());
  std::vector<int> args_to_erase;
  for (int idx = 0; idx < args.size(); idx++) {
    const XlaCompiler::Argument& xla_arg = args[idx];
    mlir::BlockArgument mlir_arg = main_fn.getArgument(idx);
    if (xla_arg.kind != XlaCompiler::Argument::kConstant) {
      params.push_back(idx);
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto value_attr,
                        ConvertTensor(xla_arg.constant_value, &builder));
    // TODO(hinsu): Use the actual location of the constant.
    auto constant = builder.create<mlir::TF::ConstOp>(
        mlir::UnknownLoc::get(module.getContext()), value_attr);
    mlir_arg.replaceAllUsesWith(constant);
    args_to_erase.push_back(idx);
  }

  for (int idx : llvm::reverse(args_to_erase)) main_fn.eraseArgument(idx);
  return params;
}

Status CompileGraphToXlaHlo(
    const Graph& graph, llvm::ArrayRef<const XlaCompiler::Argument> args,
    llvm::StringRef device_type, bool use_tuple_args,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes) {
  RegisterDialects();

  mlir::MLIRContext context;
  GraphImportConfig config;
  config.graph_as_function = true;
  auto module_or =
      ConvertGraphToMlir(graph, debug_info, flib_def, config, &context);
  if (!module_or.ok()) return module_or.status();

  mlir::ModuleOp module = module_or.ValueOrDie().get();
  TF_ASSIGN_OR_RETURN(std::vector<int> remaining_params,
                      RewriteWithArgs(module, {args.data(), args.size()}));
  llvm::SmallVector<TensorShape, 4> arg_shapes;
  arg_shapes.reserve(args.size());
  for (unsigned idx : remaining_params)
    arg_shapes.push_back(absl::get<TensorShape>(args[idx].shape));

  auto status = CompileMlirToXlaHlo(
      module, arg_shapes, device_type, use_tuple_args, shape_representation_fn,
      compilation_result, std::move(custom_legalization_passes));
  compilation_result->input_mapping = remaining_params;
  return status;
}

}  // namespace tensorflow
