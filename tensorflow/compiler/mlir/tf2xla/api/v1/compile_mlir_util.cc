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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "tensorflow/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_pass_instrumentation.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/lowering_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/layout_util.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {
constexpr absl::string_view kGroupSizeAttrName =
    "tf2xla.collective_info.group_size";
constexpr absl::string_view kGroupKeyAttrName =
    "tf2xla.collective_info.group_key";

// Extracts shape from XlaArgument as TensorShape. If shape is a xla::Shape,
// that is converted to a TensorShape.
absl::StatusOr<TensorShape> GetTensorShapeFromXlaArgument(
    const XlaArgument& arg) {
  if (std::holds_alternative<xla::Shape>(arg.shape)) {
    TensorShape arg_shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(std::get<xla::Shape>(arg.shape), &arg_shape));
    return arg_shape;
  } else {
    return std::get<TensorShape>(arg.shape);
  }
}

Status MaybeRewriteLayoutWithShardedShape(
    mlir::StringAttr sharding,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    xla::Shape* shape) {
  if (!sharding) return absl::OkStatus();

  xla::OpSharding op_sharding;
  if (tensorflow::DecodeShardingAttribute(sharding, op_sharding).failed()) {
    return errors::InvalidArgument("failed to parse sharding '",
                                   sharding.getValue().str(), "'");
  }
  std::optional<xla::HloSharding> hlo_sharding;
  TF_ASSIGN_OR_RETURN(hlo_sharding, xla::HloSharding::FromProto(op_sharding));
  TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
      hlo_sharding, /*use_fast_memory=*/false, shape_determination_fns, shape));
  return absl::OkStatus();
}

// Converts arg_shapes to xla::Shape's and store into xla_input_shapes.
Status GetXlaInputShapes(
    mlir::ModuleOp module, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    std::vector<xla::Shape>* xla_input_shapes) {
  xla_input_shapes->clear();

  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  TF_RET_CHECK(main_func != nullptr) << "No main function found";
  mlir::FunctionType func_type = main_func.getFunctionType();

  int num_args = func_type.getNumInputs();
  xla_input_shapes->reserve(num_args);

  std::vector<xla::Shape> individual_arg_shapes;
  individual_arg_shapes.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    individual_arg_shapes.emplace_back();
    xla::Shape& xla_shape = individual_arg_shapes.back();

    DataType arg_dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(func_type.getInput(i), &arg_dtype));

    auto layout_preference = shape_determination_fns.layout_preference_fn(
        arg_shapes[i].shape, arg_dtype, std::nullopt);
    TF_ASSIGN_OR_RETURN(xla_shape,
                        shape_determination_fns.shape_representation_fn(
                            arg_shapes[i].shape, arg_dtype,
                            /*use_fast_memory=*/false, layout_preference));

    // Rewrite layout with sharding, if sharding is set.
    auto sharding =
        main_func.getArgAttrOfType<mlir::StringAttr>(i, "mhlo.sharding");
    TF_RETURN_IF_ERROR(MaybeRewriteLayoutWithShardedShape(
        sharding, shape_determination_fns, &xla_shape));
  }
  if (use_tuple_args) {
    xla_input_shapes->push_back(
        xla::ShapeUtil::MakeTupleShape(individual_arg_shapes));
  } else {
    *xla_input_shapes = individual_arg_shapes;
  }
  return absl::OkStatus();
}

// Returns a static ranked tensor type corresponding to the given static or
// bounded type by using the bounds as dimension sizes. Returns null if is
// neither.
mlir::RankedTensorType GetBufferType(mlir::Type ty) {
  auto ranked_ty = mlir::dyn_cast_or_null<mlir::RankedTensorType>(ty);
  if (!ranked_ty) return {};

  int64_t rank = ranked_ty.getRank();
  llvm::SmallVector<int64_t, 4> dims = llvm::to_vector<4>(ranked_ty.getShape());
  auto encoding = mlir::dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>(
      ranked_ty.getEncoding());
  if (encoding && !encoding.getBounds().empty()) {
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dims[dim] == mlir::ShapedType::kDynamic) {
        dims[dim] = encoding.getBounds()[dim];
      }
    }
  }
  return GetTypeFromTFTensorShape(dims, ranked_ty.getElementType());
}

// Calculates computation output shape and build OutputDescription for each
// output based on static shapes in MLIR module. If an output is a resource
// write, `resource_updates` is populated instead of `outputs` for that output.
Status GetOutputInfo(
    mlir::ModuleOp module, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    xla::Shape* xla_output_shape, std::vector<XlaOutputDescription>* outputs,
    std::vector<XlaResourceUpdate>* resource_updates) {
  auto shape_representation_fn_no_fast_memory =
      [shape_determination_fns](
          const xla::Shape& xla_shape) -> absl::StatusOr<xla::Shape> {
    TensorShape shape;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, &shape));
    TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                            xla_shape.element_type()));
    auto layout_preference = shape_determination_fns.layout_preference_fn(
        shape, dtype, std::nullopt);
    return shape_determination_fns.shape_representation_fn(
        shape, dtype, /*use_fast_memory=*/false, layout_preference);
  };

  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  mlir::FunctionType func_type = main_func.getFunctionType();

  outputs->clear();
  outputs->reserve(func_type.getNumResults());
  resource_updates->clear();
  resource_updates->reserve(func_type.getNumResults());

  std::vector<xla::Shape> shapes;
  shapes.reserve(func_type.getNumResults());

  llvm::SmallDenseMap<unsigned, unsigned> output_to_input_alias;
  for (unsigned i = 0; i < main_func.getNumArguments(); ++i)
    if (auto aliasing_output = main_func.getArgAttrOfType<mlir::IntegerAttr>(
            i, "tf.aliasing_output"))
      output_to_input_alias[aliasing_output.getInt()] = i;

  auto return_op = main_func.begin()->getTerminator();
  for (const auto& type_and_idx : llvm::enumerate(func_type.getResults())) {
    size_t idx = type_and_idx.index();
    auto result_ty = mlir::cast<mlir::RankedTensorType>(type_and_idx.value());

    // If the result type isn't static, then the owner of the result may be a
    // cast op from a more specific bounded type to an unbounded dynamic type.
    // Use the bounded type to get the buffer size.
    mlir::RankedTensorType buffer_ty = result_ty;
    if (!buffer_ty.hasStaticShape()) {
      mlir::Value return_val = return_op->getOperand(idx);
      if (auto owner = mlir::dyn_cast_or_null<mlir::tensor::CastOp>(
              return_val.getDefiningOp())) {
        // For bounded dynamic type, get a static size by taking bounds as the
        // dimensions. These dimensions are marked as dynamic in xla::Shape
        // below.
        buffer_ty = GetBufferType(owner.getOperand().getType());
        if (!buffer_ty || !buffer_ty.hasStaticShape()) {
          return errors::InvalidArgument(
              "results needs to be static or bounded");
        }
      }
    }

    xla::Shape shape = xla::TypeToShape(buffer_ty);
    if (shape.element_type() == xla::PRIMITIVE_TYPE_INVALID) {
      return errors::InvalidArgument("XLA conversion failed for MLIR type.");
    }
    TF_ASSIGN_OR_RETURN(shape, shape_representation_fn_no_fast_memory(shape));

    if (!result_ty.hasStaticShape()) {
      int64_t rank = result_ty.getRank();
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (result_ty.isDynamicDim(dim)) {
          shape.set_dynamic_dimension(dim, true);
        }
      }
    }

    auto sharding = main_func.getResultAttrOfType<mlir::StringAttr>(
        type_and_idx.index(), "mhlo.sharding");
    TF_RETURN_IF_ERROR(MaybeRewriteLayoutWithShardedShape(
        sharding, shape_determination_fns, &shape));

    auto tensor_type =
        mlir::dyn_cast<mlir::RankedTensorType>(type_and_idx.value());
    shapes.push_back(shape);

    auto it = output_to_input_alias.find(type_and_idx.index());
    if (it != output_to_input_alias.end() && use_resource_updates_for_aliases) {
      // Add resource write.
      resource_updates->emplace_back();
      XlaResourceUpdate& resource_update = resource_updates->back();
      resource_update.input_index = it->getSecond();
      resource_update.modified = true;
      TF_RETURN_IF_ERROR(ConvertToDataType(tensor_type, &resource_update.type));
      TF_RETURN_IF_ERROR(XLAShapeToTensorShape(shape, &resource_update.shape));
      continue;
    }
    // Construct OutputDescription for result.
    outputs->emplace_back();
    XlaOutputDescription& out_desc = outputs->back();
    TF_RETURN_IF_ERROR(ConvertToDataType(tensor_type, &out_desc.type));
    // TODO(ycao): Support constant output.
    out_desc.is_constant = false;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(shape, &out_desc.shape));
    // Input_index is only meaningful for resource output. Setting it to
    // meaningless value -1 for non resource outputs.
    out_desc.input_index =
        it != output_to_input_alias.end() ? it->getSecond() : -1;
    // MLIR-based TF-Compiler bridge doesn't support tensorlist output yet.
    // TODO(ycao): Support tensorlist-type output.
    out_desc.is_tensor_list = false;
  }

  // XLA computation always uses Tuple shape.
  *xla_output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  return absl::OkStatus();
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

static void RegisterDialects(mlir::DialectRegistry& registry) {
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
}

// Checks if functions can be inlined after TF -> HLO legalization. Currently
// TPU's are supported, to follow the behavior of inlining functions via the
// Graph based bridge in the TPUCompile op kernel.
bool CanInlineFunctionsPostLegalization(llvm::StringRef device_type) {
  return device_type == DEVICE_TPU_XLA_JIT;
}

// These passes are grouped together and must run in this specific order.
void AddLegalizationPasses(mlir::OpPassManager& pm, bool legalize_chlo,
                           llvm::StringRef device_type, bool enable_op_fallback,
                           bool lower_to_xla_hlo) {
  if (lower_to_xla_hlo) {
    // Lower TF quant ops and types to MHLO int.
    mlir::quant::stablehlo::AddQuantizationLoweringPasses(pm);

    pm.addPass(mlir::mhlo::createLegalizeTFPass(
        legalize_chlo,
        /*tf2xla_fallback_device_type=*/device_type, enable_op_fallback));
  }

  // This has to run after legalization.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::CreateInfeedsOpsXlaAdjustLayoutPass());

  if (lower_to_xla_hlo) {
    // This has to run after legalization to delete non legal but dead ops.
    // This must run before Shape Inference.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

    // Run shape inference pass to propagate shapes through tensor_cast
    // operations from static to dynamic shapes. This could be generated if the
    // shape inference was originally missing in a TF op but the corresponding
    // HLO op had static shape after lowering. This has to run after
    // canonicalization.
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  }
}

// Builds a pipeline that runs only those passes required to lower TF to HLO.
void CreateMlirLoweringPassesPipeline(mlir::PassManager& pm) {
  applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateLowerQuantizedPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::CreateConvertTFQuantTypesPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addPass(mlir::mhlo::CreateLegalizeTFCollectivePass());
  mlir::quant::stablehlo::AddQuantizationLoweringPasses(pm);
  // Legalize to HLO but force MLIR native only.
  pm.addPass(mlir::mhlo::createLegalizeTFPass(
      /*legalize_chlo=*/true,
      /*tf2xla_fallback_device_type=*/std::nullopt, /*prefer_tf2xla=*/false));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::mhlo::CreateLegalizeTFCommunicationPass());

  auto pass_instrumentors = mlir::GetPassInstrumentors();
  for (const auto& creator : pass_instrumentors) {
    pm.addInstrumentation(creator());
  }
}

// If the module should be dumped then dumps the file and turns on the before
// and after IR printing for the passes.
void MaybeDumpMlirModuleAndPasses(mlir::PassManager& pm,
                                  mlir::ModuleOp module_op,
                                  std::string module_name, std::string tag) {
  if (DEBUG_DATA_DUMPER()->ShouldDump(module_name, kDebugGroupMain) ||
      VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name, kDebugGroupMain, tag),
        module_op, /*dirname=*/"", &pm);
  }

  if (VLOG_IS_ON(2) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name, kDebugGroupBridgePhase2)) {
    // Print the whole module after each pass which requires disabling
    // multi-threading as well.
    module_op.getContext()->disableMultithreading();
    pm.enableIRPrinting(std::make_unique<::tensorflow::DataDumperLoggerConfig>(
        [module_name](const std::string& pass_tag_name, mlir::Operation* op) {
          return DEBUG_DATA_DUMPER()->GetDumpFilename(
              module_name, kDebugGroupBridgePhase2, pass_tag_name);
        },
        /*pass_prefix=*/"",
        /*print_module_scope=*/true));
  }
}

// Runs the MLIR pipeline and if successful dumps the results if enabled.
absl::Status RunMlirPipelineAndMaybeDumpResults(mlir::PassManager& pm,
                                                mlir::ModuleOp module_op,
                                                std::string module_name = "") {
  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system. Report a generic error if pass manager failed
  // without emitting a diagnostic.
  mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());

  if (failed(pm.run(module_op))) {
    auto status = absl::InvalidArgumentError("TF to XLA legalization failed: ");
    tensorflow::OkOrSetErrorCounterPayload(
        tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_2,
        status);
    return error_handler.Combine(status);
  }

  if (DEBUG_DATA_DUMPER()->ShouldDump(module_name, kDebugGroupMain) ||
      VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name, kDebugGroupMain,
                                             "legalize_hlo_after"),
        module_op, /*dirname=*/"", &pm);
  }

  Status status = error_handler.ConsumeStatus();
  tensorflow::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_2,
      status);
  return status;
}
}  //  namespace

// Creates the MLIR Pipeline.
// If the temporary parameter lower_to_xla_hlo is
// true then the pipeline will include all the legalization passes.
void CreateConvertMlirToXlaHloPipeline(
    mlir::OpPassManager& pm, llvm::StringRef device_type,
    bool enable_op_fallback,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    bool lower_to_xla_hlo, bool allow_partial_conversion) {
  bool legalize_chlo = true;

  pm.addNestedPass<mlir::func::FuncOp>(
      tf2xla::internal::CreateInputLoweringMetricsPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::CreateTFXLADeviceSpecificTransformsPass(device_type));

  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateDropWhileShapeInvariantPass());
  // Create a replicated TensorList initialization ops for all of its uses. This
  // pass undo some CSE because shape_inference is not correctly able to
  // identify the shapes of TensorList initialization ops.
  // This pass requires CanonicalizerPass before
  // CreateTensorListOpsDecompositionPass for clean-ups.
  if (lower_to_xla_hlo) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateReplicateTensorListInitOpsPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // The SCCP pass performs constant propagation across the IR, which, for
  // example, propagates constant arguments into callee functions.
  // TODO(hinsu): Investigate if we really need SCCP pass before shape inference
  // and can do with just one pass after the shape inference.
  pm.addPass(mlir::createSCCPPass());
  // Guarantee all functions have one use, which enables shape inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run shape inference pass before tensorlist decomposition to get buffer
  // shape of uninitialized TensorLists.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Run SCCP pass again as the availability of shapes may open up new
  // opportunities for constant propagation. Note that the shape inference pass
  // doesn't materialize new constants even if those are computed internally for
  // the purpose of shape inference. These constants might be required by the
  // legalization passes.
  pm.addPass(mlir::createSCCPPass());

  if (lower_to_xla_hlo) {
    pm.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
  }
  pm.addPass(mlir::TF::CreateStackOpsDecompositionPass());
  if (lower_to_xla_hlo) {
    pm.addPass(mlir::TF::CreateTensorArrayOpsDecompositionPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());
  pm.addPass(mlir::TF::CreatePromoteResourcesToArgsPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Sink constants to regions so that ops requiring constant operands can
  // access the constant and there is no indirection through control flow region
  // arguments. Also, note that this pass is in MHLO but it is generic and sinks
  // constants for all ops with regions.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  if (lower_to_xla_hlo) {
    // Legalize any StableHLO ops to MHLO. Bridge still doesn't use StableHLO
    // but such ops might be present in the input from upstream like TFRT
    // compilation. Later on, this could be merged in the legalization pass when
    // we migrate bridge to StableHLO.

    // TODO(b/259459405): Avoid this peculiar use through some refactoring in
    // the caller.
    // This needs to happen before legalization.
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  }

  pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateLowerQuantizedPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::CreateConvertTFQuantTypesPass());

  if (lower_to_xla_hlo) {
    for (auto& target_pass : custom_legalization_passes) {
      pm.addNestedPass<mlir::func::FuncOp>(std::move(target_pass));
    }
    pm.addPass(mlir::mhlo::CreateLegalizeTFCollectivePass());
  }

  // These passes are grouped together as they have to run in specific order.
  // Passes before this can run relatively in any order, as long as they happen
  // before legalization.
  AddLegalizationPasses(pm, legalize_chlo, device_type, enable_op_fallback,
                        lower_to_xla_hlo);

  if (lower_to_xla_hlo) {
    // This pass operates on MHLO control flow ops so it should be legalized
    // after the control flow ops are legalized.
    pm.addPass(mlir::mhlo::CreateLegalizeTFCommunicationPass());

    // Everything should be MHLO after this.
    if (!allow_partial_conversion) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::mhlo::CreateVerifyTFXLALegalizationPass(legalize_chlo));
    }
  }

  if (CanInlineFunctionsPostLegalization(device_type)) {
    pm.addPass(mlir::createInlinerPass());
  }

  // In order to export to XLA, we must sink constants to control flow regions,
  // since XLA uses functional control flow.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());
}

Status RefineShapes(llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                    mlir::ModuleOp module) {
  auto producer_or = GetTfGraphProducerVersion(module);
  if (!producer_or.ok()) return producer_or.status();
  int64_t producer_version = producer_or.value();

  llvm::SmallVector<int64_t, 16> shape_backing;
  llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> arg_shapes_copy;
  {
    // Convert arg_shapes to a mlir friendly format.
    size_t count = 0;
    for (const TensorOrResourceShape& tensor_resource_shape : arg_shapes) {
      if (tensor_resource_shape.is_resource) continue;
      count += tensor_resource_shape.shape.dims();
    }
    shape_backing.resize(count);
    arg_shapes_copy.reserve(arg_shapes.size());
    size_t offset = 0;
    for (const TensorOrResourceShape& tensor_resource_shape : arg_shapes) {
      if (tensor_resource_shape.is_resource) {
        arg_shapes_copy.push_back(llvm::ArrayRef<int64_t>());
        continue;
      }
      size_t start = offset;
      for (tensorflow::TensorShapeDim dim : tensor_resource_shape.shape) {
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

  auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");

  mlir::StatusScopedDiagnosticHandler error_handler(module.getContext());
  mlir::LogicalResult result = mlir::TF::InferShapeForFunction(
      main_func, arg_shapes_copy, producer_version);

  if (failed(result)) {
    return error_handler.Combine(
        errors::Internal("MLIR Shape refinement failed"));
  }
  return error_handler.ConsumeStatus();
}

Status CreateAndRunMlirBridge(mlir::ModuleOp module_op,
                              llvm::StringRef device_type,
                              bool enable_op_fallback,
                              llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                                  custom_legalization_passes,
                              bool lower_to_xla_hlo,
                              llvm::StringRef module_name = llvm::StringRef()) {
  mlir::PassManager tf2xla(module_op.getContext());
  applyTensorflowAndCLOptions(tf2xla);
  CreateConvertMlirToXlaHloPipeline(tf2xla, device_type, enable_op_fallback,
                                    custom_legalization_passes,
                                    lower_to_xla_hlo,
                                    /*allow_partial_conversion=*/false);

  auto pass_instrumentors = mlir::GetPassInstrumentors();
  for (const auto& creator : pass_instrumentors) {
    tf2xla.addInstrumentation(creator());
  }
  MaybeDumpMlirModuleAndPasses(tf2xla, module_op, module_name.str(),
                               /*tag=*/"legalize_hlo_before");

  return RunMlirPipelineAndMaybeDumpResults(tf2xla, module_op,
                                            module_name.str());
}

Status BuildHloFromTfInner(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                           llvm::ArrayRef<xla::XlaOp> xla_params,
                           std::vector<xla::XlaOp>& returns,
                           llvm::StringRef device_type,
                           llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                               custom_legalization_passes) {
  TF_RETURN_IF_ERROR(CreateAndRunMlirBridge(module_op, device_type,
                                            /*enable_op_fallback=*/false,
                                            custom_legalization_passes,
                                            /*lower_to_xla_hlo=*/true));

  mlir::Block& block =
      module_op.lookupSymbol<mlir::func::FuncOp>("main").front();
  return mlir::BuildHloFromMlirHlo(block, builder, xla_params, returns);
}

Status ConvertMLIRWithOptionalXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool enable_op_fallback, bool return_tuple,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name, bool lower_to_xla_hlo) {
  TF_RETURN_IF_ERROR(CreateAndRunMlirBridge(
      module_op, device_type, enable_op_fallback, custom_legalization_passes,
      lower_to_xla_hlo, module_name));

  mlir::MlirToHloConversionOptions options;
  options.layout_preference_fn = [&](const xla::Shape& xla_shape)
      -> absl::StatusOr<mlir::XlaLayoutPreference> {
    TensorShape shape;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, &shape));
    TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                            xla_shape.element_type()));
    return shape_determination_fns.layout_preference_fn(shape, dtype,
                                                        std::nullopt);
  };
  options.shape_representation_fn =
      [&](const xla::Shape& xla_shape, bool fast_mem,
          mlir::XlaLayoutPreference layout_preference)
      -> absl::StatusOr<xla::Shape> {
    TensorShape shape;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, &shape));
    TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                            xla_shape.element_type()));
    return shape_determination_fns.shape_representation_fn(
        shape, dtype, fast_mem, layout_preference);
  };
  xla::HloProto hlo_proto;

  if (lower_to_xla_hlo) {
    TF_RETURN_IF_ERROR(mlir::ConvertMlirHloToHlo(
        module_op, &hlo_proto, use_tuple_args, return_tuple, options));
    *xla_computation = xla::XlaComputation(hlo_proto.hlo_module());
  }
  return absl::OkStatus();
}

// Wraps the optional lowering version to keep the api the same for clients.
Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool enable_op_fallback, bool return_tuple,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name) {
  return ConvertMLIRWithOptionalXlaComputation(
      module_op, device_type, xla_computation, use_tuple_args,
      enable_op_fallback, return_tuple, shape_determination_fns,
      custom_legalization_passes, module_name, /*lower_to_xla_hlo=*/true);
}

Status CompileMlirSetup(mlir::ModuleOp module_op,
                        llvm::ArrayRef<TensorOrResourceShape> arg_shapes) {
  // Use arg_shapes to improve the mlir type information of `main` in module_op.
  TF_RETURN_IF_ERROR(RefineShapes(arg_shapes, module_op));

  if (VLOG_IS_ON(2))
    tensorflow::DumpMlirOpToFile("compile_mlir_shape_refiner", module_op);

  return absl::OkStatus();
}

Status BuildHloFromTf(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                      llvm::ArrayRef<xla::XlaOp> xla_params,
                      std::vector<xla::XlaOp>& returns,
                      llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                      llvm::StringRef device_type,
                      llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                          custom_legalization_passes) {
  if (VLOG_IS_ON(2))
    tensorflow::DumpMlirOpToFile("build_hlo_tf_before", module_op);

  TF_RETURN_IF_ERROR(CompileMlirSetup(module_op, arg_shapes));

  // Convert MLIR module to XLA HLO proto contained in XlaComputation.
  TF_RETURN_IF_ERROR(BuildHloFromTfInner(module_op, builder, xla_params,
                                         returns, device_type,
                                         custom_legalization_passes));

  if (VLOG_IS_ON(2))
    tensorflow::DumpMlirOpToFile("build_hlo_tf_after", module_op);

  return absl::OkStatus();
}

Status PopulateCollectiveInfo(mlir::ModuleOp module_op,
                              XlaCompilationResult* compilation_result) {
  // The StringRef cast is necessary before cxx14.
  mlir::IntegerAttr group_key_attr =
      module_op->getAttrOfType<mlir::IntegerAttr>(
          mlir::StringRef(kGroupKeyAttrName.data(), kGroupKeyAttrName.size()));
  mlir::IntegerAttr group_size_attr =
      module_op->getAttrOfType<mlir::IntegerAttr>(mlir::StringRef(
          kGroupSizeAttrName.data(), kGroupSizeAttrName.size()));
  if (group_key_attr == nullptr && group_size_attr == nullptr) {
    // No CollectiveInfo is present.
    return absl::OkStatus();
  }
  DCHECK(group_key_attr != nullptr)
      << "module attribute " << kGroupKeyAttrName
      << " is required for CollectiveInfo but not found.";
  DCHECK(group_size_attr != nullptr)
      << "module attribute " << kGroupSizeAttrName
      << " is required for CollectiveInfo but not found.";
  int32_t group_key = group_key_attr.getInt();
  int32_t group_size = group_size_attr.getInt();
  VLOG(2) << "Populating CollectiveInfo: group_key=" << group_key
          << " group_size=" << group_size;
  compilation_result->collective_info = {group_key, group_size, 0};
  return absl::OkStatus();
}

Status PopulateResultIOInfo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    bool use_tuple_args, bool use_resource_updates_for_aliases,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result) {
  // Construct mapping from XlaComputation's arg to input edges of execute
  // node.
  GetInputMappingForMlir(arg_shapes.size(), &compilation_result->input_mapping);

  // Compute all input shapes.
  TF_RETURN_IF_ERROR(GetXlaInputShapes(module_op, arg_shapes, use_tuple_args,
                                       shape_determination_fns,
                                       &compilation_result->xla_input_shapes));

  // Compute all output descriptions and resource writes
  return GetOutputInfo(
      module_op, use_resource_updates_for_aliases, shape_determination_fns,
      &compilation_result->xla_output_shape, &compilation_result->outputs,
      &compilation_result->resource_updates);
}

absl::StatusOr<std::string> CompileMlirToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    bool use_return_tuple, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name, bool lower_to_xla_hlo) {
  // If we don't want to enable fall back or any form or lowering then we
  // don't need to check if the module has unsupported ops.
  if (enable_op_fallback && lower_to_xla_hlo &&
      GetMlirBridge2ndPhaseRolloutPolicy(module_op) ==
          MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis) {
    return CompileToHloGraphAnalysisFailedError();
  }

  TF_RETURN_IF_ERROR(CompileMlirSetup(module_op, arg_shapes));

  // Convert MLIR module to XLA HLO proto contained in XlaComputation.
  compilation_result->computation = std::make_shared<xla::XlaComputation>();
  TF_RETURN_IF_ERROR(ConvertMLIRWithOptionalXlaComputation(
      module_op, device_type, compilation_result->computation.get(),
      use_tuple_args, enable_op_fallback, use_return_tuple,
      shape_determination_fns, custom_legalization_passes, module_name,
      lower_to_xla_hlo));

  auto mlir_compilation = SerializeMlirModule(module_op);

  // Only attempt to fill in the compilation result's IO info if lowering.
  if (lower_to_xla_hlo) {
    TF_RETURN_IF_ERROR(PopulateCollectiveInfo(module_op, compilation_result));

    auto populate_result = PopulateResultIOInfo(
        module_op, arg_shapes, use_tuple_args, use_resource_updates_for_aliases,
        shape_determination_fns, compilation_result);
    if (!populate_result.ok()) {
      llvm::errs() << "Failed to populate result io info";
      return populate_result;
    }
  }
  return mlir_compilation;
}

absl::StatusOr<std::string> CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name, bool lower_to_xla_hlo) {
  mlir::DialectRegistry mlir_registry;
  RegisterDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;

  TF_RETURN_IF_ERROR(
      DeserializeMlirModule(mlir_module_string, &mlir_context, &mlir_module));
  llvm::SmallVector<TensorOrResourceShape, 4> tensor_or_resource_shapes;
  tensor_or_resource_shapes.reserve(arg_shapes.size());
  for (const auto& arg_shape : arg_shapes)
    tensor_or_resource_shapes.push_back({arg_shape});
  return CompileMlirToXlaHlo(
      mlir_module.get(), tensor_or_resource_shapes, device_type, use_tuple_args,
      enable_op_fallback, /*use_return_tuple=*/true,
      /*use_resource_updates_for_aliases=*/false, shape_determination_fns,
      compilation_result, custom_legalization_passes, module_name,
      lower_to_xla_hlo);
}

// Rewrites the given module with specified args. Each of the constant args gets
// inlined in the "main' function and the corresponding argument is removed from
// the signature. For resource args, their subtypes are populated. Returns the
// original indices for the other arguments on success.
static absl::StatusOr<std::vector<int>> RewriteWithArgs(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args) {
  mlir::func::FuncOp main_fn =
      module_op.lookupSymbol<mlir::func::FuncOp>("main");
  std::vector<int> params;

  bool has_resource_args = false;
  auto builder = mlir::OpBuilder(main_fn.getBody());
  std::vector<int> args_to_erase;
  for (int idx = 0; idx < args.size(); idx++) {
    const XlaArgument& xla_arg = args[idx];
    mlir::BlockArgument mlir_arg = main_fn.getArgument(idx);
    if (xla_arg.kind == XlaArgument::kResource) {
      mlir::Type element_type;
      if (xla_arg.type == DT_INVALID) {
        return errors::Unimplemented(absl::StrCat(
            "Argument ", idx,
            " is an uninitialized resource variable which is currently"
            " unsupported in the MLIR-based TPU bridge"));
      }
      TF_RETURN_IF_ERROR(ConvertDataType(xla_arg.type, builder, &element_type));
      TF_ASSIGN_OR_RETURN(TensorShape arg_shape,
                          GetTensorShapeFromXlaArgument(xla_arg));
      auto resource_shape = arg_shape.dim_sizes();
      llvm::SmallVector<int64_t, 4> resource_subtype_shape(
          resource_shape.begin(), resource_shape.end());
      auto resource_subtype =
          GetTypeFromTFTensorShape(resource_subtype_shape, element_type);
      auto resource_type =
          mlir::TF::ResourceType::get({resource_subtype}, builder.getContext());

      auto tensor_type = mlir::cast<mlir::TensorType>(mlir_arg.getType());
      if (tensor_type.hasRank()) {
        mlir_arg.setType(
            GetTypeFromTFTensorShape(tensor_type.getShape(), resource_type));
      } else {
        mlir_arg.setType(mlir::UnrankedTensorType::get(resource_type));
      }
      has_resource_args = true;
    }
    if (xla_arg.kind != XlaArgument::kConstant) {
      params.push_back(idx);
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto value_attr,
                        ConvertTensor(xla_arg.constant_value, &builder));
    // TODO(hinsu): Use the actual location of the constant.
    auto constant = builder.create<mlir::TF::ConstOp>(
        mlir::UnknownLoc::get(module_op.getContext()), value_attr);
    mlir_arg.replaceAllUsesWith(constant);
    args_to_erase.push_back(idx);
  }

  if (has_resource_args) {
    llvm::SmallVector<mlir::Type, 4> updated_argument_types;
    updated_argument_types.reserve(main_fn.getNumArguments());
    for (mlir::BlockArgument& arg : main_fn.getArguments())
      updated_argument_types.push_back(arg.getType());

    main_fn.setType(
        mlir::FunctionType::get(main_fn.getContext(), updated_argument_types,
                                main_fn.getFunctionType().getResults()));
  }

  for (int idx : llvm::reverse(args_to_erase)) main_fn.eraseArgument(idx);

  return params;
}

Status CompileGraphSetup(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args,
    std::vector<int>* remaining_params,
    llvm::SmallVector<TensorOrResourceShape, 4>& arg_shapes) {
  TF_ASSIGN_OR_RETURN(*remaining_params, RewriteWithArgs(module_op, args));
  arg_shapes.reserve(remaining_params->size());
  for (unsigned idx : *remaining_params) {
    const auto& arg = args[idx];
    TF_ASSIGN_OR_RETURN(TensorShape arg_shape,
                        GetTensorShapeFromXlaArgument(arg));
    arg_shapes.push_back({arg_shape,
                          /*is_resource=*/arg.kind == XlaArgument::kResource});
  }

  mlir::PassManager pm(module_op.getContext());
  applyTensorflowAndCLOptions(pm);
  mlir::TF::StandardPipelineOptions tf_options;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("compile_graph_setup_before", module_op);
  if (VLOG_IS_ON(2)) {
    // Print the whole module after each pass which requires disabling
    // multi-threading as well.
    module_op.getContext()->disableMultithreading();
    pm.enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>(
        /*print_module_scope=*/true));
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module_op.getContext());
  if (failed(pm.run(module_op))) return diag_handler.ConsumeStatus();
  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("compile_graph_setup_after", module_op);

  return absl::OkStatus();
}

Status BuildHloFromModule(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                          llvm::ArrayRef<xla::XlaOp> xla_params,
                          std::vector<xla::XlaOp>& returns,
                          llvm::ArrayRef<XlaArgument> args,
                          llvm::StringRef device_type) {
  std::vector<int> remaining_params;
  llvm::SmallVector<TensorOrResourceShape, 4> arg_shapes;
  TF_RETURN_IF_ERROR(
      CompileGraphSetup(module_op, args, &remaining_params, arg_shapes));
  llvm::SmallVector<xla::XlaOp, 2> remaining_xla_params;
  for (auto i : remaining_params) remaining_xla_params.push_back(xla_params[i]);
  TF_RETURN_IF_ERROR(CompileMlirSetup(module_op, arg_shapes));

  mlir::PassManager tf2xla(module_op.getContext());
  CreateMlirLoweringPassesPipeline(tf2xla);

  MaybeDumpMlirModuleAndPasses(tf2xla, module_op, /*module_name=*/"",
                               /*tag=*/"legalize_hlo_before");

  TF_RETURN_IF_ERROR(RunMlirPipelineAndMaybeDumpResults(tf2xla, module_op));

  mlir::Block& block =
      module_op.lookupSymbol<mlir::func::FuncOp>("main").front();
  TF_RETURN_IF_ERROR(
      mlir::BuildHloFromMlirHlo(block, builder, remaining_xla_params, returns));

  if (VLOG_IS_ON(2)) {
    tensorflow::DumpMlirOpToFile("build_hlo_tf_after", module_op);
  }

  return absl::OkStatus();
}

Status CompileGraphToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    bool use_return_tuple,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes) {
  std::vector<int> remaining_params;
  llvm::SmallVector<TensorOrResourceShape, 4> arg_shapes;
  TF_RETURN_IF_ERROR(
      CompileGraphSetup(module_op, args, &remaining_params, arg_shapes));

  auto compile_mlir_result = CompileMlirToXlaHlo(
      module_op, arg_shapes, device_type, use_tuple_args, enable_op_fallback,
      use_return_tuple,
      /*use_resource_updates_for_aliases=*/true, shape_determination_fns,
      compilation_result, custom_legalization_passes);
  compilation_result->input_mapping = remaining_params;
  return compile_mlir_result.status();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphToModule(
    bool unconditionally_use_set_output_shapes, const Graph& graph,
    llvm::ArrayRef<std::string> control_rets,
    const FunctionLibraryDefinition& flib_def, mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  context->appendDialectRegistry(registry);
  GraphImportConfig config;
  config.graph_as_function = true;
  config.control_outputs = control_rets;
  // Disable shape inference during import as some TensorFlow op fails during
  // shape inference with dynamic shaped operands. This in turn causes the
  // import to fail. Shape inference during import is going to be removed and
  // the shape inference pass is run early in the pass pipeline, shape inference
  // during import is not necessary.
  config.enable_shape_inference = false;
  // Some graphs may require _output_shapes (an unregistered attribute)
  // to override shapes. It is unfortunately not always set correctly so only
  // do it optionally.
  config.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes;
  GraphDebugInfo debug_info;
  return ConvertGraphToMlir(graph, debug_info, flib_def, config, context);
}

Status BuildHloFromGraph(
    const Graph& graph, xla::XlaBuilder& builder,
    mlir::MLIRContext& mlir_context, llvm::ArrayRef<xla::XlaOp> xla_params,
    std::vector<xla::XlaOp>& returns, bool unconditionally_use_output_shapes,
    llvm::ArrayRef<XlaArgument> args, llvm::ArrayRef<std::string> control_rets,
    llvm::StringRef device_type, const FunctionLibraryDefinition& flib_def) {
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      GraphToModule(unconditionally_use_output_shapes, graph,
                                    control_rets, flib_def, &mlir_context));
  return BuildHloFromModule(module.get(), builder, xla_params, returns, args,
                            device_type);
}

void RegisterConvertMlirToXlaHloPipelineWithDefaults() {
  static mlir::PassPipelineRegistration<> pipeline(
      "tf-to-hlo-pipeline",
      "Convert TF dialect to HLO dialect (used for compilation in bridge).",
      [](mlir::OpPassManager& pm) {
        tensorflow::CreateConvertMlirToXlaHloPipeline(
            pm, /*device_type=*/"XLA_CPU_JIT", /*enable_op_fallback=*/false,
            /*custom_legalization_passes=*/{});
      });
}

}  // namespace tensorflow
