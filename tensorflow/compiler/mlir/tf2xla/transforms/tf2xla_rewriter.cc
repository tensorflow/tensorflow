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
#include "tensorflow/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tpu_embedding_ops_registry.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/mlir_hlo_builder.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {
namespace {

using ::mlir::FunctionType;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::func::FuncOp;
using ::tensorflow::Tensor;
using ::tsl::StatusOr;
using ::xla::XlaComputation;

static std::unique_ptr<tensorflow::StaticDeviceMgr> CreateDeviceMgr(
    const std::string& device_type) {
  // Register compilation kernels for all registered XLA backends.
  tensorflow::XlaOpRegistry::RegisterCompilationKernels();

  auto device = std::make_unique<tensorflow::XlaCompilationDevice>(
      tensorflow::SessionOptions(), tensorflow::DeviceType(device_type));
  return std::make_unique<tensorflow::StaticDeviceMgr>(std::move(device));
}

};  // namespace

LogicalResult Tf2XlaRewriter::RewriteOp(Operation* op,
                                        PatternRewriter& rewriter,
                                        const std::string& device_type,
                                        bool is_module_pass,
                                        bool use_tf2xla_hlo_importer) {
  Tf2XlaRewriter tf2xla_rewriter(op, rewriter, device_type, is_module_pass,
                                 use_tf2xla_hlo_importer);
  return tf2xla_rewriter.LegalizeOp();
}

Tf2XlaRewriter::Tf2XlaRewriter(Operation* op, PatternRewriter& rewriter,
                               const std::string& device_type,
                               bool is_module_pass,
                               bool use_tf2xla_hlo_importer)
    : op_(op),
      device_type_(device_type),
      rewriter_(rewriter),
      hlo_builder_(op->getName().getStringRef().str(), rewriter_, op->getLoc(),
                   /*build_functions=*/is_module_pass),
      context_(nullptr),
      use_tf2xla_hlo_importer_(use_tf2xla_hlo_importer),
      xla_builder_(op_->getName().getStringRef().str()) {}

Tf2XlaRewriter::~Tf2XlaRewriter() {
  if (context_) context_->Unref();
}

tsl::StatusOr<std::string> Tf2XlaRewriter::CreateUniqueTranslatedFunctionName(
    std::string candidate_name) {
  ModuleOp parent_module = op_->getParentOfType<ModuleOp>();
  for (int i = 0; i < INT_MAX; i++) {
    std::string renamed_kernel = absl::StrCat(
        "tf2xla_rewriter.", candidate_name, ".", std::to_string(i));

    mlir::func::FuncOp candidate_func =
        parent_module.lookupSymbol<mlir::func::FuncOp>(renamed_kernel);
    if (!candidate_func) {
      return renamed_kernel;
    }
  }

  return tsl::errors::AlreadyExists(
      absl::StrCat("Could not create a unique function name for op ",
                   op_->getName().getStringRef().str()));
}

tsl::StatusOr<mlir::func::FuncOp> Tf2XlaRewriter::ImportXlaComputation(
    XlaComputation& computation) {
  ModuleOp mlir_module = op_->getParentOfType<ModuleOp>();
  mlir::Builder builder(mlir_module);
  mlir::SymbolTable symbol_table(mlir_module);

  xla::DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(auto hlo_module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          computation.proto(), debug_options));
  TF_ASSIGN_OR_RETURN(
      auto hlo_module,
      xla::HloModule::CreateFromProto(computation.proto(), hlo_module_config));

  std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>
      function_map;

  TF_ASSIGN_OR_RETURN(FuncOp translated_function,
                      xla::HloFunctionImporter::ImportAsFunc(
                          *hlo_module->entry_computation(), symbol_table,
                          &function_map, &builder, /*is_main*/ false));

  return translated_function;
}

LogicalResult Tf2XlaRewriter::PrepareParams() {
  // XlaCompiler within the context is only used by the functional ops to
  // compile functions. We are not handling those at the moment so
  // XlaCompiler is not required.
  if (use_tf2xla_hlo_importer_) {
    context_ = new tensorflow::XlaContext(/*compiler=*/nullptr, &xla_builder_,
                                          /*graph=*/nullptr);
  } else {
    context_ = new tensorflow::XlaContext(/*compiler=*/nullptr, &hlo_builder_,
                                          /*graph=*/nullptr);
  }
  context_->Ref();

  device_mgr_ = CreateDeviceMgr(device_type_);
  if (!device_mgr_) return failure();

  // Type of params_.device is DeviceBase* so store it as Device* to access
  // derived class method.
  device_ = device_mgr_->ListDevices().front();
  params_.device = device_;
  params_.resource_manager = device_->resource_manager();

  // Resources are cleared at the time of device manager destruction so pass
  // no-op cleanup function.
  auto cleanup = [](const std::string& name) {};
  // Use step_id zero as we only have a single context concurrently and
  // concurrently running each of the MLIR functions create a new device.
  step_container_ = std::make_unique<tensorflow::ScopedStepContainer>(
      /*step_id=*/0, cleanup);
  tsl::Status status = step_container_->Create(
      device_->resource_manager(),
      tensorflow::XlaContext::kXlaContextResourceName, context_);
  if (!status.ok()) {
    return emitRemark(op_->getLoc())
           << "failed to create XlaContext resource: " << status.ToString();
  }
  params_.step_container = step_container_.get();

  tsl::StatusOr<int64_t> version_or = tensorflow::GetTfGraphProducerVersion(
      op_->getParentOfType<mlir::ModuleOp>());
  if (!version_or.ok()) {
    return emitError(op_->getLoc()) << version_or.status().ToString();
  }

  flib_def_ = std::make_unique<tensorflow::FunctionLibraryDefinition>(
      tensorflow::OpRegistry::Global(), tensorflow::FunctionDefLibrary());
  pflr_ = std::make_unique<tensorflow::ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), tensorflow::Env::Default(), /*config=*/nullptr,
      version_or.value(), flib_def_.get(), tensorflow::OptimizerOptions());
  params_.function_library = pflr_->GetFLR(device_->name());
  return success();
}

// Returns true if the given type is a ranked tensor type with static or
// bounded dimensions.
bool IsBounded(Type ty) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding =
      ranked_ty.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>();
  if (!encoding) return false;

  for (int i = 0; i < ranked_ty.getRank(); ++i) {
    if (ranked_ty.isDynamicDim(i) &&
        encoding.getBounds()[i] == ShapedType::kDynamic) {
      return false;
    }
  }
  return true;
}

bool HasSymbolRefAttr(Operation* op) {
  for (const auto& attr : op->getAttrs()) {
    Attribute attr_value = attr.getValue();
    if (attr_value.isa<SymbolRefAttr>()) {
      return true;
    } else if (auto array_attr = attr_value.dyn_cast<ArrayAttr>()) {
      if (!array_attr.empty() && array_attr.begin()->isa<SymbolRefAttr>()) {
        return true;
      }
    }
  }
  return false;
}

LogicalResult Tf2XlaRewriter::PrepareKernelInputs(
    const llvm::SmallDenseSet<int>& required_consts,
    std::vector<tensorflow::XlaExpression>& expressions,
    std::vector<tensorflow::Tensor>& tensors,
    std::vector<tensorflow::TensorValue>& inputs) {
  // Prepare the list of Tensor inputs for the kernel.
  for (auto it : llvm::enumerate(op_->getOperands())) {
    Value operand = it.value();
    size_t idx = it.index();

    tensorflow::XlaExpression expr = GetExprForOperand(operand, op_, idx);
    tensorflow::XlaExpression::Kind kind = expr.kind();
    if (kind == tensorflow::XlaExpression::Kind::kInvalid) return failure();
    if (required_consts.count(idx) &&
        kind != tensorflow::XlaExpression::Kind::kConstant) {
      return op_->emitRemark()
             << "lowering requires operand #" << idx << " to be a constant";
    }
    expressions.push_back(expr);

    if (!tensorflow::DataTypeCanUseMemcpy(expr.dtype())) {
      return op_->emitRemark()
             << "skipping legalization due to unsupported type "
             << operand.getType();
    }

    auto shape_or = expr.GetShape();
    if (!shape_or.ok()) {
      return op_->emitRemark()
             << "failed to get shape for expression. " << expr.HumanString();
    }

    tensors.emplace_back(
        device_->GetAllocator(tensorflow::AllocatorAttributes()), expr.dtype(),
        shape_or.value());

    tensorflow::Tensor& tensor = tensors.back();
    tensorflow::XlaExpression::AssignExpressionToTensor(expr, &tensor);
    inputs.emplace_back(&tensor);
  }

  return success();
}

LogicalResult Tf2XlaRewriter::LegalizeOp() {
  for (Type ty : op_->getOperandTypes()) {
    auto ranked_ty = ty.dyn_cast<ShapedType>();
    // Only bounded operands are supported in the XLA builders.
    if (!IsBounded(ranked_ty)) {
      return op_->emitRemark()
             << "lowering requires bounded tensor operands " << ranked_ty;
    }
  }

  if (HasSymbolRefAttr(op_)) {
    return op_->emitRemark() << "ops with symbol references are not supported";
  }

  auto nodedef_or = tensorflow::ConvertTFDialectOpToNodeDef(
      op_, name_mapper_.GetUniqueName(op_),
      /*ignore_unregistered_attrs=*/true);
  if (!nodedef_or.ok()) {
    return op_->emitRemark() << "failed to convert op to NodeDef: "
                             << nodedef_or.status().ToString();
  }

  if (failed(PrepareParams())) return failure();

  std::shared_ptr<const tensorflow::NodeProperties> props;
  tsl::Status status = tensorflow::NodeProperties::CreateFromNodeDef(
      *nodedef_or.value(),
      params_.function_library->GetFunctionLibraryDefinition(), &props);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to create NodeProperties: " << status.ToString();
  }
  tensorflow::OpKernel* op_kernel_raw;
  status = params_.function_library->CreateKernel(props, &op_kernel_raw);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to create tf2xla kernel: " << status.ToString();
  }
  // Transfer ownership of the kernel to a local smart pointer.
  auto op_kernel = absl::WrapUnique(op_kernel_raw);

  std::vector<int> required_constants;
  status = tensorflow::XlaOpRegistry::CompileTimeConstantInputs(
      *op_kernel, &required_constants);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to compute required constants: " << status.ToString();
  }

  llvm::SmallDenseSet<int> required_consts;
  required_consts.insert(required_constants.begin(), required_constants.end());

  // TensorValue in inputs are backed by tensors which in turn depend on
  // expressions. So, pre-allocate them to the required size. Subtle note:
  // Since these are assigned to params_, these have to live past the kernel
  // compilation.
  std::vector<tensorflow::XlaExpression> expressions;
  std::vector<tensorflow::Tensor> tensors;
  std::vector<tensorflow::TensorValue> inputs;
  expressions.reserve(op_->getNumOperands());
  tensors.reserve(op_->getNumOperands());
  inputs.reserve(op_->getNumOperands());

  if (failed(
          PrepareKernelInputs(required_consts, expressions, tensors, inputs)))
    return failure();

  params_.inputs = inputs;
  params_.op_kernel = op_kernel.get();
  llvm::SmallVector<tensorflow::AllocatorAttributes, 4> output_attr(
      op_->getNumResults());
  params_.output_attr_array = output_attr.data();

  hlo_builder_.setInsertionPoint(op_);
  hlo_builder_.SetLocation(op_->getLoc());

  tensorflow::OpKernelContext op_context(&params_, op_->getNumResults());
  device_->Compute(params_.op_kernel, &op_context);

  status = op_context.status();
  status.Update(hlo_builder_.GetCurrentStatus());
  if (!status.ok()) {
    return op_->emitRemark()
           << "compilation to HLO failed: " << status.ToString();
  }

  if (failed(VerifyOpResults(op_context))) return failure();

  FuncOp translated_function;
  if (use_tf2xla_hlo_importer_) {
    StatusOr<FuncOp> translated_function_or_status =
        CompileWithHloImporter(op_context);
    if (!translated_function_or_status.ok()) {
      return op_->emitRemark()
             << translated_function_or_status.status().ToString();
    }
    translated_function = translated_function_or_status.value();
  }

  llvm::SmallVector<Value> output_values;
  if (failed(
          GetKernelOutputs(op_context, translated_function, output_values))) {
    return failure();
  }

  rewriter_.replaceOp(op_, output_values);
  return success();
}

tsl::Status Tf2XlaRewriter::CreateUniqueComputationNames(
    XlaComputation& computation) {
  int entry_computation = computation.proto().entry_computation_id();
  std::string new_entry_computation_name = "";

  for (xla::HloComputationProto& sub_computation :
       *computation.mutable_proto()->mutable_computations()) {
    TF_ASSIGN_OR_RETURN(
        std::string renamed_computation,
        CreateUniqueTranslatedFunctionName(sub_computation.name()));
    sub_computation.set_name(renamed_computation);

    if (sub_computation.id() == entry_computation) {
      new_entry_computation_name = renamed_computation;
    }
  }

  computation.mutable_proto()->set_entry_computation_name(
      new_entry_computation_name);
  computation.mutable_proto()->set_name(new_entry_computation_name);
  return tsl::OkStatus();
}

tsl::StatusOr<mlir::func::FuncOp> Tf2XlaRewriter::CompileWithHloImporter(
    tensorflow::OpKernelContext& op_context) {
  if (!use_tf2xla_hlo_importer_) {
    return tsl::errors::InvalidArgument(
        "Cannot compile with HloImporter because it isn't supported");
  }

  // XLA can only return a single value. Wrap all output op return values
  // in a Tuple op that gets unpacked later.
  std::vector<xla::XlaOp> output_values;
  for (int i = 0, e = op_->getNumResults(); i < e; i++) {
    tensorflow::Tensor* output = op_context.mutable_output(i);
    const tensorflow::XlaExpression* expr =
        tensorflow::XlaExpression::CastExpressionFromTensor(*output);
    output_values.push_back(expr->AsXlaOp(&xla_builder_));
  }

  absl::Span<const xla::XlaOp> return_values(output_values);
  xla::XlaOp root_value = xla::Tuple(&xla_builder_, return_values);

  TF_ASSIGN_OR_RETURN(XlaComputation computation,
                      xla_builder_.Build(root_value,
                                         /*remove_dynamic_dimensions=*/false));
  TF_RETURN_IF_ERROR(CreateUniqueComputationNames(computation));

  return ImportXlaComputation(computation);
}

mlir::LogicalResult Tf2XlaRewriter::VerifyOpResults(
    tensorflow::OpKernelContext& op_context) {
  for (int i = 0, e = op_->getNumResults(); i < e; i++) {
    tensorflow::Tensor* output = op_context.mutable_output(i);
    const tensorflow::XlaExpression* expr =
        tensorflow::XlaExpression::CastExpressionFromTensor(*output);

    if (expr->kind() != tensorflow::XlaExpression::Kind::kXlaOp &&
        expr->kind() != tensorflow::XlaExpression::Kind::kConstant) {
      return op_->emitRemark(absl::StrCat(
          "expects XlaExpression of kind kXlaOp or kConstant in compiled "
          "output index ",
          i));
    }
  }
  return success();
}

// XLA computations can only return a single value, but TF ops can return
// multiple values. We get around this by returning a tuple as an XLA op. We
// then unpack it here to return the multiple values instead.
mlir::LogicalResult Tf2XlaRewriter::UnpackTupleResults(
    mlir::func::FuncOp translated_function) {
  if (translated_function.getBlocks().size() != 1) {
    return op_->emitRemark() << "Translated function has more than one block. "
                                "This isn't supported yet.";
  }

  func::ReturnOp xla_return_op = llvm::dyn_cast<func::ReturnOp>(
      translated_function.back().getTerminator());
  if (!xla_return_op) {
    return op_->emitRemark() << "Could not find return value";
  }

  if (xla_return_op->getNumOperands() != 1) {
    return op_->emitRemark() << "Return value has more than one op, returning";
  }

  mhlo::TupleOp tuple_result = llvm::dyn_cast<mhlo::TupleOp>(
      xla_return_op->getOperand(0).getDefiningOp());
  if (!tuple_result) {
    return op_->emitRemark()
           << "Translated Function didn't return a tuple type";
  }

  if (tuple_result->getNumOperands() != op_->getNumResults()) {
    return op_->emitRemark() << "Translated function tuple has different "
                                "number of results than original op";
  }

  FunctionType new_type =
      FunctionType::get(op_->getContext(), op_->getOperandTypes(),
                        // Note: Tuple results might have been type specialized
                        // so we overwrite the return type with the tuple result
                        // types instead of the original op_ return type.
                        tuple_result->getOperandTypes());
  translated_function.setType(new_type);

  xla_return_op->setOperands(tuple_result->getOperands());
  tuple_result.getOperation()->erase();

  return success();
}

mlir::LogicalResult Tf2XlaRewriter::InsertCallToTranslatedFunction(
    mlir::func::FuncOp translated_function, llvm::SmallVector<Value>& outputs) {
  if (translated_function.getFunctionType().getNumResults() !=
      op_->getNumResults()) {
    return op_->emitRemark() << "Translated function doesn't have the same "
                                "number of results as the original op";
  }

  mlir::OpBuilder builder(op_);
  auto call_op = builder.create<mlir::func::CallOp>(
      op_->getLoc(), translated_function, op_->getOperands());

  for (int i = 0; i < op_->getNumResults(); i++) {
    outputs.emplace_back(call_op.getResult(i));
  }

  return success();
}

mlir::LogicalResult Tf2XlaRewriter::GetKernelOutputs(
    tensorflow::OpKernelContext& op_context,
    mlir::func::FuncOp translated_function, llvm::SmallVector<Value>& outputs) {
  outputs.reserve(op_->getNumResults());

  if (use_tf2xla_hlo_importer_) {
    if (failed(UnpackTupleResults(translated_function))) return failure();
    return InsertCallToTranslatedFunction(translated_function, outputs);
  }

  for (int i = 0, e = op_->getNumResults(); i < e; i++) {
    tensorflow::Tensor* output = op_context.mutable_output(i);
    const tensorflow::XlaExpression* expr =
        tensorflow::XlaExpression::CastExpressionFromTensor(*output);

    mlir::Value value = hlo_builder_.GetValue(expr->AsXlaOp(&hlo_builder_));
    outputs.push_back(value);
  }

  return success();
}

tensorflow::XlaExpression Tf2XlaRewriter::GetExprForOperand(
    Value operand, Operation* op, int64_t operand_index) {
  ElementsAttr const_attr;
  auto defining_op = operand.getDefiningOp();

  ::xla::XlaOp xla_op;
  if (use_tf2xla_hlo_importer_) {
    xla_op = xla::Parameter(&xla_builder_, operand_index,
                            xla::TypeToShape(operand.getType()),
                            std::to_string(operand_index));
  }

  if (defining_op && matchPattern(defining_op, m_Constant(&const_attr))) {
    tensorflow::Tensor tensor;
    auto status = tensorflow::ConvertToTensor(const_attr, &tensor);
    if (!status.ok()) {
      op->emitRemark() << "skipping legalization due to failed const conversion"
                       << status.ToString();
      return tensorflow::XlaExpression::Invalid();
    }

    return tensorflow::XlaExpression::Constant(tensor);
  }

  if (!use_tf2xla_hlo_importer_) {
    auto xla_op_or = hlo_builder_.MakeXlaOp(operand);
    if (!xla_op_or.ok()) {
      op->emitRemark() << "skipping legalization due to "
                       << xla_op_or.status().ToString();
      return tensorflow::XlaExpression::Invalid();
    }
    xla_op = xla_op_or.value();
  }

  tensorflow::DataType dtype;
  auto status = tensorflow::ConvertToDataType(operand.getType(), &dtype);
  if (!status.ok()) {
    op->emitRemark() << "skipping legalization due to " << status.ToString();
    return tensorflow::XlaExpression::Invalid();
  }
  return tensorflow::XlaExpression::XlaOp(xla_op, dtype);
}

}  // namespace mhlo
}  // namespace mlir
