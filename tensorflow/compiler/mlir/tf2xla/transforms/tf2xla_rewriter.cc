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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
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
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {
namespace {

using ::mlir::ModuleOp;
using ::tensorflow::Tensor;
using ::tsl::StatusOr;
using ::xla::XlaComputation;

// The OpOrArgLocNameMapper adds invalid characters to the name of the op when
// concatenating locations. This version removes those characters to make the
// name valid for NodeDef.
class OpOrArgLocNameMapperWithoutInvalidCharacters
    : public tensorflow::OpOrArgLocNameMapper {
 public:
  OpOrArgLocNameMapperWithoutInvalidCharacters() = default;
  ~OpOrArgLocNameMapperWithoutInvalidCharacters() override = default;

 protected:
  std::string GetName(tensorflow::OpOrVal op_or_val) override {
    std::string name = OpOrArgLocNameMapper::GetName(op_or_val);
    return absl::StrReplaceAll(name, {{";", "."}});
  }
};

static std::unique_ptr<tensorflow::StaticDeviceMgr> CreateDeviceMgr(
    const std::string& device_type) {
  // Register compilation kernels for all registered XLA backends.
  tensorflow::XlaOpRegistry::RegisterCompilationKernels();

  auto device = std::make_unique<tensorflow::XlaCompilationDevice>(
      tensorflow::SessionOptions(), tensorflow::DeviceType(device_type));
  return std::make_unique<tensorflow::StaticDeviceMgr>(std::move(device));
}

bool RootInstructionIsTuple(const xla::HloModule& hlo_module) {
  xla::HloInstruction* root_instruction =
      hlo_module.entry_computation()->root_instruction();

  return root_instruction->opcode() == xla::HloOpcode::kTuple;
}

};  // namespace

LogicalResult Tf2XlaRewriter::RewriteOp(Operation* op,
                                        PatternRewriter& rewriter,
                                        const std::string& device_type) {
  Tf2XlaRewriter tf2xla_rewriter(op, rewriter, device_type);
  return tf2xla_rewriter.LegalizeOp();
}

Tf2XlaRewriter::Tf2XlaRewriter(Operation* op, PatternRewriter& rewriter,
                               const std::string& device_type)
    : op_(op),
      device_type_(device_type),
      rewriter_(rewriter),
      name_mapper_(
          std::make_unique<OpOrArgLocNameMapperWithoutInvalidCharacters>()),
      context_(nullptr),
      xla_builder_(op_->getName().getStringRef().str()) {}

Tf2XlaRewriter::~Tf2XlaRewriter() {
  if (context_) context_->Unref();
}

absl::StatusOr<mhlo::TupleOp> Tf2XlaRewriter::ImportXlaComputation(
    XlaComputation& computation) {
  xla::DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(auto hlo_module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          computation.proto(), debug_options));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::HloModule::CreateFromProto(computation.proto(), hlo_module_config));

  if (!RootInstructionIsTuple(*hlo_module)) {
    return tsl::errors::InvalidArgument("Imported XLA Root is not a tuple op");
  }

  if (op_->getNumOperands() !=
      hlo_module->entry_computation()->num_parameters()) {
    return tsl::errors::InvalidArgument(
        "Entry computation does not have equal number of parameters to op "
        "operands");
  }

  ModuleOp mlir_module = op_->getParentOfType<ModuleOp>();
  mlir::OpBuilder builder(op_);
  mlir::SymbolTable symbol_table(mlir_module);

  llvm::SmallVector<mlir::Value> arguments;
  for (int i = 0; i < op_->getNumOperands(); i++) {
    arguments.push_back(op_->getOperand(i));
  }

  // Ideally we could use the Function Importer but it increases compilation
  // time when we have a model with thousands of tf2xla op fallbacks. At time
  // of writing, this caused compilation time to be greater than 2x slower.
  // So we have to directly import these instructions.
  TF_ASSIGN_OR_RETURN(
      mlir::Value root_value,
      xla::HloFunctionImporter::ImportInstructions(
          *hlo_module->entry_computation(), arguments, symbol_table, &builder));

  mhlo::TupleOp root_tuple =
      mlir::dyn_cast_or_null<mhlo::TupleOp>(root_value.getDefiningOp());
  if (!root_tuple) {
    return tsl::errors::InvalidArgument(
        "Imported XLA Root Value is not a tuple op");
  }

  return root_tuple;
}

LogicalResult Tf2XlaRewriter::PrepareParams() {
  // XlaCompiler within the context is only used by the functional ops to
  // compile functions. We are not handling those at the moment so
  // XlaCompiler is not required.
  context_ = new tensorflow::XlaContext(/*compiler=*/nullptr, &xla_builder_,
                                        /*graph=*/nullptr);
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
  absl::Status status = step_container_->Create(
      device_->resource_manager(),
      tensorflow::XlaContext::kXlaContextResourceName, context_);
  if (!status.ok()) {
    return emitRemark(op_->getLoc())
           << "failed to create XlaContext resource: " << status.ToString();
  }
  params_.step_container = step_container_.get();

  absl::StatusOr<int64_t> version_or = tensorflow::GetTfGraphProducerVersion(
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
  auto ranked_ty = mlir::dyn_cast<RankedTensorType>(ty);
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding =
      mlir::dyn_cast_or_null<TypeExtensionsAttr>(ranked_ty.getEncoding());
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
    if (mlir::isa<SymbolRefAttr>(attr_value)) {
      return true;
    } else if (auto array_attr = mlir::dyn_cast<ArrayAttr>(attr_value)) {
      if (!array_attr.empty() &&
          mlir::isa<SymbolRefAttr>(*array_attr.begin())) {
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
    auto ranked_ty = mlir::dyn_cast<ShapedType>(ty);
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
      op_, name_mapper_->GetUniqueName(op_),
      /*ignore_unregistered_attrs=*/true);
  if (!nodedef_or.ok()) {
    return op_->emitRemark() << "failed to convert op to NodeDef: "
                             << nodedef_or.status().ToString();
  }

  if (failed(PrepareParams())) return failure();

  std::shared_ptr<const tensorflow::NodeProperties> props;
  absl::Status status = tensorflow::NodeProperties::CreateFromNodeDef(
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

  tensorflow::OpKernelContext op_context(&params_, op_->getNumResults());
  device_->Compute(params_.op_kernel, &op_context);

  status = op_context.status();
  if (!status.ok()) {
    return op_->emitRemark()
           << "compilation to HLO failed: " << status.ToString();
  }

  if (failed(VerifyOpResults(op_context))) return failure();

  absl::StatusOr<mhlo::TupleOp> tuple_result_or_status =
      CompileWithHloImporter(op_context);
  if (!tuple_result_or_status.ok()) {
    return op_->emitRemark() << tuple_result_or_status.status().ToString();
  }
    mhlo::TupleOp tuple_result = tuple_result_or_status.value();

    llvm::SmallVector<Value> output_values;
    if (failed(GetKernelOutputs(op_context, tuple_result, output_values))) {
      return failure();
    }

  rewriter_.replaceOp(op_, output_values);
  return success();
}

absl::StatusOr<mhlo::TupleOp> Tf2XlaRewriter::CompileWithHloImporter(
    tensorflow::OpKernelContext& op_context) {
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
    mhlo::TupleOp tuple_result, llvm::SmallVector<Value>& outputs) {
  if (tuple_result->getNumOperands() != op_->getNumResults()) {
    return op_->emitRemark() << "Translated TF2XLA tuple has different "
                                "number of results than original op";
  }

  for (int i = 0; i < tuple_result->getNumOperands(); i++) {
    outputs.push_back(tuple_result->getOperand(i));
  }

  tuple_result.getOperation()->erase();
  return success();
}

mlir::LogicalResult Tf2XlaRewriter::GetKernelOutputs(
    tensorflow::OpKernelContext& op_context, mhlo::TupleOp tuple_results,
    llvm::SmallVector<Value>& outputs) {
  outputs.reserve(op_->getNumResults());

  return UnpackTupleResults(tuple_results, outputs);
}

tensorflow::XlaExpression Tf2XlaRewriter::GetExprForOperand(
    Value operand, Operation* op, int64_t operand_index) {
  ElementsAttr const_attr;
  auto defining_op = operand.getDefiningOp();

  ::xla::XlaOp xla_op = xla::Parameter(&xla_builder_, operand_index,
                                       xla::TypeToShape(operand.getType()),
                                       std::to_string(operand_index));

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
