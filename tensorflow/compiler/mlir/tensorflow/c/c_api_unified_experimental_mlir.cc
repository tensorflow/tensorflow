/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace mlir {
namespace TF {
using tensorflow::AbstractFunction;
using tensorflow::AbstractOperation;
using tensorflow::AbstractTensorHandle;
using tensorflow::AbstractTensorInterface;
using tensorflow::dyn_cast;
using tensorflow::OutputList;
using tensorflow::string;
using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

namespace {

static void RegisterDialects() {
  static bool init_once = []() {
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::tf_device::TensorFlowDeviceDialect>();
    mlir::registerDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
    mlir::registerDialect<mlir::TF::TensorFlowDialect>();
    return true;
  }();
  (void)init_once;
}

Status ConvertDataTypeToTensor(tensorflow::DataType dtype, Builder builder,
                               Type* type) {
  Status s = tensorflow::ConvertDataType(dtype, builder, type);
  if (s.ok()) *type = UnrankedTensorType::get(*type);
  return s;
}

class MlirTensor : public TracingTensorHandle {
 public:
  explicit MlirTensor(Value value)
      : TracingTensorHandle(kMlir), value_(value) {}

  void Release() override { delete this; }

  Value getValue() { return value_; }

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kMlir;
  }

 private:
  Value value_;
};

class MlirFunctionContext;

class MlirAbstractOp : public TracingOperation {
 public:
  explicit MlirAbstractOp(MLIRContext* context,
                          MlirFunctionContext* function_context)
      : TracingOperation(kMlir),
        context_(context),
        function_context_(function_context) {}

  void Release() override { delete this; }

  Status Reset(const char* op, const char* raw_device_name) override;

  const string& Name() const override;

  const string& DeviceName() const override;

  Status SetDeviceName(const char* name) override;

  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle*> inputs) override;
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name,
                     tensorflow::DataType dtype) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* value,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name,
                         const tensorflow::DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;

  Status SetOpName(const char* const op_name) override;

  MLIRContext* GetContext() { return context_; }

  Status AddRef(Type type, Type* output_type);

  Status Create(ArrayRef<Value> operands, OperationState**);

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
    return ptr->getKind() == kMlir;
  }

 private:
  MLIRContext* context_;
  MlirFunctionContext* function_context_;
  SmallVector<Value, 8> operands_;
  llvm::StringMap<Attribute> attrs_;
  std::unique_ptr<OperationState> state_;
  const char* op_name_ = nullptr;
  string tf_op_type_;
  // TODO(srbs): Use this.
  string device_name_;
};

// MlirFunction is a thin wrapper over a FuncOp.
class MlirFunction : public AbstractFunction {
 public:
  explicit MlirFunction(std::unique_ptr<MLIRContext> context,
                        OwningModuleRef module, FuncOp func)
      : AbstractFunction(kMlir),
        context_(std::move(context)),
        module_(std::move(module)),
        func_(func) {}

  Status GetFunctionDef(tensorflow::FunctionDef** f) override;

  // For LLVM style RTTI.
  static bool classof(const AbstractFunction* ptr) {
    return ptr->getKind() == kMlir;
  }

 private:
  std::unique_ptr<MLIRContext> context_;
  OwningModuleRef module_;
  FuncOp func_;
};

class MlirFunctionContext : public TracingContext {
 public:
  explicit MlirFunctionContext(const char* name)
      : TracingContext(kMlir),
        context_(std::make_unique<MLIRContext>()),
        builder_(context_.get()) {
    // TODO(aminim) figure out the location story here
    module_ = ModuleOp::create(builder_.getUnknownLoc());
    func_ = FuncOp::create(builder_.getUnknownLoc(), name,
                           builder_.getFunctionType(llvm::None, llvm::None));
    module_->push_back(func_);
    builder_ = OpBuilder::atBlockBegin(func_.addEntryBlock());
  }

  void Release() override { delete this; }

  AbstractOperation* CreateOperation() override {
    return new MlirAbstractOp(context_.get(), this);
  }
  Status AddParameter(tensorflow::DataType dtype,
                      TracingTensorHandle** handle) override;

  Status Finalize(OutputList* outputs, AbstractFunction** f) override;

  Status RegisterFunction(AbstractFunction* func) override {
    return tensorflow::errors::Unimplemented(
        "Registering graph functions has not been implemented yet.");
  }

  Status RemoveFunction(const string& func) override {
    return tensorflow::errors::Unimplemented(
        "MlirFunctionContext::RemoveFunction has not been implemented yet.");
  }

  Operation* CreateOperationFromState(const OperationState& state);

 private:
  std::unique_ptr<MLIRContext> context_;
  OpBuilder builder_;
  FuncOp func_;
  OwningModuleRef module_;
};

Status MlirAbstractOp::Reset(const char* op, const char* device_name) {
  if (state_) {
    return tensorflow::errors::FailedPrecondition(
        "Reset called on already built op.");
  }
  tf_op_type_ = op;
  std::string name = "tf.";
  name += op;
  // TODO(aminim) figure out the location story here
  state_ = std::make_unique<OperationState>(UnknownLoc::get(context_), name);
  return Status::OK();
}

Status MlirAbstractOp::SetAttrType(const char* attr_name,
                                   tensorflow::DataType dtype) {
  if (!state_) {
    return Status(tensorflow::error::Code::FAILED_PRECONDITION,
                  "op_type must be specified before specifying attrs.");
  }
  Type mlir_type;
  Builder builder(context_);
  TF_RETURN_IF_ERROR(ConvertDataTypeToTensor(dtype, builder, &mlir_type));
  attrs_[attr_name] = TypeAttr::get(mlir_type);
  return Status::OK();
}

Status MlirAbstractOp::SetOpName(const char* const op_name) {
  // TODO(aminim): should we use a location?
  if (op_name_) {
    return tensorflow::errors::FailedPrecondition(
        "SetOpName called on already built op.");
  }
  op_name_ = op_name;
  return Status::OK();
}

Status MlirAbstractOp::AddRef(Type type, Type* output_type) {
  Type elt_type = getElementTypeOrSelf(type);
  if (elt_type.isa<mlir::TF::TensorFlowRefType>()) {
    return tensorflow::errors::InvalidArgument(
        "Requested reference to a reference type");
  }
  elt_type = TensorFlowRefType::get(elt_type);
  if (RankedTensorType tensor_type = type.dyn_cast<RankedTensorType>()) {
    *output_type = RankedTensorType::get(tensor_type.getShape(), elt_type);
  }
  *output_type = UnrankedTensorType::get(elt_type);
  return Status::OK();
}

Status MlirAbstractOp::Create(ArrayRef<Value> operands,
                              OperationState** state) {
  state_->operands = llvm::to_vector<4>(operands);
  const tensorflow::OpDef* op_def;
  auto node_name = state_->name.getStringRef().drop_front(
      TensorFlowDialect::getDialectNamespace().size() + 1);
  TF_RETURN_IF_ERROR(
      tensorflow::OpRegistry::Global()->LookUpOpDef(node_name.str(), &op_def));
  Builder builder(context_);
  // Process operands according to the op_def and infer derived attributes.
  int current_operand = 0;
  for (const tensorflow::OpDef::ArgDef& input_arg : op_def->input_arg()) {
    if (!input_arg.number_attr().empty()) {
      // TODO(b/156122856): we don't support variadic operands.
      return tensorflow::errors::Unimplemented(
          "Unsupported 'number_attr' for '", input_arg.number_attr(), "'");
    } else if (!input_arg.type_list_attr().empty()) {
      return tensorflow::errors::InvalidArgument(
          "Unsupported 'type_list_attr' for '", input_arg.number_attr(), "'");
    }
    if (current_operand >= operands.size()) {
      return tensorflow::errors::InvalidArgument("Missing operand for '",
                                                 input_arg.name(), "'");
    }
    Type expected_type;
    if (input_arg.type() != tensorflow::DT_INVALID) {
      TF_RETURN_IF_ERROR(
          ConvertDataTypeToTensor(input_arg.type(), builder, &expected_type));
      Type output_type;
      if (input_arg.is_ref())
        TF_RETURN_IF_ERROR(AddRef(expected_type, &output_type));
      expected_type = output_type;
    } else {
      expected_type = operands[current_operand].getType();
    }
    if (!input_arg.type_attr().empty()) {
      attrs_[input_arg.type_attr()] = TypeAttr::get(expected_type);
    }
    ++current_operand;
  }

  for (const tensorflow::OpDef::ArgDef& output_arg : op_def->output_arg()) {
    int original_size = state_->types.size();
    if (!output_arg.number_attr().empty()) {
      // Same type repeated "repeats" times.
      Attribute repeats_attr = attrs_[output_arg.number_attr()];
      if (!repeats_attr) {
        return tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.number_attr(),
            "' required for output list '", output_arg.name(), "'");
      }
      if (!repeats_attr.isa<IntegerAttr>()) {
        return tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.number_attr(),
            "' required for output list '", output_arg.name(),
            "' isn't an integer");
      }
      int64_t repeats = repeats_attr.cast<IntegerAttr>().getInt();

      if (!output_arg.type_attr().empty()) {
        // Same type repeated "repeats" times.
        Attribute attr = attrs_[output_arg.type_attr()];
        if (!attr) {
          return tensorflow::errors::InvalidArgument(
              "Missing attribute '", output_arg.type_attr(),
              "' required for output '", output_arg.name(), "'");
        }
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr) {
          return tensorflow::errors::InvalidArgument(
              "Attribute '", output_arg.type_attr(), "' required for output '",
              output_arg.name(), "' isn't a type attribute");
        }
        for (int i = 0; i < repeats; ++i)
          state_->types.push_back(type_attr.getType());
      } else if (output_arg.type() != tensorflow::DT_INVALID) {
        for (int i = 0; i < repeats; ++i) {
          Type type;
          TF_RETURN_IF_ERROR(
              ConvertDataTypeToTensor(output_arg.type(), builder, &type));
          state_->types.push_back(type);
        }
      } else {
        return tensorflow::errors::InvalidArgument(
            "Missing type or type_attr field in ",
            output_arg.ShortDebugString());
      }
    } else if (!output_arg.type_attr().empty()) {
      Attribute attr = attrs_[output_arg.type_attr()];
      if (!attr) {
        return tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.type_attr(),
            "' required for output '", output_arg.name(), "'");
      }
      TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
      if (!type_attr) {
        return tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.type_attr(), "' required for output '",
            output_arg.name(), "' isn't a type attribute");
      }
      state_->types.push_back(type_attr.getValue());
    } else if (!output_arg.type_list_attr().empty()) {
      // This is pointing to an attribute which is an array of types.
      Attribute attr = attrs_[output_arg.type_list_attr()];
      if (!attr) {
        return tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(), "'");
      }
      ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>();
      if (!array_attr) {
        return tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(),
            "' isn't an array attribute");
      }
      for (Attribute attr : array_attr) {
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr) {
          return tensorflow::errors::InvalidArgument(
              "Array Attribute '", output_arg.type_list_attr(),
              "' required for output '", output_arg.name(),
              "' has a non-Type element");
        }
        state_->types.push_back(type_attr.getValue());
      }
    } else if (output_arg.type() != tensorflow::DT_INVALID) {
      Type type;
      Builder builder(context_);
      TF_RETURN_IF_ERROR(
          ConvertDataTypeToTensor(output_arg.type(), builder, &type));
      state_->types.push_back(type);
    } else {
      return tensorflow::errors::InvalidArgument("No type fields in ",
                                                 output_arg.ShortDebugString());
    }
    if (output_arg.is_ref()) {
      // For all types that were added by this function call, make them refs.
      for (Type& type : llvm::make_range(&state_->types[original_size],
                                         state_->types.end())) {
        Type output_type;
        TF_RETURN_IF_ERROR(AddRef(type, &output_type));
        type = output_type;
      }
    }
  }
  *state = state_.get();
  return Status::OK();
}

const string& MlirAbstractOp::Name() const { return tf_op_type_; }

const string& MlirAbstractOp::DeviceName() const { return device_name_; }

Status MlirAbstractOp::SetDeviceName(const char* name) {
  device_name_ = name;
  return Status::OK();
}

Status MlirAbstractOp::AddInputList(absl::Span<AbstractTensorHandle*> inputs) {
  return tensorflow::errors::Unimplemented(
      "AddInputList has not been implemented yet.");
}

Status MlirAbstractOp::SetAttrString(const char* attr_name, const char* data,
                                     size_t length) {
  return tensorflow::errors::Unimplemented(
      "SetAttrString has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrInt(const char* attr_name, int64_t value) {
  return tensorflow::errors::Unimplemented(
      "SetAttrInt has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloat(const char* attr_name, float value) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFloat has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBool(const char* attr_name, bool value) {
  return tensorflow::errors::Unimplemented(
      "SetAttrBool has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrShape(const char* attr_name, const int64_t* dims,
                                    const int num_dims) {
  return tensorflow::errors::Unimplemented(
      "SetAttrShape has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunction(const char* attr_name,
                                       const AbstractOperation* value) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunction has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionName(const char* attr_name,
                                           const char* value, size_t length) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionName has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTensor(const char* attr_name,
                                     AbstractTensorInterface* tensor) {
  return tensorflow::errors::Unimplemented(
      "SetAttrTensor has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrStringList(const char* attr_name,
                                         const void* const* values,
                                         const size_t* lengths,
                                         int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrStringList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloatList(const char* attr_name,
                                        const float* values, int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFloatList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrIntList(const char* attr_name,
                                      const int64_t* values, int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrIntList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTypeList(const char* attr_name,
                                       const tensorflow::DataType* values,
                                       int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrTypeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBoolList(const char* attr_name,
                                       const unsigned char* values,
                                       int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrBoolList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrShapeList(const char* attr_name,
                                        const int64_t** dims,
                                        const int* num_dims, int num_values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrShapeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionList has not been implemented yet.");
}

Status MlirFunction::GetFunctionDef(tensorflow::FunctionDef** f) {
  PassManager pm(func_.getContext());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addNestedPass<FuncOp>(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to
  // the MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(func_.getContext());
  LogicalResult result = pm.run(func_.getParentOfType<ModuleOp>());
  (void)result;
  TF_RETURN_IF_ERROR(diag_handler.ConsumeStatus());

  tensorflow::GraphExportConfig configs;
  *f = new tensorflow::FunctionDef();
  return ConvertMlirFunctionToFunctionLibraryDef(func_, configs, *f);
}

Status MlirAbstractOp::Execute(absl::Span<AbstractTensorHandle*> retvals,
                               int* num_retvals) {
  OperationState* state;
  TF_RETURN_IF_ERROR(Create(operands_, &state));
  Operation* op = function_context_->CreateOperationFromState(*state);
  *num_retvals = op->getNumResults();
  for (int i = 0; i < *num_retvals; i++)
    retvals[i] = new MlirTensor(op->getResult(i));
  return Status::OK();
}

Operation* MlirFunctionContext::CreateOperationFromState(
    const OperationState& state) {
  return builder_.createOperation(state);
}

Status MlirFunctionContext::AddParameter(tensorflow::DataType dtype,
                                         TracingTensorHandle** handle) {
  Type type;
  TF_RETURN_IF_ERROR(ConvertDataTypeToTensor(dtype, builder_, &type));
  *handle = new MlirTensor(func_.getBody().front().addArgument(type));
  return Status::OK();
}

Status MlirAbstractOp::AddInput(AbstractTensorHandle* input) {
  auto* operand = dyn_cast<MlirTensor>(input);
  if (!operand) {
    return tensorflow::errors::InvalidArgument(
        "Unable to cast input to MlirTensor");
  }
  operands_.push_back(operand->getValue());
  return Status::OK();
}
Status MlirFunctionContext::Finalize(OutputList* outputs,
                                     AbstractFunction** f) {
  Block& body = func_.getBody().front();
  SmallVector<Value, 8> ret_operands;
  for (auto* output : outputs->outputs) {
    auto* operand = dyn_cast<MlirTensor>(output);
    if (!operand) {
      return tensorflow::errors::InvalidArgument(
          "Capturing eager tensors is not supported yet.");
    }
    if (operand->getValue().getContext() != context_.get()) {
      return tensorflow::errors::InvalidArgument(
          "Capturing tensors from other context is not supported.");
    }
    ret_operands.push_back(operand->getValue());
  }
  builder_.create<ReturnOp>(func_.getLoc(), ret_operands);

  auto arg_types = llvm::to_vector<8>(body.getArgumentTypes());
  auto result_types =
      llvm::to_vector<8>(body.getTerminator()->getOperandTypes());
  func_.setType(FunctionType::get(arg_types, result_types, func_.getContext()));
  *f = new MlirFunction(std::move(context_), std::move(module_), func_);
  return Status::OK();
}

extern "C" {
TracingContext* MlirTracingFactory(const char* fn_name, TF_Status* s) {
  RegisterDialects();
  return new MlirFunctionContext(fn_name);
}
}

}  // namespace
}  // namespace TF
}  // namespace mlir
