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
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
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
#include "tensorflow/core/platform/errors.h"

namespace mlir {
namespace TF {
using tensorflow::internal::AbstractFunction;
using tensorflow::internal::AbstractOp;
using tensorflow::internal::AbstractTensor;
using tensorflow::internal::dyncast;
using tensorflow::internal::ExecutionContext;
using tensorflow::internal::OutputList;

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

class MlirTensor : public AbstractTensor {
 public:
  explicit MlirTensor(Value value) : AbstractTensor(kKind), value_(value) {}

  Value getValue() { return value_; }

  static constexpr AbstractTensorKind kKind = kMlirTensor;

 private:
  Value value_;
};

class MlirAbstractOp : public AbstractOp {
 public:
  explicit MlirAbstractOp(MLIRContext* context)
      : AbstractOp(kKind), context_(context) {}

  void SetOpType(const char* op_type, TF_Status* s) override;

  void SetAttrType(const char* attr_name, TF_DataType dtype,
                   TF_Status* s) override;

  void SetOpName(const char* const op_name, TF_Status* s) override;

  MLIRContext* GetContext() { return context_; }

  Type AddRef(Type type, TF_Status* s);

  OperationState* Create(ArrayRef<Value> operands, TF_Status* s);

  static constexpr AbstractOpKind kKind = kMlirOp;

 private:
  MLIRContext* context_;
  llvm::StringMap<Attribute> attrs_;
  std::unique_ptr<OperationState> state_;
  const char* op_name_ = nullptr;
};

// MlirFunction is a thin wrapper over a FuncOp.
class MlirFunction : public AbstractFunction {
 public:
  explicit MlirFunction(std::unique_ptr<MLIRContext> context,
                        OwningModuleRef module, FuncOp func)
      : AbstractFunction(kKind),
        context_(std::move(context)),
        module_(std::move(module)),
        func_(func) {}

  TF_Function* GetTfFunction(TF_Status* s) override;

  static constexpr AbstractFunctionKind kKind = kGraphFunc;

 private:
  std::unique_ptr<MLIRContext> context_;
  OwningModuleRef module_;
  FuncOp func_;
};

class MlirFunctionContext : public ExecutionContext {
 public:
  explicit MlirFunctionContext(const char* name)
      : ExecutionContext(kKind),
        context_(std::make_unique<MLIRContext>()),
        builder_(context_.get()) {
    // TODO(aminim) figure out the location story here
    module_ = ModuleOp::create(builder_.getUnknownLoc());
    func_ = FuncOp::create(builder_.getUnknownLoc(), name,
                           builder_.getFunctionType(llvm::None, llvm::None));
    module_->push_back(func_);
    builder_ = OpBuilder::atBlockBegin(func_.addEntryBlock());
  }

  AbstractOp* CreateOperation() override {
    return new MlirAbstractOp(context_.get());
  }

  void ExecuteOperation(AbstractOp* abstract_op, int num_inputs,
                        AbstractTensor* const* inputs, OutputList* o,
                        TF_Status* s) override;

  AbstractTensor* AddParameter(TF_DataType dtype, TF_Status* s) override;

  AbstractFunction* Finalize(OutputList* outputs, TF_Status* s) override;

  void RegisterFunction(AbstractFunction* func, TF_Status* s) override {
    s->status = tensorflow::errors::Unimplemented(
        "Registering graph functions has not been implemented yet.");
  }

  static constexpr ExecutionContextKind kKind = kMlirContext;

 private:
  std::unique_ptr<MLIRContext> context_;
  OpBuilder builder_;
  FuncOp func_;
  OwningModuleRef module_;
};

void MlirAbstractOp::SetOpType(const char* op_type, TF_Status* s) {
  if (state_) {
    s->status = tensorflow::errors::FailedPrecondition(
        "SetOpType called on already built op.");
    return;
  }
  std::string name = "tf.";
  name += op_type;
  // TODO(aminim) figure out the location story here
  state_ = std::make_unique<OperationState>(UnknownLoc::get(context_), name);
}

void MlirAbstractOp::SetAttrType(const char* attr_name, TF_DataType dtype,
                                 TF_Status* s) {
  if (!state_) {
    s->status = tensorflow::errors::FailedPrecondition(
        "op_type must be specified before specifying attrs.");
    return;
  }
  Type mlir_type;
  Builder builder(context_);
  s->status = ConvertDataTypeToTensor(static_cast<tensorflow::DataType>(dtype),
                                      builder, &mlir_type);
  if (!s->status.ok()) return;
  attrs_[attr_name] = TypeAttr::get(mlir_type);
}

void MlirAbstractOp::SetOpName(const char* const op_name, TF_Status* s) {
  // TODO(aminim): should we use a location?
  if (op_name_) {
    s->status = tensorflow::errors::FailedPrecondition(
        "SetOpName called on already built op.");
    return;
  }
  op_name_ = op_name;
}

Type MlirAbstractOp::AddRef(Type type, TF_Status* s) {
  Type elt_type = getElementTypeOrSelf(type);
  if (elt_type.isa<mlir::TF::TensorFlowRefType>()) {
    s->status = tensorflow::errors::InvalidArgument(
        "Requested reference to a reference type");
    return nullptr;
  }
  elt_type = TensorFlowRefType::get(elt_type);
  if (RankedTensorType tensor_type = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(tensor_type.getShape(), elt_type);
  }
  return UnrankedTensorType::get(elt_type);
}

OperationState* MlirAbstractOp::Create(ArrayRef<Value> operands, TF_Status* s) {
  state_->operands = llvm::to_vector<4>(operands);
  const tensorflow::OpDef* op_def;
  auto node_name = state_->name.getStringRef().drop_front(
      TensorFlowDialect::getDialectNamespace().size() + 1);
  s->status =
      tensorflow::OpRegistry::Global()->LookUpOpDef(node_name.str(), &op_def);
  if (!s->status.ok()) return nullptr;
  Builder builder(context_);
  // Process operands according to the op_def and infer derived attributes.
  int current_operand = 0;
  for (const tensorflow::OpDef::ArgDef& input_arg : op_def->input_arg()) {
    if (!input_arg.number_attr().empty()) {
      // TODO(b/156122856): we don't support variadic operands.
      s->status = tensorflow::errors::Unimplemented(
          "Unsupported 'number_attr' for '", input_arg.number_attr(), "'");
      return nullptr;
    } else if (!input_arg.type_list_attr().empty()) {
      s->status = tensorflow::errors::InvalidArgument(
          "Unsupported 'type_list_attr' for '", input_arg.number_attr(), "'");
      return nullptr;
    }
    if (current_operand >= operands.size()) {
      s->status = tensorflow::errors::InvalidArgument("Missing operand for '",
                                                      input_arg.name(), "'");
      return nullptr;
    }
    Type expected_type;
    if (input_arg.type() != tensorflow::DT_INVALID) {
      s->status =
          ConvertDataTypeToTensor(input_arg.type(), builder, &expected_type);
      if (!s->status.ok()) return nullptr;
      if (input_arg.is_ref()) expected_type = AddRef(expected_type, s);
      if (!s->status.ok()) return nullptr;
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
        s->status = tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.number_attr(),
            "' required for output list '", output_arg.name(), "'");
        return nullptr;
      }
      if (!repeats_attr.isa<IntegerAttr>()) {
        s->status = tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.number_attr(),
            "' required for output list '", output_arg.name(),
            "' isn't an integer");
        return nullptr;
      }
      int64_t repeats = repeats_attr.cast<IntegerAttr>().getInt();

      if (!output_arg.type_attr().empty()) {
        // Same type repeated "repeats" times.
        Attribute attr = attrs_[output_arg.type_attr()];
        if (!attr) {
          s->status = tensorflow::errors::InvalidArgument(
              "Missing attribute '", output_arg.type_attr(),
              "' required for output '", output_arg.name(), "'");
          return nullptr;
        }
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr) {
          s->status = tensorflow::errors::InvalidArgument(
              "Attribute '", output_arg.type_attr(), "' required for output '",
              output_arg.name(), "' isn't a type attribute");
          return nullptr;
        }
        for (int i = 0; i < repeats; ++i)
          state_->types.push_back(type_attr.getType());
      } else if (output_arg.type() != tensorflow::DT_INVALID) {
        for (int i = 0; i < repeats; ++i) {
          Type type;
          s->status =
              ConvertDataTypeToTensor(output_arg.type(), builder, &type);
          if (!s->status.ok()) return nullptr;
          state_->types.push_back(type);
        }
      } else {
        s->status = tensorflow::errors::InvalidArgument(
            "Missing type or type_attr field in ",
            output_arg.ShortDebugString());
        return nullptr;
      }
    } else if (!output_arg.type_attr().empty()) {
      Attribute attr = attrs_[output_arg.type_attr()];
      if (!attr) {
        s->status = tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.type_attr(),
            "' required for output '", output_arg.name(), "'");
        return nullptr;
      }
      TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
      if (!type_attr) {
        s->status = tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.type_attr(), "' required for output '",
            output_arg.name(), "' isn't a type attribute");
        return nullptr;
      }
      state_->types.push_back(type_attr.getValue());
    } else if (!output_arg.type_list_attr().empty()) {
      // This is pointing to an attribute which is an array of types.
      Attribute attr = attrs_[output_arg.type_list_attr()];
      if (!attr) {
        s->status = tensorflow::errors::InvalidArgument(
            "Missing attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(), "'");
        return nullptr;
      }
      ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>();
      if (!array_attr) {
        s->status = tensorflow::errors::InvalidArgument(
            "Attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(),
            "' isn't an array attribute");
        return nullptr;
      }
      for (Attribute attr : array_attr) {
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr) {
          s->status = tensorflow::errors::InvalidArgument(
              "Array Attribute '", output_arg.type_list_attr(),
              "' required for output '", output_arg.name(),
              "' has a non-Type element");
          return nullptr;
        }
        state_->types.push_back(type_attr.getValue());
      }
    } else if (output_arg.type() != tensorflow::DT_INVALID) {
      Type type;
      Builder builder(context_);
      s->status = ConvertDataTypeToTensor(output_arg.type(), builder, &type);
      if (!s->status.ok()) return nullptr;
      state_->types.push_back(type);
    } else {
      s->status = tensorflow::errors::InvalidArgument(
          "No type fields in ", output_arg.ShortDebugString());
      if (!s->status.ok()) return nullptr;
    }
    if (output_arg.is_ref()) {
      // For all types that were added by this function call, make them refs.
      for (Type& type : llvm::make_range(&state_->types[original_size],
                                         state_->types.end())) {
        type = AddRef(type, s);
        if (!s->status.ok()) return nullptr;
      }
    }
  }
  return state_.get();
}

TF_Function* MlirFunction::GetTfFunction(TF_Status* s) {
  PassManager pm(func_.getContext());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addNestedPass<FuncOp>(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to
  // the MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(func_.getContext());
  LogicalResult result = pm.run(func_.getParentOfType<ModuleOp>());
  (void)result;
  s->status = diag_handler.ConsumeStatus();
  if (!s->status.ok()) return nullptr;

  tensorflow::GraphExportConfig configs;
  std::unique_ptr<TF_Function> tf_function(new TF_Function);
  s->status = ConvertMlirFunctionToFunctionLibraryDef(func_, configs,
                                                      &tf_function->fdef);
  return tf_function.release();
}

void MlirFunctionContext::ExecuteOperation(AbstractOp* abstract_op,
                                           int num_inputs,
                                           AbstractTensor* const* inputs,
                                           OutputList* o, TF_Status* s) {
  auto* mlir_op = dyncast<MlirAbstractOp>(abstract_op);
  if (mlir_op == nullptr) {
    s->status = tensorflow::errors::InvalidArgument(
        "Unable to cast AbstractOp to TF_GraphOp.");
    return;
  }
  SmallVector<Value, 8> operands;
  for (int i = 0; i < num_inputs; ++i) {
    auto* operand = dyncast<MlirTensor>(inputs[i]);
    if (!operand) {
      s->status = tensorflow::errors::InvalidArgument(
          "Capturing eager tensors is not supported yet.");
      return;
    }
    if (operand->getValue().getContext() != context_.get()) {
      s->status = tensorflow::errors::InvalidArgument(
          "Capturing tensors from other context is not supported.");
      return;
    }
    operands.push_back(operand->getValue());
  }
  OperationState* state = mlir_op->Create(operands, s);
  if (!s->status.ok() || !state) return;
  Operation* op = builder_.createOperation(*state);
  int num_results = op->getNumResults();
  o->outputs.clear();
  o->outputs.reserve(num_results);
  for (Value result : op->getResults())
    o->outputs.push_back(new MlirTensor(result));
}

AbstractTensor* MlirFunctionContext::AddParameter(TF_DataType dtype,
                                                  TF_Status* s) {
  Type type;
  s->status = ConvertDataTypeToTensor(static_cast<tensorflow::DataType>(dtype),
                                      builder_, &type);
  if (!s->status.ok()) return nullptr;
  return new MlirTensor(func_.getBody().front().addArgument(type));
}

AbstractFunction* MlirFunctionContext::Finalize(OutputList* outputs,
                                                TF_Status* s) {
  Block& body = func_.getBody().front();
  SmallVector<Value, 8> ret_operands;
  for (AbstractTensor* output : outputs->outputs) {
    auto* operand = dyncast<MlirTensor>(output);
    if (!operand) {
      s->status = tensorflow::errors::InvalidArgument(
          "Capturing eager tensors is not supported yet.");
      return nullptr;
    }
    if (operand->getValue().getContext() != context_.get()) {
      s->status = tensorflow::errors::InvalidArgument(
          "Capturing tensors from other context is not supported.");
      return nullptr;
    }
    ret_operands.push_back(operand->getValue());
  }
  builder_.create<ReturnOp>(func_.getLoc(), ret_operands);

  auto arg_types = llvm::to_vector<8>(body.getArgumentTypes());
  auto result_types =
      llvm::to_vector<8>(body.getTerminator()->getOperandTypes());
  func_.setType(FunctionType::get(arg_types, result_types, func_.getContext()));
  return new MlirFunction(std::move(context_), std::move(module_), func_);
}

extern "C" {
ExecutionContext* MlirTracingFactory(const char* fn_name, TF_Status* s) {
  RegisterDialects();
  return new MlirFunctionContext(fn_name);
}
}

}  // end anonymous namespace
}  // end namespace TF
}  // end namespace mlir
