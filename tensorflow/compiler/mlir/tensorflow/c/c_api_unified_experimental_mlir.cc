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
#include <optional>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;
using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

namespace {

void RegisterDialects(mlir::MLIRContext& ctx) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
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

  tensorflow::DataType DataType() const override {
    tensorflow::DataType type;
    Status s = ConvertToDataType(value_.getType(), &type);
    if (!s.ok()) {
      return tensorflow::DT_INVALID;
    }
    return type;
  }

  tensorflow::Status Shape(
      tensorflow::PartialTensorShape* shape) const override {
    // TODO(b/173074167): Implement this and enable tests in
    // unified_api_test.cc.
    return Unimplemented("MlirTensor::Shape is not implemented yet.");
  }

  Value getValue() { return value_; }
  Type getElementType() {
    return value_.getType().cast<ShapedType>().getElementType();
  }

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kMlir;
  }

  // Return default (TFT_UNSET) full type information. This could be updated in
  // the future if full type information is needed.
  tensorflow::FullTypeDef FullType() const override {
    return tensorflow::FullTypeDef();
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
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
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
  // Return true is there are still unfilled ODS slots for adding more inputs.
  bool IsNextODSArgAvailable();

  MLIRContext* context_;
  MlirFunctionContext* function_context_;
  SmallVector<Value, 8> operands_;
  llvm::StringMap<Attribute> attrs_;
  std::unique_ptr<OperationState> state_;
  // This is the index of the next ODS operand that will be added with AddInput
  // or AddInput;
  int current_ods_input_ = 0;
  const tensorflow::OpDef* op_def_ = nullptr;
  const char* op_name_ = nullptr;
  string tf_op_type_;
  // TODO(srbs): Use this.
  string device_name_;
};

// MlirFunction is a thin wrapper over a FuncOp.
class MlirFunction : public AbstractFunction {
 public:
  explicit MlirFunction(std::unique_ptr<MLIRContext> context,
                        OwningOpRef<mlir::ModuleOp> module, func::FuncOp func)
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
  OwningOpRef<mlir::ModuleOp> module_;
  func::FuncOp func_;
  std::unique_ptr<tensorflow::FunctionDef> fdef_;
};

class MlirFunctionContext : public TracingContext {
 public:
  explicit MlirFunctionContext(const char* name)
      : TracingContext(kMlir),
        context_(std::make_unique<MLIRContext>()),
        builder_(context_.get()) {
    RegisterDialects(*context_);
    // TODO(aminim) figure out the location story here
    module_ = ModuleOp::create(builder_.getUnknownLoc());
    func_ = func::FuncOp::create(
        builder_.getUnknownLoc(), name,
        builder_.getFunctionType(std::nullopt, std::nullopt));
    module_->push_back(func_);
    builder_ = OpBuilder::atBlockBegin(func_.addEntryBlock());
  }

  void Release() override { delete this; }

  AbstractOperation* CreateOperation() override {
    return new MlirAbstractOp(context_.get(), this);
  }
  Status AddParameter(tensorflow::DataType dtype,
                      const tensorflow::PartialTensorShape& shape,
                      TracingTensorHandle** handle) override;

  Status Finalize(OutputList* outputs, AbstractFunction** f) override;

  Status RegisterFunction(AbstractFunction* func) override {
    return Unimplemented(
        "Registering graph functions has not been implemented yet.");
  }

  Status RemoveFunction(const string& func) override {
    return Unimplemented(
        "MlirFunctionContext::RemoveFunction has not been implemented yet.");
  }

  Operation* CreateOperationFromState(const OperationState& state);

 private:
  std::unique_ptr<MLIRContext> context_;
  OpBuilder builder_;
  func::FuncOp func_;
  OwningOpRef<mlir::ModuleOp> module_;
};

Status MlirAbstractOp::Reset(const char* op, const char* device_name) {
  if (state_) {
    return FailedPrecondition("Reset called on already built op.");
  }
  TF_RETURN_IF_ERROR(
      tensorflow::OpRegistry::Global()->LookUpOpDef(op, &op_def_));
  assert(op_def_);

  tf_op_type_ = op;
  std::string name = "tf.";
  name += op;
  // TODO(aminim) figure out the location story here
  state_ = std::make_unique<OperationState>(UnknownLoc::get(context_), name);
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::SetAttrType(const char* attr_name,
                                   tensorflow::DataType dtype) {
  if (!state_)
    return FailedPrecondition(
        "op_type must be specified before specifying attrs.");
  Type mlir_type;
  Builder builder(context_);
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &mlir_type));
  attrs_[attr_name] = TypeAttr::get(mlir_type);
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::SetOpName(const char* const op_name) {
  // TODO(aminim): should we use a location?
  if (op_name_) {
    return FailedPrecondition("SetOpName called on already built op.");
  }
  op_name_ = op_name;
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::AddRef(Type type, Type* output_type) {
  Type elt_type = getElementTypeOrSelf(type);
  if (elt_type.isa<mlir::TF::TensorFlowRefType>()) {
    return InvalidArgument("Requested reference to a reference type");
  }
  elt_type = TensorFlowRefType::get(elt_type);
  if (RankedTensorType tensor_type = type.dyn_cast<RankedTensorType>()) {
    *output_type = RankedTensorType::get(tensor_type.getShape(), elt_type);
  }
  *output_type = UnrankedTensorType::get(elt_type);
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::Create(ArrayRef<Value> operands,
                              OperationState** state) {
  state_->operands = llvm::to_vector<4>(operands);
  Builder builder(context_);

  if (current_ods_input_ != op_def_->input_arg_size())
    return InvalidArgument(absl::StrCat("Mismatch in operands number: got ",
                                        current_ods_input_, " expected ",
                                        op_def_->input_arg_size(), " ; for op ",
                                        state_->name.getStringRef().str()));

  // Process results according to the op_def and infer types for derived
  // attributes.
  for (const tensorflow::OpDef::ArgDef& output_arg : op_def_->output_arg()) {
    int original_size = state_->types.size();
    if (!output_arg.number_attr().empty()) {
      // Same type repeated "repeats" times.
      Attribute repeats_attr = attrs_[output_arg.number_attr()];
      if (!repeats_attr)
        return InvalidArgument("Missing attribute '", output_arg.number_attr(),
                               "' required for output list '",
                               output_arg.name(), "'");
      if (!repeats_attr.isa<IntegerAttr>())
        return InvalidArgument("Attribute '", output_arg.number_attr(),
                               "' required for output list '",
                               output_arg.name(), "' isn't an integer");
      int64_t repeats = repeats_attr.cast<IntegerAttr>().getInt();

      if (!output_arg.type_attr().empty()) {
        // Same type repeated "repeats" times.
        Attribute attr = attrs_[output_arg.type_attr()];
        if (!attr)
          return InvalidArgument("Missing attribute '", output_arg.type_attr(),
                                 "' required for output '", output_arg.name(),
                                 "'");
        TypedAttr type_attr = attr.dyn_cast<TypedAttr>();
        if (!type_attr)
          return InvalidArgument("Attribute '", output_arg.type_attr(),
                                 "' required for output '", output_arg.name(),
                                 "' isn't a type attribute");
        for (int i = 0; i < repeats; ++i)
          state_->types.push_back(UnrankedTensorType::get(type_attr.getType()));
      } else if (output_arg.type() != tensorflow::DT_INVALID) {
        for (int i = 0; i < repeats; ++i) {
          Type type;
          TF_RETURN_IF_ERROR(
              ConvertDataType(output_arg.type(), builder, &type));
          state_->types.push_back(type);
        }
      } else {
        return InvalidArgument("Missing type or type_attr field in ",
                               output_arg.ShortDebugString());
      }
    } else if (!output_arg.type_attr().empty()) {
      Attribute attr = attrs_[output_arg.type_attr()];
      if (!attr)
        return InvalidArgument("Missing attribute '", output_arg.type_attr(),
                               "' required for output '", output_arg.name(),
                               "'");
      TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
      if (!type_attr)
        return InvalidArgument("Attribute '", output_arg.type_attr(),
                               "' required for output '", output_arg.name(),
                               "' isn't a type attribute");
      state_->types.push_back(UnrankedTensorType::get(type_attr.getValue()));
    } else if (!output_arg.type_list_attr().empty()) {
      // This is pointing to an attribute which is an array of types.
      Attribute attr = attrs_[output_arg.type_list_attr()];
      if (!attr)
        return InvalidArgument(
            "Missing attribute '", output_arg.type_list_attr(),
            "' required for output '", output_arg.name(), "'");
      ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>();
      if (!array_attr)
        return InvalidArgument("Attribute '", output_arg.type_list_attr(),
                               "' required for output '", output_arg.name(),
                               "' isn't an array attribute");
      for (Attribute attr : array_attr) {
        TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
        if (!type_attr)
          return InvalidArgument("Array Attribute '",
                                 output_arg.type_list_attr(),
                                 "' required for output '", output_arg.name(),
                                 "' has a non-Type element");
        state_->types.push_back(UnrankedTensorType::get(type_attr.getValue()));
      }
    } else if (output_arg.type() != tensorflow::DT_INVALID) {
      Type type;
      Builder builder(context_);
      TF_RETURN_IF_ERROR(ConvertDataType(output_arg.type(), builder, &type));
      state_->types.push_back(type);
    } else {
      return InvalidArgument("No type fields in ",
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
  for (auto& it : attrs_) state_->addAttribute(it.first(), it.second);
  *state = state_.get();
  return ::tensorflow::OkStatus();
}

const string& MlirAbstractOp::Name() const { return tf_op_type_; }

const string& MlirAbstractOp::DeviceName() const { return device_name_; }

Status MlirAbstractOp::SetDeviceName(const char* name) {
  device_name_ = name;
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::SetAttrString(const char* attr_name, const char* data,
                                     size_t length) {
  return Unimplemented("SetAttrString has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrInt(const char* attr_name, int64_t value) {
  return Unimplemented("SetAttrInt has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloat(const char* attr_name, float value) {
  return Unimplemented("SetAttrFloat has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBool(const char* attr_name, bool value) {
  attrs_[attr_name] = BoolAttr::get(context_, value);
  return ::tensorflow::OkStatus();
}
Status MlirAbstractOp::SetAttrShape(const char* attr_name, const int64_t* dims,
                                    const int num_dims) {
  return Unimplemented("SetAttrShape has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunction(const char* attr_name,
                                       const AbstractOperation* value) {
  return Unimplemented("SetAttrFunction has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionName(const char* attr_name,
                                           const char* value, size_t length) {
  return Unimplemented("SetAttrFunctionName has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTensor(const char* attr_name,
                                     AbstractTensorInterface* tensor) {
  return Unimplemented("SetAttrTensor has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrStringList(const char* attr_name,
                                         const void* const* values,
                                         const size_t* lengths,
                                         int num_values) {
  return Unimplemented("SetAttrStringList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFloatList(const char* attr_name,
                                        const float* values, int num_values) {
  return Unimplemented("SetAttrFloatList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrIntList(const char* attr_name,
                                      const int64_t* values, int num_values) {
  return Unimplemented("SetAttrIntList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrTypeList(const char* attr_name,
                                       const tensorflow::DataType* values,
                                       int num_values) {
  return Unimplemented("SetAttrTypeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrBoolList(const char* attr_name,
                                       const unsigned char* values,
                                       int num_values) {
  return Unimplemented("SetAttrBoolList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrShapeList(const char* attr_name,
                                        const int64_t** dims,
                                        const int* num_dims, int num_values) {
  return Unimplemented("SetAttrShapeList has not been implemented yet.");
}
Status MlirAbstractOp::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  return Unimplemented("SetAttrFunctionList has not been implemented yet.");
}

Status MlirFunction::GetFunctionDef(tensorflow::FunctionDef** f) {
  if (fdef_) {
    *f = fdef_.get();
    return ::tensorflow::OkStatus();
  }
  PassManager pm(func_.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(pm);
  pm.addNestedPass<func::FuncOp>(
      CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to
  // the MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(func_.getContext());
  LogicalResult result = pm.run(func_->getParentOfType<ModuleOp>());
  (void)result;
  TF_RETURN_IF_ERROR(diag_handler.ConsumeStatus());

  tensorflow::GraphExportConfig configs;
  fdef_.reset(new tensorflow::FunctionDef());
  TF_RETURN_IF_ERROR(
      ConvertMlirFunctionToFunctionLibraryDef(func_, configs, fdef_.get()));
  *f = fdef_.get();
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::Execute(absl::Span<AbstractTensorHandle*> retvals,
                               int* num_retvals) {
  OperationState* state;
  TF_RETURN_IF_ERROR(Create(operands_, &state));
  Operation* op = function_context_->CreateOperationFromState(*state);
  *num_retvals = op->getNumResults();
  for (int i = 0; i < *num_retvals; i++)
    retvals[i] = new MlirTensor(op->getResult(i));
  return ::tensorflow::OkStatus();
}

Operation* MlirFunctionContext::CreateOperationFromState(
    const OperationState& state) {
  return builder_.create(state);
}

Status MlirFunctionContext::AddParameter(
    tensorflow::DataType dtype, const tensorflow::PartialTensorShape& shape,
    TracingTensorHandle** handle) {
  // TODO(b/173073199): Use shape. Enable tests in unified_api_test.cc once
  // resolved.
  Type type;
  TF_RETURN_IF_ERROR(ConvertDataTypeToTensor(dtype, builder_, &type));
  *handle =
      new MlirTensor(func_.getBody().front().addArgument(type, func_.getLoc()));
  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::AddInput(AbstractTensorHandle* input) {
  if (current_ods_input_ >= op_def_->input_arg_size())
    return InvalidArgument(
        absl::StrCat("More Input() (", current_ods_input_, ") calls than the ",
                     op_def_->input_arg_size(), " allowed input_args ; for op ",
                     state_->name.getStringRef().str()));

  auto* operand = dyn_cast<MlirTensor>(input);
  if (!operand) return InvalidArgument("Unable to cast input to MlirTensor");
  operands_.push_back(operand->getValue());

  // Get the next ArgDef and use it to infer the derived attributes associated
  // to this input.
  const tensorflow::OpDef::ArgDef& arg_def =
      op_def_->input_arg(current_ods_input_++);
  Type expected_type;
  if (arg_def.type() != tensorflow::DT_INVALID) {
    Builder builder(context_);
    TF_RETURN_IF_ERROR(
        tensorflow::ConvertDataType(arg_def.type(), builder, &expected_type));
    if (arg_def.is_ref()) {
      Type output_type;
      TF_RETURN_IF_ERROR(AddRef(expected_type, &output_type));
      expected_type = output_type;
    }
  } else {
    expected_type = cast<MlirTensor>(input)->getElementType();
  }
  if (!arg_def.type_attr().empty())
    attrs_[arg_def.type_attr()] = TypeAttr::get(expected_type);

  return ::tensorflow::OkStatus();
}

Status MlirAbstractOp::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
  if (current_ods_input_ >= op_def_->input_arg_size())
    return InvalidArgument(
        absl::StrCat("More Input() (", current_ods_input_, ") calls than the ",
                     op_def_->input_arg_size(), " allowed input_args"));

  for (AbstractTensorHandle* input : inputs) {
    auto* operand = dyn_cast<MlirTensor>(input);
    if (!operand) return InvalidArgument("Unable to cast input to MlirTensor");
    operands_.push_back(operand->getValue());
  }

  // Get the next ArgDef and use it to infer the derived attributes associated
  // to this input.
  const tensorflow::OpDef::ArgDef& arg_def =
      op_def_->input_arg(current_ods_input_++);
  if (!arg_def.number_attr().empty()) {
    Builder builder(context_);
    attrs_[arg_def.number_attr()] = builder.getI32IntegerAttr(inputs.size());
    // TODO(aminim): handle ref variable.
    if (arg_def.type() != tensorflow::DT_INVALID) {
      // TODO(aminim): check type wrt input
      Type arg_def_type;
      TF_RETURN_IF_ERROR(
          ConvertDataType(arg_def.type(), builder, &arg_def_type));
      // Ensure each of the type in the list matches the op def type.
      // TODO(aminim): can we improve the error message with the actual types?
      for (AbstractTensorHandle* input : inputs)
        if (arg_def_type != cast<MlirTensor>(input)->getElementType())
          return InvalidArgument(
              "Invalid input list: type mismatch the op def expectation");
    } else if (!inputs.empty()) {
      if (arg_def.type_attr().empty())
        return FailedPrecondition(
            "Invalid opdef type constraint: either type or type_attr required");

      attrs_[arg_def.type_attr()] =
          TypeAttr::get(cast<MlirTensor>(inputs.front())->getElementType());
    }
  } else if (!arg_def.type_list_attr().empty()) {
    // TODO(aminim): handle ref variable.
    SmallVector<Attribute, 8> types;
    types.reserve(inputs.size());
    for (AbstractTensorHandle* input : inputs)
      types.push_back(TypeAttr::get(cast<MlirTensor>(input)->getElementType()));
    attrs_[arg_def.type_list_attr()] = ArrayAttr::get(GetContext(), types);
  }
  return ::tensorflow::OkStatus();
}

Status MlirFunctionContext::Finalize(OutputList* outputs,
                                     AbstractFunction** f) {
  Block& body = func_.getBody().front();
  SmallVector<Value, 8> ret_operands;
  for (auto* output : outputs->outputs) {
    auto* operand = dyn_cast<MlirTensor>(output);
    if (!operand)
      return InvalidArgument("Capturing eager tensors is not supported yet.");
    if (operand->getValue().getContext() != context_.get())
      return InvalidArgument(
          "Capturing tensors from other context is not supported.");
    ret_operands.push_back(operand->getValue());
  }
  builder_.create<func::ReturnOp>(func_.getLoc(), ret_operands);

  auto arg_types = body.getArgumentTypes();
  auto result_types = body.getTerminator()->getOperandTypes();
  func_.setType(FunctionType::get(func_.getContext(), arg_types, result_types));
  *f = new MlirFunction(std::move(context_), std::move(module_), func_);
  return ::tensorflow::OkStatus();
}

extern "C" {
TracingContext* MlirTracingFactory(const char* fn_name, TF_Status* s) {
  return new MlirFunctionContext(fn_name);
}
}

}  // namespace
}  // namespace TF
}  // namespace mlir
