/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

#include <numeric>

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

XlaOpKernelContext::XlaOpKernelContext(OpKernelContext* context)
    : context_(context) {}

bool XlaOpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  return context_->ValidateInputsAreSameShape(op);
}

XlaContext* XlaOpKernelContext::xla_context() const {
  return &XlaContext::Get(context_);
}

xla::XlaBuilder* XlaOpKernelContext::builder() const {
  return xla_context()->builder();
}

XlaCompiler* XlaOpKernelContext::compiler() const {
  return xla_context()->compiler();
}

// Retrieves an XlaExpression that was allocated by a previous Op.
static const XlaExpression* CastExpressionFromTensor(const Tensor& tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor.tensor_data().data());
  CHECK(expression->kind() != XlaExpression::Kind::kInvalid)
      << expression->HumanString();
  return expression;
}

// Assigns an XlaExpression to a tensor on an XLA compilation device.
static void AssignExpressionToTensor(Tensor* tensor,
                                     const XlaExpression& value) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
  CHECK(expression->kind() == XlaExpression::Kind::kInvalid)
      << expression->HumanString();
  *const_cast<XlaExpression*>(expression) = value;
}

const XlaExpression& XlaOpKernelContext::InputExpression(int index) {
  return *CastExpressionFromTensor(context_->input(index));
}

const XlaExpression& XlaOpKernelContext::InputExpression(
    absl::string_view name) {
  return *CastExpressionFromTensor(GetInputTensorByName(name));
}

xla::XlaOp XlaOpKernelContext::Input(int index) {
  return InputExpression(index).AsXlaOp(builder());
}

xla::XlaOp XlaOpKernelContext::Input(absl::string_view name) {
  return InputExpression(name).AsXlaOp(builder());
}

TensorShape XlaOpKernelContext::InputShape(int index) {
  return context_->input(index).shape();
}

TensorShape XlaOpKernelContext::InputShape(absl::string_view name) {
  return GetInputTensorByName(name).shape();
}

DataType XlaOpKernelContext::input_type(int index) const {
  return context_->input(index).dtype();
}

DataType XlaOpKernelContext::InputType(absl::string_view name) {
  return GetInputTensorByName(name).dtype();
}

xla::PrimitiveType XlaOpKernelContext::input_xla_type(int index) {
  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(input_type(index), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

Status XlaOpKernelContext::ConstantInput(int index,
                                         xla::Literal* constant_literal) {
  return ConstantInputReshaped(
      index, context_->input(index).shape().dim_sizes(), constant_literal);
}

static xla::StatusOr<int> InputIndex(XlaOpKernelContext* context,
                                     absl::string_view name) {
  int start, stop;
  TF_RETURN_IF_ERROR(context->op_kernel().InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  return start;
}

Status XlaOpKernelContext::ConstantInput(absl::string_view name,
                                         xla::Literal* constant_literal) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInput(index, constant_literal);
}

Status XlaOpKernelContext::ConstantInputReshaped(
    int index, absl::Span<const int64> new_dims,
    xla::Literal* constant_literal) {
  XlaExpression e = InputExpression(index);
  xla::StatusOr<absl::optional<Tensor>> constant_or_status =
      e.ResolveConstant(compiler()->client());
  if (!constant_or_status.ok()) {
    Status status = constant_or_status.status();
    errors::AppendToMessage(&status, "while evaluating input ", index, " of ",
                            context_->op_kernel().type_string(),
                            " operator as a compile-time constant.");
    return status;
  }
  absl::optional<Tensor> constant = constant_or_status.ValueOrDie();
  if (!constant.has_value()) {
    return errors::InvalidArgument(
        "Input ", index, " to ", context_->op_kernel().type_string(),
        " operator must be a compile-time constant.\n"
        "\n"
        "XLA compilation requires that operator arguments that represent "
        "shapes or dimensions be evaluated to concrete values at compile time. "
        "This error means that a shape or dimension argument could not be "
        "evaluated at compile time, usually because the value of the argument "
        "depends on a parameter to the computation, on a variable, or on a "
        "stateful operation such as a random number generator.");
  }

  Tensor temp(constant->dtype());
  if (!temp.CopyFrom(*constant, TensorShape(new_dims))) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        constant->shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        TensorShape(new_dims).DebugString());
  }

  TF_ASSIGN_OR_RETURN(*constant_literal, HostTensorToLiteral(temp));
  return Status::OK();
}

// Converts an int32 or int64 scalar literal to an int64.
static Status LiteralToInt64Scalar(const xla::LiteralSlice& literal,
                                   int64* out) {
  if (xla::ShapeUtil::Rank(literal.shape()) != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::S32) {
    *out = literal.Get<int32>({});
  } else if (literal.shape().element_type() == xla::S64) {
    *out = literal.Get<int64>({});
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return Status::OK();
}

// Converts an float32 or float64 scalar literal to a float64.
static Status LiteralToFloat64Scalar(const xla::LiteralSlice& literal,
                                     double* out) {
  if (xla::ShapeUtil::Rank(literal.shape()) != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::F32) {
    *out = literal.Get<float>({});
  } else if (literal.shape().element_type() == xla::F64) {
    *out = literal.Get<double>({});
  } else {
    return errors::InvalidArgument("value must be either float32 or float64");
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(int index, int64* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  return LiteralToInt64Scalar(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(absl::string_view name,
                                                    int64* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntScalar(index, out);
}

Status XlaOpKernelContext::ConstantInputAsFloatScalar(int index, double* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  return LiteralToFloat64Scalar(literal, out);
}

// Converts an int32 or int64 1D literal to an int64 vector.
static Status LiteralToInt64Vector(const xla::LiteralSlice& literal,
                                   std::vector<int64>* out) {
  if (xla::ShapeUtil::Rank(literal.shape()) != 1) {
    return errors::InvalidArgument("value is not 1D");
  }
  int64 size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() == xla::S32) {
    for (int64 i = 0; i < size; ++i) {
      out->push_back(literal.Get<int32>({i}));
    }
  } else if (literal.shape().element_type() == xla::S64) {
    for (int64 i = 0; i < size; ++i) {
      out->push_back(literal.Get<int64>({i}));
    }
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputAsIntVector(int index,
                                                    std::vector<int64>* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntVector(absl::string_view name,
                                                    std::vector<int64>* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntVector(index, out);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    int index, std::vector<int64>* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    absl::string_view name, std::vector<int64>* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(int index,
                                                       xla::Literal* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  switch (literal.shape().element_type()) {
    case xla::S32: {
      *out = xla::Literal(
          xla::ShapeUtil::ChangeElementType(literal.shape(), xla::S64));
      auto src_data = literal.data<int32>();
      for (int64 i = 0; i < src_data.size(); ++i) {
        out->data<int64>()[i] = src_data[i];
      }
      return Status::OK();
    }
    case xla::S64:
      *out = std::move(literal);
      return Status::OK();

    default:
      return errors::InvalidArgument(
          "Invalid argument to ConstantInputAsInt64Literal: ",
          xla::ShapeUtil::HumanString(literal.shape()));
  }
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(absl::string_view name,
                                                       xla::Literal* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsInt64Literal(index, out);
}

// TODO(phawkins): validate that the dimensions form a valid shape, fail
// gracefully if they do not.
Status XlaOpKernelContext::ConstantInputAsShape(int index, TensorShape* shape) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  std::vector<int64> dims;
  TF_RETURN_IF_ERROR(LiteralToInt64Vector(literal, &dims));
  *shape = TensorShape(dims);
  return Status::OK();
}

Status XlaOpKernelContext::InputList(absl::string_view name,
                                     std::vector<xla::XlaOp>* handles,
                                     std::vector<TensorShape>* shapes) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(context_->input_list(name, &inputs));
  handles->clear();
  shapes->clear();
  for (const Tensor& input : inputs) {
    handles->push_back(CastExpressionFromTensor(input)->AsXlaOp(builder()));
    shapes->push_back(input.shape());
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputList(
    absl::string_view name, std::vector<xla::Literal>* outputs) {
  int start, stop;
  TF_RETURN_IF_ERROR(op_kernel().InputRange(name, &start, &stop));
  outputs->resize(stop - start);
  for (int i = start; i < stop; ++i) {
    TF_RETURN_IF_ERROR(ConstantInput(i, &(*outputs)[i]));
  }
  return Status::OK();
}

namespace {

Status ReadVariableInputTensor(const Tensor& tensor, DataType type,
                               const XlaOpKernelContext* ctx,
                               TensorShape* shape, xla::XlaOp* value) {
  const XlaExpression* expression = CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::InvalidArgument("Read of uninitialized variable ",
                                   variable->name());
  }
  if (variable->type() != type) {
    return errors::InvalidArgument(
        "Type mismatch for read of variable ", variable->name(), ". Expected ",
        DataTypeString(type), "; got ", DataTypeString(variable->type()));
  }
  if (shape) {
    *shape = variable->shape();
  }

  TF_ASSIGN_OR_RETURN(xla::Shape representation_shape,
                      ctx->compiler()->options().shape_representation_fn(
                          variable->shape(), variable->type()));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(variable->type(), variable->shape(), &xla_shape));
  if (xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    *value = variable->value();
  } else {
    *value = xla::Reshape(variable->value(), variable->shape().dim_sizes());
  }
  return Status::OK();
}

}  // namespace

Status XlaOpKernelContext::ReadVariableInput(int index, DataType type,
                                             TensorShape* shape,
                                             xla::XlaOp* value) {
  return ReadVariableInputTensor(context_->input(index), type, this, shape,
                                 value);
}

Status XlaOpKernelContext::ReadVariableInput(absl::string_view name,
                                             DataType type, TensorShape* shape,
                                             xla::XlaOp* value) {
  return ReadVariableInputTensor(GetInputTensorByName(name), type, this, shape,
                                 value);
}

Status XlaOpKernelContext::GetVariableTypeAndShape(int index, DataType* type,
                                                   TensorShape* shape) const {
  const Tensor& tensor = context_->input(index);
  const XlaExpression* expression = CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::InvalidArgument("Read of uninitialized variable ",
                                   variable->name());
  }
  *type = variable->type();
  *shape = variable->shape();
  return Status::OK();
}

void XlaOpKernelContext::SetOutputExpression(int index,
                                             const XlaExpression& expression) {
  Status status = [&] {
    // The step's default allocator is the dummy XlaCompilationAllocator which
    // simply allocates a metadata buffer to hold the expression to which it
    // corresponds.
    Tensor* output = nullptr;
    // Provides a special behavior for DT_VARIANT: a variant is treated as
    // DT_UINT8 scalar as the type to allow mapping for variant to more generic
    // types.
    if (expression.dtype() == DT_VARIANT) {
      // tensor_data() is not supported for variant Tensor (i.e.,
      // DataTypeCanUseMemcpy is false for DT_VARIANT), and so storing the
      // XlaExpression inside the Tensor's tensor_data() does not work for
      // variant. Instead construct a uint8 tensor and store the expression in
      // its value.
      // TODO(jpienaar): This should be refactored to stop masquerading
      // XlaExpressions as Tensors.
      output = new Tensor();
      TensorShape tensor_shape;
      TF_RETURN_IF_ERROR(
          context_->allocate_temp(DT_UINT8, tensor_shape, output));
      context_->set_output(index, *output);
    } else {
      TF_ASSIGN_OR_RETURN(TensorShape shape, expression.GetShape());
      TF_RETURN_IF_ERROR(context_->allocate_output(index, shape, &output));
    }
    AssignExpressionToTensor(output, expression);
    return Status::OK();
  }();
  if (!status.ok()) {
    SetStatus(status);
  }
}

void XlaOpKernelContext::SetOutput(int index, const xla::XlaOp& handle) {
  SetOutputExpression(
      index,
      XlaExpression::XlaOp(handle, context_->expected_output_dtype(index)));
}

void XlaOpKernelContext::SetConstantOutput(int index, const Tensor& constant) {
  SetOutputExpression(index, XlaExpression::Constant(constant));
}

void XlaOpKernelContext::SetResourceOutput(int index, XlaResource* resource) {
  SetOutputExpression(index, XlaExpression::Resource(resource));
}

Status XlaOpKernelContext::GetResourceInput(int index, XlaResource** resource) {
  const XlaExpression* expression =
      CastExpressionFromTensor(context_->input(index));
  TF_RET_CHECK(expression->resource() != nullptr);
  *resource = expression->resource();
  return Status::OK();
}

namespace {

Status AssignVariableTensor(const Tensor& tensor, DataType type,
                            const XlaOpKernelContext* ctx, xla::XlaOp handle,
                            xla::XlaBuilder* builder) {
  const XlaExpression* expression = CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);

  auto shape_or_status = builder->GetShape(handle);
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  TensorShape shape;
  TF_RETURN_IF_ERROR(
      XLAShapeToTensorShape(shape_or_status.ValueOrDie(), &shape));

  TF_RETURN_IF_ERROR(variable->SetTypeAndShape(type, shape));

  TF_ASSIGN_OR_RETURN(
      xla::Shape representation_shape,
      ctx->compiler()->options().shape_representation_fn(shape, type));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
  if (!xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    handle = xla::Reshape(handle,
                          xla::AsInt64Slice(representation_shape.dimensions()));
  }
  return variable->SetValue(handle);
}

}  // namespace

Status XlaOpKernelContext::AssignVariable(int input_index, DataType type,
                                          xla::XlaOp handle) {
  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(context_->input(input_index), type, this, handle,
                              builder());
}

Status XlaOpKernelContext::AssignVariable(absl::string_view name, DataType type,
                                          xla::XlaOp handle) {
  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(GetInputTensorByName(name), type, this, handle,
                              builder());
}

void XlaOpKernelContext::CtxFailure(const Status& s) {
  context_->CtxFailure(s);
}
void XlaOpKernelContext::CtxFailureWithWarning(const Status& s) {
  context_->CtxFailureWithWarning(s);
}
void XlaOpKernelContext::CtxFailure(const char* file, int line,
                                    const Status& s) {
  context_->CtxFailure(file, line, s);
}
void XlaOpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                               const Status& s) {
  context_->CtxFailureWithWarning(file, line, s);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMax(
    const DataType type) {
  return xla_context()->GetOrCreateMax(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMin(
    const DataType type) {
  return xla_context()->GetOrCreateMin(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateAdd(
    const DataType type) {
  return xla_context()->GetOrCreateAdd(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMul(
    const DataType type) {
  return xla_context()->GetOrCreateMul(type);
}

const Tensor& XlaOpKernelContext::GetInputTensorByName(absl::string_view name) {
  const Tensor* tensor;
  CHECK(context_->input(name, &tensor).ok());
  return *tensor;
}

XlaOpKernel::XlaOpKernel(OpKernelConstruction* context) : OpKernel(context) {}

void XlaOpKernel::Compute(OpKernelContext* context) {
  XlaOpKernelContext xla_context(context);
  Compile(&xla_context);
}

}  // namespace tensorflow
