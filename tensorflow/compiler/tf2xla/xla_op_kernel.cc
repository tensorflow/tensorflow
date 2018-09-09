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

xla::XlaBuilder* XlaOpKernelContext::builder() const {
  return XlaContext::Get(this).builder();
}

// Retrieves an XlaExpression that was allocated by a previous Op.
static const XlaExpression* CastExpressionFromTensor(const Tensor& tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor.tensor_data().data());
  CHECK(expression->handle().valid() || expression->resource() != nullptr);
  VLOG(1) << "Fetched T" << expression->handle();
  return expression;
}

// Retrieves an uninitialized XlaExpression from a newly-allocated tensor.
static XlaExpression* CastExpressionFromUninitializedTensor(Tensor* tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
  CHECK(!expression->handle().valid());
  return const_cast<XlaExpression*>(expression);
}

// Retrieves the XlaOp from an input Tensor to an Op. This computation was
// constructed by an Op that executed previously and created the output Tensor
// using CreateOutputTensorFromComputation or CreateConstantOutputTensor.
static const xla::XlaOp& GetComputationFromTensor(const Tensor& tensor) {
  return CastExpressionFromTensor(tensor)->handle();
}

const xla::XlaOp& XlaOpKernelContext::Input(int index) {
  return GetComputationFromTensor(context_->input(index));
}

const xla::XlaOp& XlaOpKernelContext::Input(absl::string_view name) {
  return GetComputationFromTensor(GetInputTensorByName(name));
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
  TF_RETURN_IF_ERROR(context->op_kernel().InputRange(
      StringPiece(name.data(), name.length()), &start, &stop));
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
  const Tensor& tensor = context_->input(index);
  TensorShape new_shape(new_dims);
  if (tensor.NumElements() != new_shape.num_elements()) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        tensor.shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        new_shape.DebugString());
  }
  const XlaExpression* expression = CastExpressionFromTensor(tensor);

  auto copy_tensor_to_literal = [](const Tensor& tensor,
                                   xla::Literal* literal) {
    xla::Shape literal_shape;
    TF_RETURN_IF_ERROR(
        TensorShapeToXLAShape(tensor.dtype(), tensor.shape(), &literal_shape));

    *literal = xla::Literal(literal_shape);

    // memcpy over the payload ...
    // TODO(phawkins): handle string types.
    size_t total_bytes = tensor.TotalBytes();
    if (total_bytes > 0) {
      void* dst_ptr = literal->untyped_data();
      const void* src_ptr = DMAHelper::base(&tensor);
      memcpy(dst_ptr, src_ptr, total_bytes);
    }
    return Status::OK();
  };

  // If the tensor has a known constant value, there is no need to invoke XLA.
  if (expression->has_constant_value()) {
    Tensor temp(tensor.dtype());
    if (!temp.CopyFrom(expression->constant_value(), new_shape)) {
      // This should never happen. The constant should have a shape compatible
      // with the enclosing Tensor.
      return errors::Internal("Incompatible shapes in ConstantInputReshaped.");
    }

    return copy_tensor_to_literal(temp, constant_literal);
  }

  // Make sure we treat zero-element tensors as constant.
  if (new_shape.num_elements() == 0) {
    Tensor temp(tensor.dtype(), new_shape);

    return copy_tensor_to_literal(temp, constant_literal);
  }

  xla::XlaOp handle = expression->handle();
  if (new_shape != tensor.shape()) {
    // Reshape the handle to the desired shape.
    handle = xla::Reshape(handle, new_shape.dim_sizes());
  }

  // The XLA layout is specified minor to major, and TensorFlow's minor
  // dimension is the last one.
  std::vector<int64> layout_indices(new_shape.dims());
  std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
  xla::Layout layout = xla::LayoutUtil::MakeLayout(layout_indices);

  xla::StatusOr<bool> is_constant = builder()->IsConstant(handle);
  if (!is_constant.ok()) {
    Status status = is_constant.status();
    errors::AppendToMessage(&status, "while evaluating input ", index, " of ",
                            context_->op_kernel().type_string(),
                            " operator as a compile-time constant.");
    return status;
  }

  if (!is_constant.ValueOrDie()) {
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

  // Ask the XLA compiler to evaluate the data handle to a literal.
  xla::StatusOr<xla::XlaComputation> constant_graph =
      builder()->BuildConstantSubGraph(handle);
  if (!constant_graph.ok()) {
    return errors::Internal(
        "Error getting a compile-time constant graph for ",
        context_->op_kernel().name(), " input ", index,
        ".\nError: ", constant_graph.status().error_message());
  }
  xla::StatusOr<std::unique_ptr<xla::Literal>> computed =
      compiler()->client()->ComputeConstant(constant_graph.ValueOrDie(),
                                            &layout);
  if (!computed.ok()) {
    return errors::Internal("Error evaluating ", context_->op_kernel().name(),
                            " input ", index,
                            "as a compile-time constant.\nError: ",
                            computed.status().error_message());
  }
  *constant_literal = std::move(*computed.ValueOrDie());

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
  TF_RETURN_IF_ERROR(
      context_->input_list(StringPiece(name.data(), name.size()), &inputs));
  handles->clear();
  shapes->clear();
  for (const Tensor& input : inputs) {
    handles->push_back(GetComputationFromTensor(input));
    shapes->push_back(input.shape());
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputList(
    absl::string_view name, std::vector<xla::Literal>* outputs) {
  int start, stop;
  TF_RETURN_IF_ERROR(op_kernel().InputRange(
      StringPiece(name.data(), name.size()), &start, &stop));
  outputs->resize(stop - start);
  for (int i = start; i < stop; ++i) {
    TF_RETURN_IF_ERROR(ConstantInput(i, &(*outputs)[i]));
  }
  return Status::OK();
}

namespace {

Status ReadVariableInputTensor(const Tensor& tensor, DataType type,
                               const OpKernelContext* ctx, TensorShape* shape,
                               xla::XlaOp* value) {
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

  XlaContext& xla_context = XlaContext::Get(ctx);
  TF_ASSIGN_OR_RETURN(
      TensorShape representation_shape,
      xla_context.RepresentationShape(variable->shape(), variable->type()));
  if (representation_shape == variable->shape()) {
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
  return ReadVariableInputTensor(context_->input(index), type, context_, shape,
                                 value);
}

Status XlaOpKernelContext::ReadVariableInput(absl::string_view name,
                                             DataType type, TensorShape* shape,
                                             xla::XlaOp* value) {
  return ReadVariableInputTensor(GetInputTensorByName(name), type, context_,
                                 shape, value);
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

void XlaOpKernelContext::SetOutput(int index, const xla::XlaOp& handle) {
  // Makes the host Tensor that will refer to the expression.
  Tensor* output = nullptr;
  auto shape = builder()->GetShape(handle);
  if (!shape.ok()) {
    SetStatus(shape.status());
    return;
  }

  // The step's default allocator is the dummy XlaCompilationAllocator which
  // simply allocates a metadata buffer to hold the expression to which it
  // corresponds.
  TensorShape tensor_shape;
  OP_REQUIRES_OK(context_,
                 XLAShapeToTensorShape(shape.ValueOrDie(), &tensor_shape));
  OP_REQUIRES_OK(context_,
                 context_->allocate_output(index, tensor_shape, &output));

  // The expression is stored in the tensor's data buffer. Fill in the
  // fields now.
  XlaExpression* expression = CastExpressionFromUninitializedTensor(output);
  expression->set_handle(handle);
}

void XlaOpKernelContext::SetConstantOutput(int index, const Tensor& constant) {
  const TensorShape& shape = constant.shape();

  xla::BorrowingLiteral literal;
  OP_REQUIRES_OK(context_, HostTensorToBorrowingLiteral(constant, &literal));

  xla::XlaOp handle = xla::ConstantLiteral(builder(), literal);
  CHECK(handle.valid());

  // Make the Tensor that will refer to the expression.
  Tensor* output = nullptr;
  // The step's default allocator is the dummy XlaCompilationAllocator which
  // simply allocates a metadata buffer to hold the expression to which it
  // corresponds.
  OP_REQUIRES_OK(context_, context_->allocate_output(index, shape, &output));

  // The expression is stored in the tensor's data buffer. Fill in the
  // fields now.
  XlaExpression* expression = CastExpressionFromUninitializedTensor(output);
  expression->set_handle(handle);
  expression->set_constant_value(constant);
}

void XlaOpKernelContext::SetInvalidOutput(int index) {
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context_,
                 context_->allocate_output(index, TensorShape({}), &output));
  XlaExpression* expression = CastExpressionFromUninitializedTensor(output);
  xla::XlaOp handle;
  expression->set_handle(handle);
}

void XlaOpKernelContext::SetResourceOutput(int index, XlaResource* resource) {
  Tensor* output = nullptr;
  // The shape of the output tensor is the shape of the resource itself
  // (i.e., a scalar), not the shape of the resource's value.
  OP_REQUIRES_OK(context_,
                 context_->allocate_output(index, TensorShape(), &output));
  XlaExpression* expression = CastExpressionFromUninitializedTensor(output);
  expression->set_resource(resource);
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
                            const OpKernelContext* ctx, xla::XlaOp handle,
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

  XlaContext& xla_context = XlaContext::Get(ctx);
  TF_ASSIGN_OR_RETURN(TensorShape representation_shape,
                      xla_context.RepresentationShape(shape, type));
  if (shape != representation_shape) {
    handle = xla::Reshape(handle, representation_shape.dim_sizes());
  }
  return variable->SetValue(handle);
}

}  // namespace

Status XlaOpKernelContext::AssignVariable(int input_index, DataType type,
                                          xla::XlaOp handle) {
  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(context_->input(input_index), type, context_,
                              handle, builder());
}

Status XlaOpKernelContext::AssignVariable(absl::string_view name, DataType type,
                                          xla::XlaOp handle) {
  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(GetInputTensorByName(name), type, context_,
                              handle, builder());
}

XlaCompiler* XlaOpKernelContext::compiler() const {
  return XlaContext::Get(context_).compiler();
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
  return XlaContext::Get(context_).GetOrCreateMax(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMin(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateMin(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateAdd(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateAdd(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMul(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateMul(type);
}

const Tensor& XlaOpKernelContext::GetInputTensorByName(absl::string_view name) {
  const Tensor* tensor;
  CHECK(context_->input(StringPiece(name.data(), name.length()), &tensor).ok());
  return *tensor;
}

XlaOpKernel::XlaOpKernel(OpKernelConstruction* context) : OpKernel(context) {}

void XlaOpKernel::Compute(OpKernelContext* context) {
  XlaOpKernelContext xla_context(context);
  Compile(&xla_context);
}

}  // namespace tensorflow
