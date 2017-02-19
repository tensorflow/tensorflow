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
#include "tensorflow/compiler/tf2xla/xla_context.h"

namespace tensorflow {

XlaOpKernelContext::XlaOpKernelContext(OpKernelContext* context)
    : context_(context) {}

bool XlaOpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  return context_->ValidateInputsAreSameShape(op);
}

xla::ComputationBuilder* XlaOpKernelContext::builder() const {
  return XlaContext::Get(this).builder();
}

// Retrieves an XlaExpression that was allocated by a previous Op.
static const XlaExpression* CastExpressionFromTensor(const Tensor& tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor.tensor_data().data());
  CHECK_NE(expression->handle().handle(), 0);
  VLOG(1) << "Fetched T" << expression->handle().handle();
  return expression;
}

// Retrieves an uninitialized XlaExpression from a newly-allocated tensor.
static XlaExpression* CastExpressionFromUninitializedTensor(Tensor* tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
  CHECK_EQ(expression->handle().handle(), 0);
  return const_cast<XlaExpression*>(expression);
}

// Retrieves the ComputationDataHandle from an input Tensor to an Op. This
// computation was constructed by an Op that executed previously and
// created the output Tensor using CreateOutputTensorFromComputation
// or CreateConstantOutputTensor.
static const xla::ComputationDataHandle& GetComputationFromTensor(
    const Tensor& tensor) {
  return CastExpressionFromTensor(tensor)->handle();
}

const xla::ComputationDataHandle& XlaOpKernelContext::Input(int index) {
  return GetComputationFromTensor(context_->input(index));
}

TensorShape XlaOpKernelContext::InputShape(int index) {
  return context_->input(index).shape();
}

Status XlaOpKernelContext::ConstantInput(int index,
                                         xla::Literal* constant_literal) {
  return ConstantInputReshaped(
      index, context_->input(index).shape().dim_sizes(), constant_literal);
}

Status XlaOpKernelContext::ConstantInputReshaped(
    int index, gtl::ArraySlice<int64> new_dims,
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

  // If the tensor has a known constant value, there is no need to invoke XLA.
  if (expression->has_constant_value()) {
    Tensor temp(tensor.dtype());
    if (!temp.CopyFrom(expression->constant_value(), new_shape)) {
      // This should never happen. The constant should have a shape compatible
      // with the enclosing Tensor.
      return errors::Internal("Incompatible shapes in ConstantInputReshaped.");
    }
    return HostTensorToLiteral(temp, constant_literal);
  }

  // Make sure we treat zero-element tensors as constant.
  if (new_shape.num_elements() == 0) {
    Tensor temp(tensor.dtype(), new_shape);
    return HostTensorToLiteral(temp, constant_literal);
  }

  xla::ComputationDataHandle handle = expression->handle();
  if (new_shape != tensor.shape()) {
    // Reshape the handle to the desired shape.
    handle = builder()->Reshape(handle, new_shape.dim_sizes());
  }

  // The XLA layout is specified minor to major, and TensorFlow's minor
  // dimension is the last one.
  std::vector<int64> layout_indices(new_shape.dims());
  std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
  xla::Layout layout = xla::LayoutUtil::MakeLayout(layout_indices);

  // Ask the XLA compiler to evaluate the data handle to a literal.
  xla::StatusOr<std::unique_ptr<xla::GlobalData>> computed =
      builder()->ComputeConstant(handle, &layout);
  if (!computed.ok()) {
    return errors::InvalidArgument(
        "Error evaluating ", context_->op_kernel().name(), " input ", index,
        ": ", computed.status().error_message());
  }
  // Fetch the literal from the compiler service.
  xla::StatusOr<std::unique_ptr<xla::Literal>> constant =
      builder()->client()->Transfer(*computed.ValueOrDie());
  if (!constant.ok()) {
    return errors::InvalidArgument(
        "Error evaluating ", context_->op_kernel().name(), " input ", index,
        ": ", constant.status().error_message());
  }
  constant_literal->Swap(constant.ValueOrDie().get());
  return Status::OK();
}

// Converts an int32 or int64 1D literal to an int64 vector.
static Status LiteralToInt64Vector(const xla::Literal& literal,
                                   std::vector<int64>* out) {
  if (xla::ShapeUtil::Rank(literal.shape()) != 1) {
    return errors::InvalidArgument("value is not 1D");
  }
  int64 size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() == xla::S32) {
    for (int64 i = 0; i < size; ++i) {
      out->push_back(xla::LiteralUtil::Get<int32>(literal, {i}));
    }
  } else if (literal.shape().element_type() == xla::S64) {
    for (int64 i = 0; i < size; ++i) {
      out->push_back(xla::LiteralUtil::Get<int64>(literal, {i}));
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

Status XlaOpKernelContext::InputList(
    StringPiece name, std::vector<xla::ComputationDataHandle>* handles,
    std::vector<TensorShape>* shapes) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(context_->input_list(name, &inputs));
  handles->clear();
  shapes->clear();
  for (const Tensor& input : inputs) {
    handles->push_back(GetComputationFromTensor(input));
    shapes->push_back(input.shape());
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputList(
    StringPiece name, std::vector<xla::Literal>* outputs) {
  int start, stop;
  TF_RETURN_IF_ERROR(op_kernel().InputRange(name, &start, &stop));
  outputs->resize(stop - start);
  for (int i = start; i < stop; ++i) {
    TF_RETURN_IF_ERROR(ConstantInput(i, &(*outputs)[i]));
  }
  return Status::OK();
}

void XlaOpKernelContext::SetOutput(int index,
                                   const xla::ComputationDataHandle& handle) {
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
  OP_REQUIRES_OK(
      context_,
      context_->allocate_output(
          index, XLAShapeToTensorShape(*shape.ValueOrDie()), &output));

  // The expression is stored in the tensor's data buffer. Fill in the
  // fields now.
  XlaExpression* expression = CastExpressionFromUninitializedTensor(output);
  expression->set_handle(handle);
}

void XlaOpKernelContext::SetConstantOutput(int index, const Tensor& constant) {
  const TensorShape& shape = constant.shape();

  xla::Literal literal;
  OP_REQUIRES_OK(context_, HostTensorToLiteral(constant, &literal));
  xla::ComputationDataHandle handle = builder()->ConstantLiteral(literal);

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

void XlaOpKernelContext::SetOpHasSideEffects() {
  XlaContext::Get(context_).AddSideEffects();
}

void XlaOpKernelContext::CtxFailure(Status s) { context_->CtxFailure(s); }
void XlaOpKernelContext::CtxFailureWithWarning(Status s) {
  context_->CtxFailureWithWarning(s);
}

const xla::Computation* XlaOpKernelContext::GetOrCreateMax(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateMax(type);
}

const xla::Computation* XlaOpKernelContext::GetOrCreateAdd(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateAdd(type);
}

const xla::Computation* XlaOpKernelContext::GetOrCreateSigmoid(
    const DataType type) {
  return XlaContext::Get(context_).GetOrCreateSigmoid(type);
}

XlaOpKernel::XlaOpKernel(OpKernelConstruction* context) : OpKernel(context) {}

void XlaOpKernel::Compute(OpKernelContext* context) {
  XlaOpKernelContext xla_context(context);
  Compile(&xla_context);
}

}  // namespace tensorflow
