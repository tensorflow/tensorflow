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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

XlaOpKernelContext::XlaOpKernelContext(OpKernelContext* context)
    : context_(context),
      dynamic_dimension_is_minus_one_(false),
      value_inference_(xla_context()->builder()) {}

bool XlaOpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  return context_->ValidateInputsAreSameShape(op);
}

XlaContext* XlaOpKernelContext::xla_context() const {
  return &XlaContext::Get(context_);
}

xla::XlaBuilder* XlaOpKernelContext::builder() const {
  return xla_context()->builder();
}

xla::ValueInference& XlaOpKernelContext::value_inference() {
  return value_inference_;
}

XlaCompiler* XlaOpKernelContext::compiler() const {
  return xla_context()->compiler();
}

const XlaExpression& XlaOpKernelContext::InputExpression(int index) {
  return *XlaExpression::CastExpressionFromTensor(context_->input(index));
}

const XlaExpression& XlaOpKernelContext::InputExpression(
    absl::string_view name) {
  return *XlaExpression::CastExpressionFromTensor(GetInputTensorByName(name));
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

StatusOr<xla::Shape> XlaOpKernelContext::InputXlaShape(int index) {
  return builder()->GetShape(Input(index));
}

StatusOr<xla::Shape> XlaOpKernelContext::InputXlaShape(absl::string_view name) {
  return builder()->GetShape(Input(name));
}

DataType XlaOpKernelContext::input_type(int index) const {
  DataType type = context_->input_dtype(index);
  if (type == DT_UINT8) {
    // Masqueraded XlaExpression could have different type. See
    // XlaOpKernelContext::SetOutputExpression for details.
    auto expression =
        XlaExpression::CastExpressionFromTensor(context_->input(index));
    type = expression->dtype();
  }
  return type;
}

DataType XlaOpKernelContext::InputType(absl::string_view name) {
  const Tensor& tensor = GetInputTensorByName(name);
  DataType type = tensor.dtype();
  if (type == DT_UINT8) {
    // Masqueraded XlaExpression could have different type. See
    // XlaOpKernelContext::SetOutputExpression for details.
    auto expression = XlaExpression::CastExpressionFromTensor(tensor);
    type = expression->dtype();
  }
  return type;
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

xla::PrimitiveType XlaOpKernelContext::InputXlaType(absl::string_view name) {
  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(InputType(name), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

Status XlaOpKernelContext::ConstantInput(int index,
                                         xla::Literal* constant_literal,
                                         xla::ValueInferenceMode mode) {
  if (this->InputXlaShape(index)->is_dynamic()) {
    return errors::InvalidArgument(
        "Reading input as constant from a dynamic tensor is not yet supported. "
        "Xla shape: ",
        this->InputXlaShape(index)->ToString());
  }
  return ConstantInputReshaped(index,
                               context_->input(index).shape().dim_sizes(),
                               constant_literal, mode);
}

static StatusOr<int> InputIndex(XlaOpKernelContext* context,
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

Status XlaOpKernelContext::ResolveInputDynamism(
    int index, xla::Literal* dynamism_literal) {
  return ResolveInputDynamismReshaped(
      index, context_->input(index).shape().dim_sizes(), dynamism_literal);
}

Status XlaOpKernelContext::ResolveInputDynamism(
    absl::string_view name, xla::Literal* dynamism_literal) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamism(index, dynamism_literal);
}

Status XlaOpKernelContext::ConstantInput(absl::string_view name,
                                         xla::Literal* constant_literal,
                                         xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInput(index, constant_literal, mode);
}

Status XlaOpKernelContext::ConstantInputReshaped(
    int index, absl::Span<const int64_t> new_dims,
    xla::Literal* constant_literal, xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(Tensor constant, ConstantInputTensor(index, mode));
  Tensor temp(constant.dtype());
  if (!temp.CopyFrom(constant, TensorShape(new_dims))) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        constant.shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        TensorShape(new_dims).DebugString());
  }

  TF_ASSIGN_OR_RETURN(*constant_literal, HostTensorToLiteral(temp));
  return OkStatus();
}

// Converts an int32 or int64 scalar literal to an int64.
static Status LiteralToInt64Scalar(const xla::LiteralSlice& literal,
                                   int64_t* out) {
  if (literal.shape().rank() != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::S32) {
    *out = literal.Get<int32>({});
  } else if (literal.shape().element_type() == xla::S64) {
    *out = literal.Get<int64_t>({});
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return OkStatus();
}

// Converts an float32 or float64 scalar literal to a float64.
static Status LiteralToFloat64Scalar(const xla::LiteralSlice& literal,
                                     double* out) {
  if (literal.shape().rank() != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::F32) {
    *out = literal.Get<float>({});
  } else if (literal.shape().element_type() == xla::F64) {
    *out = literal.Get<double>({});
  } else {
    return errors::InvalidArgument("value must be either float32 or float64");
  }
  return OkStatus();
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(
    int index, int64_t* out, xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToInt64Scalar(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(
    absl::string_view name, int64_t* out, xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntScalar(index, out, mode);
}

Status XlaOpKernelContext::ConstantInputAsFloatScalar(
    int index, double* out, xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToFloat64Scalar(literal, out);
}

static Status LiteralToPredVector(const xla::LiteralSlice& literal,
                                  std::vector<bool>* out) {
  if (literal.shape().rank() != 1) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   literal.shape().DebugString());
  }
  int64_t size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() != xla::PRED) {
    return errors::InvalidArgument("value is not PRED");
  }
  for (int64_t i = 0; i < size; ++i) {
    out->push_back(literal.Get<bool>({i}));
  }
  return OkStatus();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPred(int index, bool* out) {
  xla::Literal literal;
  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<Tensor> dynamism_or_status = e.ResolveDynamism(client);
  if (!dynamism_or_status.ok()) {
    // When failed to resolve dynamism, conservatively consider the value
    // dynamic. This could happen if the input depends on some ops like
    // custom-call that is not supported generally for dynamism computation.
    //
    // TODO(b/176993339): Support resolving dynamism across computations so
    // resolving dynamism will not fail in those cases.
    *out = true;
    return OkStatus();
  }
  Tensor dynamism = dynamism_or_status.ValueOrDie();

  Tensor temp(dynamism.dtype());
  TensorShape tensor_shape({});
  if (!temp.CopyFrom(dynamism, tensor_shape)) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        dynamism.shape().DebugString(), " which is not a R0 ", tensor_shape);
  }

  TF_ASSIGN_OR_RETURN(literal, HostTensorToLiteral(temp));
  *out = literal.Get<bool>({});
  return OkStatus();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPredVector(
    absl::string_view name, std::vector<bool>* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamismIntoPredVector(index, out);
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPred(absl::string_view name,
                                                        bool* out) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamismIntoPred(index, out);
}

Status XlaOpKernelContext::ResolveInputDynamismReshaped(
    int index, absl::Span<const int64_t> new_dims,
    xla::Literal* dynamism_literal) {
  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<Tensor> dynamism_or_status = e.ResolveDynamism(client);
  if (!dynamism_or_status.ok()) {
    xla::Literal true_literal = xla::LiteralUtil::CreateR0<bool>(true);
    // When failed to resolve dynamism, conservatively consider the value
    // dynamic. This could happen if the input depends on some ops like
    // custom-call that is not supported generally for dynamism computation.
    *dynamism_literal =
        true_literal
            .Broadcast(xla::ShapeUtil::MakeShape(xla::PRED, new_dims), {})
            .ValueOrDie();

    return OkStatus();
  }
  Tensor dynamism = dynamism_or_status.ValueOrDie();

  Tensor temp(dynamism.dtype());
  if (!temp.CopyFrom(dynamism, TensorShape(new_dims))) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        dynamism.shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        TensorShape(new_dims).DebugString());
  }

  TF_ASSIGN_OR_RETURN(*dynamism_literal, HostTensorToLiteral(temp));
  return OkStatus();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPredVector(
    int index, std::vector<bool>* out) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ResolveInputDynamismReshaped(
      index, {InputShape(index).num_elements()}, &literal));

  return LiteralToPredVector(literal, out);
}

// Converts an int32 or int64 1D literal to an int64 vector.
static Status LiteralToInt64Vector(const xla::LiteralSlice& literal,
                                   std::vector<int64_t>* out) {
  if (literal.shape().rank() != 1) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   literal.shape().DebugString());
  }
  int64_t size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() == xla::S32) {
    for (int64_t i = 0; i < size; ++i) {
      out->push_back(literal.Get<int32>({i}));
    }
  } else if (literal.shape().element_type() == xla::S64) {
    for (int64_t i = 0; i < size; ++i) {
      out->push_back(literal.Get<int64_t>({i}));
    }
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return OkStatus();
}

Status XlaOpKernelContext::ConstantInputAsIntVector(
    int index, std::vector<int64_t>* out, xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntVector(
    absl::string_view name, std::vector<int64_t>* out,
    xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntVector(index, out, mode);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    int index, std::vector<int64_t>* out, xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    absl::string_view name, std::vector<int64_t>* out,
    xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(
    int index, xla::Literal* out, xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  switch (literal.shape().element_type()) {
    case xla::S32: {
      *out = xla::Literal(
          xla::ShapeUtil::ChangeElementType(literal.shape(), xla::S64));
      auto src_data = literal.data<int32>();
      for (int64_t i = 0; i < src_data.size(); ++i) {
        out->data<int64_t>()[i] = src_data[i];
      }
      return OkStatus();
    }
    case xla::S64:
      *out = std::move(literal);
      return OkStatus();

    default:
      return errors::InvalidArgument(
          "Invalid argument to ConstantInputAsInt64Literal: ",
          xla::ShapeUtil::HumanString(literal.shape()));
  }
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(
    absl::string_view name, xla::Literal* out, xla::ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsInt64Literal(index, out, mode);
}

// TODO(phawkins): validate that the dimensions form a valid shape, fail
// gracefully if they do not.
Status XlaOpKernelContext::ConstantInputAsShape(int index, TensorShape* shape,
                                                xla::ValueInferenceMode mode) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(LiteralToInt64Vector(literal, &dims));
  *shape = TensorShape(dims);
  return OkStatus();
}

Status XlaOpKernelContext::ConstantInputAsPartialShape(
    int index, PartialTensorShape* shape) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  // If `literal` is a scalar it's value must be -1.
  if (literal.shape().rank() == 0) {
    int64_t shape_val;
    TF_RETURN_IF_ERROR(LiteralToInt64Scalar(literal, &shape_val));
    if (shape_val != -1) {
      return errors::InvalidArgument(
          "Cannot convert value to PartialTensorShape: ", shape_val);
    }
    *shape = PartialTensorShape();  // Shape with unknown rank.
    return OkStatus();
  }
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(LiteralToInt64Vector(literal, &dims));
  *shape = PartialTensorShape(dims);
  return OkStatus();
}

Status XlaOpKernelContext::InputList(absl::string_view name,
                                     std::vector<xla::XlaOp>* handles,
                                     std::vector<TensorShape>* shapes) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(context_->input_list(name, &inputs));
  handles->clear();
  shapes->clear();
  for (const Tensor& input : inputs) {
    handles->push_back(
        XlaExpression::CastExpressionFromTensor(input)->AsXlaOp(builder()));
    shapes->push_back(input.shape());
  }
  return OkStatus();
}

Status XlaOpKernelContext::ConstantInputList(absl::string_view name,
                                             std::vector<xla::Literal>* outputs,
                                             xla::ValueInferenceMode mode) {
  int start, stop;
  TF_RETURN_IF_ERROR(op_kernel().InputRange(name, &start, &stop));
  outputs->resize(stop - start);
  for (int i = start; i < stop; ++i) {
    TF_RETURN_IF_ERROR(ConstantInput(i, &(*outputs)[i], mode));
  }
  return OkStatus();
}

StatusOr<Tensor> XlaOpKernelContext::ConstantInputTensor(
    int index, xla::ValueInferenceMode mode) {
  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<std::optional<Tensor>> constant_or_status =
      e.ResolveConstant(client, dynamic_dimension_is_minus_one_, mode);
  if (!constant_or_status.ok()) {
    Status status = constant_or_status.status();
    errors::AppendToMessage(&status, "while evaluating input ", index, " of ",
                            context_->op_kernel().type_string(),
                            " operator as a compile-time constant.");
    return status;
  }
  std::optional<Tensor> constant = constant_or_status.ValueOrDie();
  if (!constant.has_value()) {
    return errors::InvalidArgument(
        "Input ", index, " to node `", context_->op_kernel().name(),
        "` with op ", context_->op_kernel().type_string(),
        " must be a compile-time constant.\n\n"
        "XLA compilation requires that operator arguments that represent "
        "shapes or dimensions be evaluated to concrete values at compile time. "
        "This error means that a shape or dimension argument could not be "
        "evaluated at compile time, usually because the value of the argument "
        "depends on a parameter to the computation, on a variable, or on a "
        "stateful operation such as a random number generator.");
  }
  return *constant;
}

namespace {

Status ReadVariableInputTensor(const Tensor& tensor, DataType type,
                               const XlaOpKernelContext* ctx,
                               TensorShape* shape, xla::XlaOp* value) {
  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::FailedPrecondition(
        "Read variable failure ", variable->name(),
        ". It could mean the variable is uninitialized or the variable is on "
        "another device ");
  }
  if (variable->type() != type) {
    return errors::InvalidArgument(
        "Trying to read variable with wrong dtype. Expected ",
        DataTypeString(type), " got ", DataTypeString(variable->type()));
  }
  if (shape) {
    *shape = variable->shape();
  }

  if (!variable->IsOverwritten() && expression->constant_value()) {
    TF_ASSIGN_OR_RETURN(xla::Literal literal,
                        HostTensorToLiteral(*expression->constant_value()));
    *value = xla::ConstantLiteral(ctx->builder(), literal);
    return OkStatus();
  }
  auto shape_determination_fns =
      ctx->compiler()->options().shape_determination_fns;
  XlaLayoutPreference layout_preference =
      shape_determination_fns.layout_preference_fn(
          variable->shape(), variable->type(), std::nullopt);
  TF_ASSIGN_OR_RETURN(xla::Shape representation_shape,
                      shape_determination_fns.shape_representation_fn(
                          variable->shape(), variable->type(),
                          /*use_fast_memory=*/false, layout_preference));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(variable->type(), variable->shape(), &xla_shape));
  if (xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    *value = variable->value();
  } else {
    *value = xla::Reshape(variable->value(), variable->shape().dim_sizes());
  }
  return OkStatus();
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
  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::InvalidArgument(
        "Read variable failure ", variable->name(),
        ". It could mean the variable is uninitialized or the variable is on "
        "another device ");
  }
  *type = variable->type();
  *shape = variable->shape();
  return OkStatus();
}

void XlaOpKernelContext::SetOutputExpression(int index,
                                             const XlaExpression& expression) {
  Status status = [&] {
    // The step's default allocator is the dummy XlaCompilationAllocator which
    // simply allocates a metadata buffer to hold the expression to which it
    // corresponds.
    // Provides a special behavior for DT_VARIANT and other types that are not
    // trivially copyable. In those cases, allocate a tensor of type DT_UINT8.
    if (!DataTypeCanUseMemcpy(expression.dtype())) {
      // tensor_data() is not supported for tensors that cannot be copied via
      // memcpy, as the copy logic might try to inspect the stored data (e.g.
      // a std::string). This is likely to fail, as the data is invalid given
      // that it actually encodes an XlaExpression. Using a uint8 tensor is
      // always safe, so simply do that.
      // TODO(jpienaar): This should be refactored to stop masquerading
      // XlaExpressions as Tensors.
      Tensor output;
      TensorShape tensor_shape;
      TF_RETURN_IF_ERROR(
          context_->allocate_temp(DT_UINT8, tensor_shape, &output));
      context_->set_output(index, output);
    } else {
      Tensor* output = nullptr;
      TF_ASSIGN_OR_RETURN(TensorShape shape, expression.GetShape());
      TF_RETURN_IF_ERROR(context_->allocate_output(index, shape, &output));
    }
    XlaExpression::AssignExpressionToTensor(expression,
                                            context_->mutable_output(index));
    return OkStatus();
  }();
  if (!status.ok()) {
    SetStatus(status);
  }
}

xla::PrimitiveType XlaOpKernelContext::output_xla_type(int index) {
  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(expected_output_dtype(index), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

void XlaOpKernelContext::SetOutput(int index, const xla::XlaOp& handle) {
  SetOutputExpression(
      index,
      XlaExpression::XlaOp(handle, context_->expected_output_dtype(index)));
}

void XlaOpKernelContext::SetConstantOutput(int index, const Tensor& constant) {
  SetOutputExpression(index, XlaExpression::Constant(constant));
}

void XlaOpKernelContext::SetTensorListOutput(int index,
                                             const xla::XlaOp& handle) {
  SetOutputExpression(index, XlaExpression::TensorList(handle));
}

void XlaOpKernelContext::SetResourceOutput(int index, XlaResource* resource) {
  SetOutputExpression(index, XlaExpression::Resource(resource));
}

Status XlaOpKernelContext::GetResourceInput(int index, XlaResource** resource) {
  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(context_->input(index));
  TF_RET_CHECK(expression->resource() != nullptr);
  *resource = expression->resource();
  return OkStatus();
}

namespace {

Status AssignVariableTensor(const Tensor& tensor, DataType type,
                            const XlaOpKernelContext* ctx, xla::XlaOp handle,
                            xla::XlaBuilder* builder) {
  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
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

  auto shape_determination_fns =
      ctx->compiler()->options().shape_determination_fns;
  XlaLayoutPreference layout_preference =
      shape_determination_fns.layout_preference_fn(shape, type, std::nullopt);
  TF_ASSIGN_OR_RETURN(xla::Shape representation_shape,
                      shape_determination_fns.shape_representation_fn(
                          shape, type,
                          /*use_fast_memory=*/false, layout_preference));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
  if (!xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    handle = xla::Reshape(handle, representation_shape.dimensions());
  }
  variable->SetRepresentationShape(representation_shape);
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

static Status GetStatusWithStackTrace(const Status& s,
                                      const XlaOpKernelContext* ctx) {
  if (s.code() == error::INVALID_ARGUMENT) {
    return Status{s.code(),
                  absl::StrCat(s.error_message(), "\n", ctx->StackTrace())};
  }
  return s;
}

void XlaOpKernelContext::CtxFailure(const Status& s) {
  context_->CtxFailure(GetStatusWithStackTrace(s, this));
}
void XlaOpKernelContext::CtxFailureWithWarning(const Status& s) {
  context_->CtxFailureWithWarning(GetStatusWithStackTrace(s, this));
}

void XlaOpKernelContext::CtxFailure(const char* file, int line,
                                    const Status& s) {
  context_->CtxFailure(file, line, GetStatusWithStackTrace(s, this));
}
void XlaOpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                               const Status& s) {
  context_->CtxFailureWithWarning(file, line, GetStatusWithStackTrace(s, this));
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

std::string XlaOpKernelContext::StackTrace() const {
  if (const AbstractStackTrace* stack_trace =
          xla_context()->StackTraceForNodeName(op_kernel().name())) {
    AbstractStackTrace::TracePrintingOptions opts;
    opts.show_line_contents = true;
    opts.filter_common_prefix = true;
    opts.drop_internal_frames = true;
    return absl::StrCat("\nStack trace for op definition: \n",
                        stack_trace->ToString(opts), "\n");
  } else {
    return "";
  }
}

}  // namespace tensorflow
