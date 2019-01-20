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

#include "tensorflow/compiler/tf2xla/xla_expression.h"

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

XlaExpression::XlaExpression() = default;

XlaExpression XlaExpression::Invalid() {
  XlaExpression e;
  e.kind_ = Kind::kInvalid;
  return e;
}

XlaExpression XlaExpression::Constant(Tensor value) {
  XlaExpression e;
  e.kind_ = Kind::kConstant;
  e.dtype_ = value.dtype();
  e.constant_value_ = value;
  return e;
}

XlaExpression XlaExpression::XlaOp(xla::XlaOp value, DataType dtype) {
  XlaExpression e;
  e.kind_ = Kind::kXlaOp;
  e.dtype_ = dtype;
  e.handle_ = value;
  return e;
}

XlaExpression XlaExpression::Resource(XlaResource* resource) {
  XlaExpression e;
  e.kind_ = Kind::kResource;
  e.dtype_ = DT_RESOURCE;
  e.resource_ = resource;
  return e;
}

string XlaExpression::HumanString() const {
  switch (kind_) {
    case Kind::kInvalid:
      return "invalid";
    case Kind::kConstant:
      return "constant";
    case Kind::kXlaOp:
      return "xla_op";
    case Kind::kResource:
      return "resource";
  }
}

xla::XlaOp XlaExpression::AsXlaOp(xla::XlaBuilder* builder) const {
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    switch (kind_) {
      case Kind::kConstant: {
        xla::BorrowingLiteral literal;
        TF_RETURN_IF_ERROR(
            HostTensorToBorrowingLiteral(constant_value_, &literal));
        return xla::ConstantLiteral(builder, literal);
      }
      case Kind::kXlaOp:
        if (builder != handle_.builder()) {
          return errors::InvalidArgument(
              "Mismatched builders in XlaExpression::AsXlaOp");
        }
        return handle_;
      default:
        return errors::InvalidArgument("AsXlaOp called on XlaExpression: ",
                                       HumanString());
    }
  });
}

xla::StatusOr<absl::optional<Tensor>> XlaExpression::ResolveConstant(
    xla::Client* client) const {
  switch (kind()) {
    case Kind::kConstant:
      return {constant_value()};
    case Kind::kXlaOp:
      break;
    case Kind::kResource:
    case Kind::kInvalid:
      return errors::InvalidArgument(
          "ResolveConstant called on XlaExpression: ", HumanString());
  }

  TF_ASSIGN_OR_RETURN(bool is_constant,
                      handle().builder()->IsConstant(handle()));
  if (!is_constant) return {absl::nullopt};

  TF_ASSIGN_OR_RETURN(xla::XlaComputation constant_graph,
                      handle().builder()->BuildConstantSubGraph(handle()));

  TF_ASSIGN_OR_RETURN(TensorShape shape, GetShape());

  // The XLA layout is specified minor to major, and TensorFlow uses a major to
  // minor order.
  std::vector<int64> layout_indices(shape.dims());
  std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
  xla::Layout layout = xla::LayoutUtil::MakeLayout(layout_indices);
  TF_ASSIGN_OR_RETURN(xla::Literal literal,
                      client->ComputeConstant(constant_graph, &layout));
  Tensor tensor;
  TF_RETURN_IF_ERROR(LiteralToHostTensor(literal, dtype(), &tensor));
  return {tensor};
}

xla::StatusOr<TensorShape> XlaExpression::GetShape() const {
  switch (kind_) {
    case Kind::kConstant:
      return constant_value().shape();
    case Kind::kXlaOp: {
      TF_ASSIGN_OR_RETURN(xla::Shape xla_shape,
                          handle().builder()->GetShape(handle()));
      TensorShape shape;
      TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, &shape));
      return shape;
    }
    case Kind::kResource:
      return TensorShape({});
    case Kind::kInvalid:
      return errors::InvalidArgument(
          "GetShape() called on invalid XlaExpression");
  }
}

}  // namespace tensorflow
