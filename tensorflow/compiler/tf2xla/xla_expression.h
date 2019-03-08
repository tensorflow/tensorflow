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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// A XlaExpression represents a symbolic TensorFlow value in a TF->XLA
// compilation.
// An expression is one of:
// * a constant tensor.
// * an xla::XlaOp, representing a symbolic XLA value.
// * a resource, e.g., a variable, represented as an XlaResource pointer.
// * a tensor list, represented by a tuple of tensors and the list length.
//
// Constant tensors are mostly an optimization to avoid passing large constants
// to XLA, but are also sometimes used to represent tensors that have no XLA
// representation, for example, DT_STRING tensors. A canonical use case might be
// an error message string.
//
// Tensor lists are very similar to xla::XlaOp, however they require some
// specific logic around shape management since the tuples are not supported by
// TensorFlow.
class XlaExpression {
 public:
  enum class Kind {
    kInvalid,
    kConstant,
    kXlaOp,
    kResource,
    kTensorList,
  };

  XlaExpression();
  XlaExpression(const XlaExpression&) = default;
  XlaExpression& operator=(const XlaExpression&) = default;

  // Builds an invalid expression. (Same as the default constructor, but makes
  // the intent clearer.)
  static XlaExpression Invalid();

  // Builds a constant XLA expression.
  static XlaExpression Constant(Tensor value);

  // Builds a XlaOp expression. Since the mapping from TF data types to XLA
  // types is not 1-1, the TF type must also be provided; in general it cannot
  // be derived from the XLA type.
  static XlaExpression XlaOp(xla::XlaOp value, DataType dtype);

  // Builds a tensor list expression.
  static XlaExpression TensorList(xla::XlaOp tensor_list);

  // Builds a resource expression.
  static XlaExpression Resource(XlaResource* resource);

  Kind kind() const { return kind_; }

  DataType dtype() const { return dtype_; }

  // handle() returns the XlaOp that backs a kXlaOp expression.
  const xla::XlaOp& handle() const { return handle_; }

  const Tensor& constant_value() const { return constant_value_; }

  XlaResource* resource() const { return resource_; }

  // Returns a human-readable summary of the expression.
  string HumanString() const;

  // Returns the value of a kConstant or kXlaOp as an xla::XlaOp. Returns
  // an erroneous XlaOp if the expression is not a constant or an expression.
  xla::XlaOp AsXlaOp(xla::XlaBuilder* builder) const;

  // If a kXlaOp or kConstant expression can be resolved to a compile-time
  // constant, returns the value as a host-memory Tensor. Returns an empty
  // optional if it cannot be resolved. Returns an error if passed a resource
  // expression.
  xla::StatusOr<absl::optional<Tensor>> ResolveConstant(
      xla::Client* client) const;

  // Returns the shape of the tensor.
  // The shape of a resource is the shape of a resource handle (i.e., a scalar),
  // not the shape of the resource's value.
  xla::StatusOr<TensorShape> GetShape() const;

 private:
  Kind kind_ = Kind::kInvalid;

  DataType dtype_ = DT_INVALID;

  // The XLA handle of the expression's computation, if kind_ == kXlaOp or
  // a tuple expression if kind_ == kTensorList.
  xla::XlaOp handle_;

  // The value of the constant, if kind_ == kConstant.
  Tensor constant_value_;

  // The resource, if kind_ == kResource. Not owned.
  XlaResource* resource_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_
