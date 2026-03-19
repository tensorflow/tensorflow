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

#include "tensorflow/compiler/tf2xla/xla_argument.h"

#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/tf2xla/xla_argument.pb.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {

bool XlaArgument::operator==(const XlaArgument& other) const {
  if (std::tie(kind, resource_kind, type, name, initialized, max_array_size,
               tensor_array_gradients) !=
      std::tie(other.kind, other.resource_kind, other.type, other.name,
               other.initialized, other.max_array_size,
               other.tensor_array_gradients)) {
    return false;
  }
  if (absl::holds_alternative<xla::Shape>(shape)) {
    if (!absl::holds_alternative<xla::Shape>(other.shape)) {
      return false;
    }
    if (!xla::Shape::Equal()(absl::get<xla::Shape>(shape),
                             absl::get<xla::Shape>(other.shape))) {
      return false;
    }
  } else {
    if (!absl::holds_alternative<TensorShape>(other.shape)) {
      return false;
    }
    if (absl::get<TensorShape>(shape) != absl::get<TensorShape>(other.shape)) {
      return false;
    }
  }
  if (constant_value.shape() != other.constant_value.shape()) {
    return false;
  }
  if (is_same_data_across_replicas != other.is_same_data_across_replicas) {
    return false;
  }
  return constant_value.tensor_data() == other.constant_value.tensor_data();
}

tf2xla::XlaArgumentProto XlaArgument::ToProto() const {
  tf2xla::XlaArgumentProto proto;
  proto.set_kind(static_cast<int>(kind));
  proto.set_type(type);
  if (std::holds_alternative<xla::Shape>(shape)) {
    *proto.mutable_shape()->mutable_xla_shape() =
        std::get<xla::Shape>(shape).ToProto();
  } else if (std::holds_alternative<TensorShape>(shape)) {
    *proto.mutable_shape()->mutable_tensor_shape() =
        std::get<TensorShape>(shape).AsProto();
  }
  constant_value.AsProtoTensorContent(proto.mutable_constant_value());
  if (value_bound.has_value()) {
    value_bound->AsProtoTensorContent(proto.mutable_value_bound());
  }
  if (value_dynamism.has_value()) {
    value_dynamism->AsProtoTensorContent(proto.mutable_value_dynamism());
  }
  proto.set_name(name);
  proto.set_node_name(node_name);
  proto.set_resource_kind(static_cast<int>(resource_kind));
  proto.set_initialized(initialized);
  proto.set_fast_mem(fast_mem);
  proto.set_max_array_size(max_array_size);
  proto.mutable_tensor_array_gradients()->Add(tensor_array_gradients.begin(),
                                              tensor_array_gradients.end());
  proto.set_is_same_data_across_replicas(is_same_data_across_replicas);
  proto.set_requires_broadcast(requires_broadcast);
  return proto;
}

absl::StatusOr<XlaArgument> XlaArgument::FromProto(
    const tf2xla::XlaArgumentProto& proto) {
  XlaArgument arg;
  arg.kind = static_cast<XlaArgument::Kind>(proto.kind());
  arg.type = proto.type();
  if (proto.has_shape()) {
    if (proto.shape().has_xla_shape()) {
      TF_ASSIGN_OR_RETURN(arg.shape,
                          xla::Shape::FromProto(proto.shape().xla_shape()));
    } else if (proto.shape().has_tensor_shape()) {
      arg.shape = TensorShape(proto.shape().tensor_shape());
    } else {
      return absl::InvalidArgumentError("Shape is not set.");
    }
  }
  if (proto.has_constant_value() &&
      !arg.constant_value.FromProto(proto.constant_value())) {
    return absl::InvalidArgumentError("Constant value is invalid.");
  }
  if (proto.has_value_bound()) {
    Tensor value_bound;
    if (!value_bound.FromProto(proto.value_bound())) {
      return absl::InvalidArgumentError("Value bound is invalid.");
    }
    arg.value_bound = value_bound;
  }
  if (proto.has_value_dynamism()) {
    Tensor value_dynamism;
    if (!value_dynamism.FromProto(proto.value_dynamism())) {
      return absl::InvalidArgumentError("Value dynamism is invalid.");
    }
    arg.value_dynamism = value_dynamism;
  }
  arg.name = proto.name();
  arg.node_name = proto.node_name();
  arg.resource_kind = static_cast<XlaResource::Kind>(proto.resource_kind());
  arg.initialized = proto.initialized();
  arg.fast_mem = proto.fast_mem();
  arg.max_array_size = proto.max_array_size();
  arg.tensor_array_gradients.insert(proto.tensor_array_gradients().begin(),
                                    proto.tensor_array_gradients().end());
  arg.is_same_data_across_replicas = proto.is_same_data_across_replicas();
  arg.requires_broadcast = proto.requires_broadcast();
  return arg;
}

bool AnyUninitializedResourceArg(absl::Span<const XlaArgument> args) {
  return llvm::any_of(args, [](const XlaArgument& arg) {
    return arg.kind == XlaArgument::kResource && arg.type == DT_INVALID;
  });
}

}  // end namespace tensorflow
