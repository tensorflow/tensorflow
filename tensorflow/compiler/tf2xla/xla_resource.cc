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

#include "tensorflow/compiler/tf2xla/xla_resource.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {

XlaResource::XlaResource(Kind kind, int arg_num, string name, DataType type,
                         TensorShape shape, const xla::XlaOp& initial_value,
                         int64 tensor_array_size,
                         const std::set<string>& tensor_array_gradients)
    : kind_(kind),
      arg_num_(arg_num),
      name_(std::move(name)),
      type_(type),
      shape_(std::move(shape)),
      value_(initial_value),
      initial_value_(initial_value),
      tensor_array_size_(tensor_array_size) {
  CHECK(kind_ != kInvalid);

  for (const string& gradient : tensor_array_gradients) {
    tensor_array_gradients_[gradient].reset(new XlaResource(
        /*kind=*/kTensorArray, /*arg_num=*/-1,
        /*name=*/strings::StrCat("TensorArrayGrad: ", name_), type_, shape_,
        xla::XlaOp(), tensor_array_size_, /*tensor_array_gradients=*/{}));
  }
}

Status XlaResource::SetTypeAndShape(DataType type, const TensorShape& shape) {
  if (type == DT_INVALID) {
    return errors::InvalidArgument("Attempted to set type of resource '", name_,
                                   "'' to an invalid type");
  }
  if (initialized() && type_ != type) {
    return errors::InvalidArgument("Type of resource ", name_,
                                   " cannot be changed after initialization: "
                                   "old type was ",
                                   DataTypeString(type_), ", new type is ",
                                   DataTypeString(type));
  }
  if (initialized() && shape_ != shape) {
    return errors::InvalidArgument("Shape of resource ", name_,
                                   " cannot be changed after initialization: "
                                   "old shape was ",
                                   shape_.DebugString(), ", new shape is ",
                                   shape.DebugString());
  }
  type_ = type;
  shape_ = shape;
  return Status::OK();
}

Status XlaResource::SetValue(const xla::XlaOp& value) {
  if (type_ == DT_INVALID) {
    return errors::InvalidArgument(
        "Resource '", name_,
        "' must be initialized with a valid type before use.");
  }
  value_ = value;
  return Status::OK();
}

Status XlaResource::SetZeroValue(xla::XlaBuilder* builder) {
  if (type_ == DT_INVALID) {
    return errors::InvalidArgument(
        "Resource '", name_,
        "' must be initialized with a valid type before use.");
  }
  switch (kind_) {
    case kVariable: {
      value_ =
          xla::Broadcast(XlaHelpers::Zero(builder, type_), shape_.dim_sizes());
      break;
    }
    case kTensorArray: {
      TensorShape ta_shape;
      ta_shape.AddDim(tensor_array_size_);
      ta_shape.AppendShape(shape_);
      value_ = xla::Broadcast(XlaHelpers::Zero(builder, type_),
                              ta_shape.dim_sizes());
      break;
    }
    case kStack: {
      TensorShape ta_shape;
      ta_shape.AddDim(tensor_array_size_);
      ta_shape.AppendShape(shape_);
      value_ =
          xla::Tuple(builder, {xla::Broadcast(XlaHelpers::Zero(builder, type_),
                                              ta_shape.dim_sizes()),
                               xla::ConstantR0<int32>(builder, 0)});
      break;
    }

    case kInvalid:
    default:
      LOG(FATAL) << "Invalid resource type";
  }
  return Status::OK();
}

Status XlaResource::GetOrCreateTensorArrayGradient(const string& source,
                                                   xla::XlaBuilder* builder,
                                                   XlaResource** gradient_out) {
  VLOG(2) << "Gradient lookup for resource: " << name_
          << " gradient: " << source;
  TF_RET_CHECK(kind_ == kTensorArray);
  std::unique_ptr<XlaResource>& gradient = tensor_array_gradients_[source];
  if (!gradient) {
    TensorShape ta_shape;
    ta_shape.AddDim(tensor_array_size_);
    ta_shape.AppendShape(shape_);
    xla::XlaOp gradient_value =
        xla::Broadcast(XlaHelpers::Zero(builder, type_), ta_shape.dim_sizes());
    gradient.reset(
        new XlaResource(/*kind=*/kTensorArray, /*arg_num=*/-1,
                        /*name=*/strings::StrCat("TensorArrayGrad: ", name_),
                        type_, shape_, gradient_value, tensor_array_size_,
                        /*tensor_array_gradients=*/{}));
  }
  *gradient_out = gradient.get();
  return Status::OK();
}

Status XlaResource::Pack(xla::XlaOp* pack, xla::XlaBuilder* builder) const {
  if (tensor_array_gradients_.empty()) {
    *pack = value_;
  } else {
    TF_RET_CHECK(kind_ == kTensorArray);
    std::vector<xla::XlaOp> elems;
    elems.push_back(value_);
    for (const auto& gradient : tensor_array_gradients_) {
      elems.push_back(gradient.second->value_);
    }
    *pack = xla::Tuple(builder, elems);
  }
  return Status::OK();
}

Status XlaResource::SetFromPack(const std::set<string>& gradient_sources,
                                const xla::XlaOp& pack,
                                xla::XlaBuilder* builder) {
  if (gradient_sources.empty()) {
    if (!initialized()) {
      initial_value_ = pack;
    }
    value_ = pack;
  } else {
    TF_RET_CHECK(kind_ == kTensorArray);
    int pos = 0;
    auto v = xla::GetTupleElement(pack, pos++);
    if (!initialized()) {
      initial_value_ = v;
    }
    value_ = v;

    for (const auto& source : gradient_sources) {
      XlaResource* gradient;
      TF_RETURN_IF_ERROR(
          GetOrCreateTensorArrayGradient(source, builder, &gradient));
      auto v = xla::GetTupleElement(pack, pos++);
      if (!gradient->initialized()) {
        gradient->initial_value_ = v;
      }
      gradient->value_ = v;
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
