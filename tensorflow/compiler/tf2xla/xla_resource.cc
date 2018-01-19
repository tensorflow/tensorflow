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

namespace tensorflow {

XlaResource::XlaResource(Kind kind, int arg_num, string name,
                         DataType initial_type,
                         const xla::ComputationDataHandle& initial_value)
    : kind_(kind),
      arg_num_(arg_num),
      name_(std::move(name)),
      type_(initial_type),
      value_(initial_value),
      initial_value_(initial_value) {
  CHECK(kind_ != kInvalid);
}

Status XlaResource::SetValue(DataType type,
                             const xla::ComputationDataHandle& value) {
  if (type_ == DT_INVALID && type == DT_INVALID) {
    return errors::InvalidArgument("Attempted to initialized resource ", name_,
                                   " to an invalid type");
  }
  if (type_ != DT_INVALID && type_ != type) {
    return errors::InvalidArgument("Type of resource ", name_,
                                   " cannot be changed after initialization: "
                                   "old type was ",
                                   DataTypeString(type_), ", new type is ",
                                   DataTypeString(type));
  }
  type_ = type;
  value_ = value;
  return Status::OK();
}

Status XlaResource::GetXlaShape(xla::ComputationBuilder* builder,
                                xla::Shape* shape) const {
  auto shape_or_status = builder->GetShape(value_);
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  *shape = *shape_or_status.ValueOrDie();
  return Status::OK();
}

Status XlaResource::GetShape(xla::ComputationBuilder* builder,
                             TensorShape* shape) const {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(GetXlaShape(builder, &xla_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, shape));
  return Status::OK();
}

Status XlaResource::GetOrCreateTensorArrayGradient(
    const string& source, xla::ComputationBuilder* builder,
    XlaResource** gradient_out) {
  VLOG(2) << "Gradient lookup for resource: " << name_
          << " gradient: " << source;
  TF_RET_CHECK(kind_ == kTensorArray);
  std::unique_ptr<XlaResource>& gradient = tensor_array_gradients_[source];
  if (!gradient) {
    TensorShape ta_shape;
    TF_RETURN_IF_ERROR(GetShape(builder, &ta_shape));
    xla::ComputationDataHandle gradient_value = builder->Broadcast(
        XlaHelpers::Zero(builder, type_), ta_shape.dim_sizes());
    gradient.reset(
        new XlaResource(/*kind=*/kTensorArray, /*arg_num=*/-1,
                        /*name=*/strings::StrCat("TensorArrayGrad: ", name_),
                        type_, gradient_value));
    gradient->tensor_array_size_ = tensor_array_size_;
  }
  *gradient_out = gradient.get();
  return Status::OK();
}

Status XlaResource::PackedShape(xla::ComputationBuilder* builder,
                                xla::Shape* packed_shape) const {
  if (tensor_array_gradients_.empty()) {
    return GetXlaShape(builder, packed_shape);
  }
  TF_RET_CHECK(kind_ == kTensorArray);
  std::vector<xla::Shape> elem_shapes(1 + tensor_array_gradients_.size());
  int pos = 0;
  TF_RETURN_IF_ERROR(GetXlaShape(builder, &elem_shapes[pos++]));
  for (const auto& gradient : tensor_array_gradients_) {
    TF_RETURN_IF_ERROR(
        gradient.second->GetXlaShape(builder, &elem_shapes[pos++]));
  }
  *packed_shape = xla::ShapeUtil::MakeTupleShape(elem_shapes);
  return Status::OK();
}

Status XlaResource::Pack(xla::ComputationDataHandle* pack,
                         xla::ComputationBuilder* builder) const {
  if (tensor_array_gradients_.empty()) {
    *pack = value_;
  } else {
    TF_RET_CHECK(kind_ == kTensorArray);
    std::vector<xla::ComputationDataHandle> elems;
    elems.push_back(value_);
    for (const auto& gradient : tensor_array_gradients_) {
      elems.push_back(gradient.second->value_);
    }
    *pack = builder->Tuple(elems);
  }
  return Status::OK();
}

Status XlaResource::SetFromPack(const std::set<string>& gradient_sources,
                                const xla::ComputationDataHandle& pack,
                                bool reset_initial_values,
                                xla::ComputationBuilder* builder) {
  if (gradient_sources.empty()) {
    value_ = pack;
  } else {
    TF_RET_CHECK(kind_ == kTensorArray);
    int pos = 0;
    value_ = builder->GetTupleElement(pack, pos++);
    for (const auto& source : gradient_sources) {
      XlaResource* gradient;
      TF_RETURN_IF_ERROR(
          GetOrCreateTensorArrayGradient(source, builder, &gradient));
      gradient->value_ = builder->GetTupleElement(pack, pos++);
      if (reset_initial_values) {
        gradient->initial_value_ = gradient->value_;
      }
    }
  }
  if (reset_initial_values) {
    initial_value_ = value_;
  }
  return Status::OK();
}

}  // namespace tensorflow
