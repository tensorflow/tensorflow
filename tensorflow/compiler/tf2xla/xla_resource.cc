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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

/*static*/ absl::string_view XlaResource::KindToString(XlaResource::Kind kind) {
  switch (kind) {
    case XlaResource::kInvalid:
      return "invalid";
    case XlaResource::kVariable:
      return "variable";
    case XlaResource::kStack:
      return "stack";
    case XlaResource::kTensorArray:
      return "tensorarray";
  }
}

/*static*/ std::unique_ptr<XlaResource> XlaResource::CreateStack(
    string name, DataType type, int64_t max_size) {
  return std::make_unique<XlaResource>(
      XlaResource::kStack, /*arg_num=*/-1, std::move(name), type, TensorShape(),
      /*initial_value=*/xla::XlaOp(),
      /*max_array_size=*/max_size,
      /*tensor_array_gradients=*/std::set<string>{},
      /*tensor_array_multiple_writes_aggregate=*/false);
}

/*static*/ std::unique_ptr<XlaResource> XlaResource::CreateTensorArray(
    string name, DataType type, TensorShape shape, xla::XlaOp initial_value,
    int64_t max_array_size) {
  return std::make_unique<XlaResource>(
      XlaResource::kTensorArray, /*arg_num=*/-1, std::move(name), type, shape,
      initial_value, max_array_size,
      /*tensor_array_gradients=*/std::set<string>{},
      /*tensor_array_multiple_writes_aggregate=*/false);
}

XlaResource::XlaResource(
    Kind kind, int arg_num, string name, DataType type, TensorShape shape,
    xla::XlaOp initial_value, int64_t max_array_size,
    const std::set<string>& tensor_array_gradients,
    bool tensor_array_multiple_writes_aggregate,
    const std::optional<ManagedStackTrace>& definition_stack_trace)
    : kind_(kind),
      arg_num_(arg_num),
      name_(std::move(name)),
      type_(type),
      shape_(std::move(shape)),
      value_(initial_value),
      initial_value_(initial_value),
      max_array_size_(max_array_size),
      tensor_array_multiple_writes_aggregate_(
          tensor_array_multiple_writes_aggregate),
      definition_stack_trace_(definition_stack_trace) {
  CHECK(kind_ != kInvalid);

  for (const string& gradient : tensor_array_gradients) {
    tensor_array_gradients_[gradient].reset(new XlaResource(
        /*kind=*/kTensorArray, /*arg_num=*/-1,
        /*name=*/absl::StrCat("TensorArrayGrad: ", name_), type_, shape_,
        xla::XlaOp(), max_array_size_, /*tensor_array_gradients=*/{},
        /*tensor_array_multiple_writes_aggregate=*/true));
  }
}

absl::Status XlaResource::SetTypeAndShape(DataType type,
                                          const TensorShape& shape) {
  if (type == DT_INVALID) {
    return errors::InvalidArgument(
        "Attempted to set type of resource '", name_, "'' to an invalid type",
        DefinitionLocationMsg(definition_stack_trace_));
  }
  if (initialized() && type_ != type) {
    return errors::InvalidArgument(
        "Trying to assign variable with wrong dtype. Expected ",
        DataTypeString(type_), " got ", DataTypeString(type),
        DefinitionLocationMsg(definition_stack_trace_));
  }
  if (initialized() && shape_ != shape) {
    return errors::InvalidArgument(
        "Shape of resource ", name_,
        " cannot be changed after initialization: "
        "old shape was ",
        shape_.DebugString(), ", new shape is ", shape.DebugString(),
        DefinitionLocationMsg(definition_stack_trace_));
  }
  type_ = type;
  shape_ = shape;
  return absl::OkStatus();
}

absl::Status XlaResource::SetValue(const xla::XlaOp value) {
  if (type_ == DT_INVALID) {
    return errors::InvalidArgument(
        "Resource '", name_,
        "' must be initialized with a valid type before use.");
  }
  value_ = value;
  is_overwritten_ = true;
  return absl::OkStatus();
}

absl::Status XlaResource::SetZeroValue(xla::XlaBuilder* builder) {
  is_overwritten_ = true;
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
      TF_RETURN_IF_ERROR(ta_shape.AddDimWithStatus(max_array_size_));
      ta_shape.AppendShape(shape_);
      value_ = xla::Broadcast(XlaHelpers::Zero(builder, type_),
                              ta_shape.dim_sizes());
      break;
    }
    case kStack: {
      TensorShape ta_shape;
      TF_RETURN_IF_ERROR(ta_shape.AddDimWithStatus(max_array_size_));
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
  return absl::OkStatus();
}

absl::Status XlaResource::GetOrCreateTensorArrayGradient(
    const string& source, xla::XlaBuilder* builder,
    XlaResource** gradient_out) {
  VLOG(2) << "Gradient lookup for resource: " << name_
          << " gradient: " << source;
  TF_RET_CHECK(kind_ == kTensorArray);
  std::unique_ptr<XlaResource>& gradient = tensor_array_gradients_[source];
  if (!gradient) {
    TensorShape ta_shape;
    TF_RETURN_IF_ERROR(ta_shape.AddDimWithStatus(max_array_size_));
    ta_shape.AppendShape(shape_);
    xla::XlaOp gradient_value =
        xla::Broadcast(XlaHelpers::Zero(builder, type_), ta_shape.dim_sizes());
    gradient.reset(
        new XlaResource(/*kind=*/kTensorArray, /*arg_num=*/-1,
                        /*name=*/absl::StrCat("TensorArrayGrad: ", name_),
                        type_, shape_, gradient_value, max_array_size_,
                        /*tensor_array_gradients=*/{},
                        /*tensor_array_multiple_writes_aggregate=*/true));
  }
  *gradient_out = gradient.get();
  return absl::OkStatus();
}

absl::Status XlaResource::Pack(xla::XlaOp* pack,
                               xla::XlaBuilder* builder) const {
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
  return absl::OkStatus();
}

absl::Status XlaResource::SetFromPack(const std::set<string>& gradient_sources,
                                      const xla::XlaOp pack,
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
  return absl::OkStatus();
}

}  // namespace tensorflow
