/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

namespace tflite {
namespace shim {

namespace {
// Converts a TF AttrValue into a TF Shim AttrValue
absl::StatusOr<AttrValue> TfAttrValueToShimAttrValue(
    const ::tensorflow::AttrValue& attr_value) {
  AttrValue ret;
  switch (attr_value.value_case()) {
    case ::tensorflow::AttrValue::kB: {
      ret = attr_value.b();
      break;
    }
    case ::tensorflow::AttrValue::kI: {
      ret = attr_value.i();
      break;
    }
    case ::tensorflow::AttrValue::kF: {
      ret = attr_value.f();
      break;
    }
    case ::tensorflow::AttrValue::kS: {
      ret = attr_value.s();
      break;
    }
    default: {
      return absl::FailedPreconditionError(absl::StrCat(
          "Unsupported attribute type: ", attr_value.DebugString()));
    }
  }
  return ret;
}
}  // namespace

TfInitContext::TfInitContext(const ::tensorflow::OpKernelConstruction* context)
    : context_(context) {}

absl::StatusOr<AttrValue> TfInitContext::GetAttr(
    const std::string& attr_name) const {
  if (!context_->HasAttr(attr_name))
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name, "\nop def:\n",
                     context_->def().DebugString()));
  const auto& attr_value = context_->def().attr().at(attr_name);
  return TfAttrValueToShimAttrValue(attr_value);
}

TfInvokeContext::TfInvokeContext(::tensorflow::OpKernelContext* context)
    : context_(context) {}

ConstTensorViewOr TfInvokeContext::GetInput(const int idx) const {
  if (idx >= context_->num_inputs()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected idx < num_inputs. idx: ", idx,
                     " num_inputs: ", context_->num_inputs()));
  }
  const auto tf_tensor = context_->input(idx);
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(&tf_tensor));
  return std::make_unique<const TfTensorView>(tensor_view);
}

TensorViewOr TfInvokeContext::GetOutput(const int idx,
                                        const Shape& shape) const {
  tensorflow::Tensor* output_t = nullptr;
  if (!shape.has_value())
    return absl::InvalidArgumentError("Output shape needs to be specified.");
  std::vector<int64_t> shape_64(shape->size());
  for (int i = 0; i < shape->size(); ++i) shape_64[i] = (*shape)[i];
  auto status = context_->allocate_output(
      idx, ::tensorflow::TensorShape(shape_64), &output_t);
  if (!status.ok()) return ToAbslStatus(status);
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(output_t));
  return std::make_unique<TfTensorView>(std::move(tensor_view));
}

int TfInvokeContext::NumInputs() const { return context_->num_inputs(); }

int TfInvokeContext::NumOutputs() const { return context_->num_outputs(); }

TfShapeInferenceContext::TfShapeInferenceContext(
    ::tensorflow::shape_inference::InferenceContext* context)
    : context_(context) {}

ShapeOr TfShapeInferenceContext::GetInputShape(const int idx) const {
  std::vector<int> ret;
  const auto& shape = context_->input(idx);
  if (!context_->RankKnown(shape)) return Shape();
  ret.resize(context_->Rank(shape));
  for (int i = 0; i < ret.size(); ++i)
    ret[i] = context_->Value(context_->Dim(shape, i));
  return Shape(ret);
}

absl::Status TfShapeInferenceContext::SetOutputShape(const int idx,
                                                     const Shape& shape) {
  tensorflow::shape_inference::ShapeHandle output_shape;
  if (shape.has_value()) {
    std::vector<::tensorflow::shape_inference::DimensionHandle> tf_shape;
    tf_shape.reserve(shape.value().size());
    for (const auto dim : shape.value())
      tf_shape.emplace_back(context_->MakeDim(dim));
    output_shape = context_->MakeShape(tf_shape);
  } else {
    output_shape = context_->UnknownShape();
  }
  context_->set_output(idx, output_shape);
  return absl::OkStatus();
}

ConstTensorViewOr TfShapeInferenceContext::GetInputTensor(const int idx) const {
  const auto* tf_tensor = context_->input_tensor(idx);
  if (tf_tensor == nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Tensor is not available. idx: ", idx));
  }
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(tf_tensor));
  return std::make_unique<const TfTensorView>(tensor_view);
}

absl::StatusOr<AttrValue> TfShapeInferenceContext::GetAttr(
    const std::string& attr_name) const {
  const auto* tf_attr_value = context_->GetAttr(attr_name);
  if (tf_attr_value == nullptr)
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name));
  return TfAttrValueToShimAttrValue(*tf_attr_value);
}

int TfShapeInferenceContext::NumInputs() const {
  return context_->num_inputs();
}

int TfShapeInferenceContext::NumOutputs() const {
  return context_->num_outputs();
}

}  // namespace shim
}  // namespace tflite
