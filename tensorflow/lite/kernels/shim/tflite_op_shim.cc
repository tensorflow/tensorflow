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
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"

namespace tflite {
namespace shim {

TfLiteInitContext::TfLiteInitContext(const TfLiteContext* context,
                                     const char* buffer, const size_t length)
    : attr_map_(
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
              .AsMap()) {}

absl::StatusOr<InitContext<TfLiteInitContext>::AttrValue>
TfLiteInitContext::GetAttr(const std::string& attr_name) const {
  const auto value = attr_map_[attr_name.data()];
  if (value.IsNull())
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name));
  AttrValue ret;
  switch (value.GetType()) {
    case ::flexbuffers::FBT_BOOL: {
      ret = value.AsBool();
      break;
    }
    case ::flexbuffers::FBT_INT: {
      ret = static_cast<int64_t>(value.AsInt64());
      break;
    }
    case ::flexbuffers::FBT_FLOAT: {
      ret = value.AsFloat();
      break;
    }
    case ::flexbuffers::FBT_STRING: {
      const auto str_val = value.AsString();
      ret = absl::string_view(str_val.c_str(), str_val.length());
      break;
    }
    default: {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported type for attribute: ", attr_name,
                       " with value: ", value.ToString()));
    }
  }
  return ret;
}

TfLiteInvokeContext::TfLiteInvokeContext(
    TfLiteContext* context, TfLiteNode* node,
    const std::vector<bool>& is_static_output)
    : context_(context), node_(node), is_static_output_(is_static_output) {}

ConstTensorViewOr TfLiteInvokeContext::GetInput(const int idx) const {
  // Scope is used to ensure tensor_view string contents are flushed
  const auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(
        absl::StrCat("input tensor is null during invocation. idx: ", idx));
  SH_ASSIGN_OR_RETURN(const TfLiteTensorView& tensor_view,
                      TensorView::New(tflite_tensor));
  return ConstTensorViewOr(
      absl::make_unique<const TfLiteTensorView>(tensor_view));
}

TensorViewOr TfLiteInvokeContext::GetOutput(const int idx,
                                            const Shape& output_shape) const {
  if (!output_shape.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("output_shape value should be populated. idx: ", idx));
  }
  auto* tflite_tensor = ::tflite::GetOutput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(
        absl::StrCat("output tensor is null during invocation. idx: ", idx));
  if (is_static_output_[idx]) {
    SH_RETURN_WITH_CONTEXT_IF_ERROR(
        AssertShapesEqual(tflite_tensor->dims, output_shape),
        ": output tensor idx: ", idx);
  } else {
    context_->ResizeTensor(context_, tflite_tensor,
                           ShapeToTfLiteShape(output_shape.value()));
  }
  SH_ASSIGN_OR_RETURN(TfLiteTensorView tensor_view,
                      TensorView::New(tflite_tensor));
  return TensorViewOr(absl::make_unique<TfLiteTensorView>(tensor_view));
}

absl::Status TfLiteInvokeContext::AssertShapesEqual(
    const TfLiteIntArray* dims, const Shape& output_shape) const {
  if (!output_shape.has_value()) return absl::OkStatus();
  const auto& shape = output_shape.value();
  if (dims->size != shape.size()) {
    return absl::FailedPreconditionError(
        ShapeMismatchErrorMsg(dims, output_shape));
  }
  for (int i = 0; i < dims->size; ++i)
    if (shape[i] != Shape::kUnknownDim && dims->data[i] != shape[i]) {
      return absl::FailedPreconditionError(
          ShapeMismatchErrorMsg(dims, output_shape));
    }
  return absl::OkStatus();
}

std::string TfLiteInvokeContext::ShapeMismatchErrorMsg(
    const TfLiteIntArray* actual_shape, const Shape& expected_shape) const {
  return absl::StrCat(
      "actual shape: ", TfLiteShapeToShape(actual_shape).ToString(),
      " vs. expected_shape: ", expected_shape.ToString());
}

TfLiteShapeInferenceContext::TfLiteShapeInferenceContext(
    TfLiteContext* context, TfLiteNode* node,
    std::vector<Shape>* inferred_shapes)
    : context_(context), node_(node), inferred_shapes_(inferred_shapes) {}

ShapeOr TfLiteShapeInferenceContext::GetInputShape(const int idx) const {
  auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(absl::StrCat(
        "input tensor is null during shape inference. idx: ", idx));
  return TfLiteShapeToShape(tflite_tensor->dims);
}

// A function object to set output shape information from a Shape
// object
absl::Status TfLiteShapeInferenceContext::SetOutputShape(const int idx,
                                                         const Shape& shape) {
  if (idx >= inferred_shapes_->size()) {
    return absl::InternalError(absl::StrCat("output idx out of bounds: ", idx,
                                            " >= ", inferred_shapes_->size()));
  }
  (*inferred_shapes_)[idx] = shape;
  return absl::OkStatus();
}

// A function object to read input tensor during shape inference as a TensorView
ConstTensorViewOr TfLiteShapeInferenceContext::GetInputTensor(
    const int idx) const {
  const auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(absl::StrCat(
        "input tensor is null during shape inference. idx: ", idx));
  if (::tflite::IsConstantTensor(tflite_tensor)) {
    SH_ASSIGN_OR_RETURN(const TfLiteTensorView& tensor_view,
                        TensorView::New(tflite_tensor));
    return ConstTensorViewOr(
        absl::make_unique<const TfLiteTensorView>(tensor_view));
  } else {
    return absl::FailedPreconditionError(absl::StrCat(
        "input tensor is unavailable during shape inference. idx: ", idx));
  }
}

TfLiteStatus StatusToTfLiteStatus(TfLiteContext* context,
                                  const absl::Status& status) {
  if (status.ok()) return kTfLiteOk;
  const auto error_string = std::string(status.message());
  TF_LITE_KERNEL_LOG(context, "error: %s", error_string.c_str());
  return kTfLiteError;
}

TfLiteIntArray* ShapeToTfLiteShape(const std::vector<int>& shape) {
  TfLiteIntArray* tflite_shape = TfLiteIntArrayCreate(shape.size());
  // TfLiteIntArray has a data array inside which is not a pointer so there's a
  // need for copy
  std::memcpy(tflite_shape->data, shape.data(), sizeof(int) * shape.size());
  return tflite_shape;
}

// Converts an int array representing shape in TFLite to Shape.
Shape TfLiteShapeToShape(const TfLiteIntArray* tflite_shape) {
  Shape ret(tflite_shape->size);
  std::memcpy(ret->data(), tflite_shape->data,
              sizeof(int) * tflite_shape->size);
  return ret;
}

}  // namespace shim
}  // namespace tflite
