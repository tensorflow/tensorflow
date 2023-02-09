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
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace shim {

namespace internal {
absl::StatusOr<AttrValue> GetAttr(const flexbuffers::Map* attr_map,
                                  const std::string& attr_name) {
  const auto value = (*attr_map)[attr_name.data()];
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
}  // namespace internal

TfLiteInitContext::TfLiteInitContext(const TfLiteContext* context,
                                     const flexbuffers::Map* attr_map)
    : attr_map_(attr_map) {}

absl::StatusOr<AttrValue> TfLiteInitContext::GetAttr(
    const std::string& attr_name) const {
  return internal::GetAttr(attr_map_, attr_name);
}

TfLiteInvokeContext::TfLiteInvokeContext(TfLiteContext* context,
                                         TfLiteNode* node)
    : context_(context), node_(node) {}

ConstTensorViewOr TfLiteInvokeContext::GetInput(const int idx) const {
  // Scope is used to ensure tensor_view string contents are flushed
  const auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(
        absl::StrCat("input tensor is null during invocation. idx: ", idx));
  SH_ASSIGN_OR_RETURN(const TfLiteTensorView& tensor_view,
                      TensorView::New(tflite_tensor));
  return std::make_unique<const TfLiteTensorView>(tensor_view);
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
  if (tflite_tensor->data.raw == nullptr ||
      tflite_tensor->allocation_type == kTfLiteDynamic) {
    // Clear out string tensor so previous values are not copied.
    if (tflite_tensor->type == kTfLiteString) {
      tflite::DynamicBuffer buf;
      buf.WriteToTensor(tflite_tensor, /*new_shape=*/nullptr);
    }
    TfLiteIntArray* output_shape_array =
        ShapeToTfLiteShape(output_shape.value());
    context_->ResizeTensor(context_, tflite_tensor, output_shape_array);
  } else {
    DCHECK(TfLiteShapeToShape(tflite_tensor->dims) == output_shape);
  }
  SH_ASSIGN_OR_RETURN(TfLiteTensorView tensor_view,
                      TensorView::New(tflite_tensor));
  return std::make_unique<TfLiteTensorView>(std::move(tensor_view));
}

int TfLiteInvokeContext::NumInputs() const {
  return ::tflite::NumInputs(node_);
}

int TfLiteInvokeContext::NumOutputs() const {
  return ::tflite::NumOutputs(node_);
}

TfLiteShapeInferenceContext::TfLiteShapeInferenceContext(
    TfLiteContext* context, TfLiteNode* node, const flexbuffers::Map* attr_map,
    std::vector<Shape>* inferred_shapes)
    : context_(context),
      node_(node),
      attr_map_(attr_map),
      inferred_shapes_(inferred_shapes) {}

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
    return std::make_unique<const TfLiteTensorView>(tensor_view);
  } else {
    return absl::FailedPreconditionError(absl::StrCat(
        "input tensor is unavailable during shape inference. idx: ", idx));
  }
}

absl::StatusOr<AttrValue> TfLiteShapeInferenceContext::GetAttr(
    const std::string& attr_name) const {
  return internal::GetAttr(attr_map_, attr_name);
}

int TfLiteShapeInferenceContext::NumInputs() const {
  return ::tflite::NumInputs(node_);
}

int TfLiteShapeInferenceContext::NumOutputs() const {
  return ::tflite::NumOutputs(node_);
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
