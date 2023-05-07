/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/util.h"

#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace flex {

static constexpr char kResourceVariablePrefix[] = "tflite_resource_variable";

TfLiteStatus ConvertStatus(TfLiteContext* context,
                           const tensorflow::Status& status) {
  if (!status.ok()) {
    TF_LITE_KERNEL_LOG(context, "%s", tsl::NullTerminatedMessage(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus CopyShapeAndType(TfLiteContext* context,
                              const tensorflow::Tensor& src,
                              TfLiteTensor* tensor) {
  tensor->type = GetTensorFlowLiteType(static_cast<TF_DataType>(src.dtype()));
  if (tensor->type == kTfLiteNoType) {
    TF_LITE_KERNEL_LOG(context,
                       "TF Lite does not support TensorFlow data type: %s",
                       DataTypeString(src.dtype()).c_str());
    return kTfLiteError;
  }

  int num_dims = src.dims();
  TfLiteIntArray* shape = TfLiteIntArrayCreate(num_dims);
  for (int j = 0; j < num_dims; ++j) {
    // We need to cast from TensorFlow's int64 to TF Lite's int32. Let's
    // make sure there's no overflow.
    if (src.dim_size(j) >= std::numeric_limits<int>::max()) {
      TF_LITE_KERNEL_LOG(context,
                         "Dimension value in TensorFlow shape is larger than "
                         "supported by TF Lite");
      TfLiteIntArrayFree(shape);
      return kTfLiteError;
    }
    shape->data[j] = static_cast<int>(src.dim_size(j));
  }
  return context->ResizeTensor(context, tensor, shape);
}

TF_DataType GetTensorFlowDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return TF_FLOAT;
    case kTfLiteFloat32:
      return TF_FLOAT;
    case kTfLiteFloat16:
      return TF_HALF;
    case kTfLiteFloat64:
      return TF_DOUBLE;
    case kTfLiteInt16:
      return TF_INT16;
    case kTfLiteUInt16:
      return TF_UINT16;
    case kTfLiteInt32:
      return TF_INT32;
    case kTfLiteUInt32:
      return TF_UINT32;
    case kTfLiteInt4:
      // TODO(b/246806634): Tensorflow DT_INT4 type doesn't exist yet
      return TF_INT8;
    case kTfLiteUInt8:
      return TF_UINT8;
    case kTfLiteInt8:
      return TF_INT8;
    case kTfLiteInt64:
      return TF_INT64;
    case kTfLiteUInt64:
      return TF_UINT64;
    case kTfLiteComplex64:
      return TF_COMPLEX64;
    case kTfLiteComplex128:
      return TF_COMPLEX128;
    case kTfLiteString:
      return TF_STRING;
    case kTfLiteBool:
      return TF_BOOL;
    case kTfLiteResource:
      return TF_RESOURCE;
    case kTfLiteVariant:
      return TF_VARIANT;
  }
}

TfLiteType GetTensorFlowLiteType(TF_DataType type) {
  switch (type) {
    case TF_FLOAT:
      return kTfLiteFloat32;
    case TF_HALF:
      return kTfLiteFloat16;
    case TF_DOUBLE:
      return kTfLiteFloat64;
    case TF_INT16:
      return kTfLiteInt16;
    case TF_UINT16:
      return kTfLiteUInt16;
    case TF_INT32:
      return kTfLiteInt32;
    case TF_UINT32:
      return kTfLiteUInt32;
    case TF_UINT8:
      return kTfLiteUInt8;
    case TF_INT8:
      return kTfLiteInt8;
    case TF_INT64:
      return kTfLiteInt64;
    case TF_UINT64:
      return kTfLiteUInt64;
    case TF_COMPLEX64:
      return kTfLiteComplex64;
    case TF_COMPLEX128:
      return kTfLiteComplex128;
    case TF_STRING:
      return kTfLiteString;
    case TF_BOOL:
      return kTfLiteBool;
    case TF_RESOURCE:
      return kTfLiteResource;
    case TF_VARIANT:
      return kTfLiteVariant;
    default:
      return kTfLiteNoType;
  }
}

// Returns the TF data type name to be stored in the FunctionDef.
const char* TfLiteTypeToTfTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "invalid";
    case kTfLiteFloat32:
      return "float";
    case kTfLiteInt16:
      return "int16";
    case kTfLiteUInt16:
      return "uint16";
    case kTfLiteInt32:
      return "int32";
    case kTfLiteUInt32:
      return "uint32";
    case kTfLiteInt4:
      return "int4";
    case kTfLiteUInt8:
      return "uint8";
    case kTfLiteInt8:
      return "int8";
    case kTfLiteInt64:
      return "int64";
    case kTfLiteUInt64:
      return "uint64";
    case kTfLiteBool:
      return "bool";
    case kTfLiteComplex64:
      return "complex64";
    case kTfLiteComplex128:
      return "complex128";
    case kTfLiteString:
      return "string";
    case kTfLiteFloat16:
      return "float16";
    case kTfLiteFloat64:
      return "float64";
    case kTfLiteResource:
      return "resource";
    case kTfLiteVariant:
      return "variant";
  }
  return "invalid";
}

std::string TfLiteResourceIdentifier(const TfLiteTensor* tensor) {
  // TODO(b/199782192): Create a util function to get Resource ID from a TF Lite
  // resource tensor.
  const int resource_id = tensor->data.i32[0];
  return absl::StrFormat("%s:%d", kResourceVariablePrefix, resource_id);
}

bool GetTfLiteResourceTensorFromResourceHandle(
    const tensorflow::ResourceHandle& resource_handle, TfLiteTensor* tensor) {
  std::vector<std::string> parts = absl::StrSplit(resource_handle.name(), ':');
  if (parts.size() != 2) {
    return false;
  }
  const int kBytesRequired = sizeof(int32_t);
  TfLiteTensorRealloc(kBytesRequired, tensor);
  int resource_id;
  if (parts[0] == kResourceVariablePrefix &&
      absl::SimpleAtoi<int32_t>(parts[1], &resource_id)) {
    // TODO(b/199782192): Create a util function to set the Resource ID of
    // a TF Lite resource tensor.
    GetTensorData<int32_t>(tensor)[0] = resource_id;
    return true;
  }
  return false;
}

tensorflow::StatusOr<tensorflow::Tensor> CreateTfTensorFromTfLiteTensor(
    const TfLiteTensor* tflite_tensor) {
  if (IsResourceOrVariant(tflite_tensor)) {
    // Returns error if the input tflite tensor has variant or resource type.
    return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                              "Input tensor has resource or variant type.");
  }

  tensorflow::TensorShape shape;
  int num_dims = tflite_tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tflite_tensor->dims->data[i]);
  }

  tensorflow::Tensor tf_tensor(
      tensorflow::DataType(GetTensorFlowDataType(tflite_tensor->type)), shape);
  if (tf_tensor.dtype() == tensorflow::DataType::DT_STRING &&
      tf_tensor.data()) {
    tensorflow::tstring* buf =
        static_cast<tensorflow::tstring*>(tf_tensor.data());
    for (int i = 0; i < tflite::GetStringCount(tflite_tensor); ++buf, ++i) {
      auto ref = GetString(tflite_tensor, i);
      buf->assign(ref.str, ref.len);
    }
  } else {
    if (tf_tensor.tensor_data().size() != tflite_tensor->bytes) {
      return tensorflow::Status(
          absl::StatusCode::kInternal,
          "TfLiteTensor's size doesn't match the TF tensor's size.");
    }
    if (!tflite_tensor->data.raw) {
      return tensorflow::Status(absl::StatusCode::kInternal,
                                "TfLiteTensor's data field is null.");
    }
    std::memcpy(tf_tensor.data(), tflite_tensor->data.raw,
                tflite_tensor->bytes);
  }

  return tf_tensor;
}

}  // namespace flex
}  // namespace tflite
