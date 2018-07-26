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
#include "tensorflow/contrib/lite/delegates/eager/util.h"

namespace tflite {
namespace eager {

TfLiteStatus ConvertStatus(TfLiteContext* context,
                           const tensorflow::Status& status) {
  if (!status.ok()) {
    context->ReportError(context, "%s", status.error_message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus CopyShape(TfLiteContext* context, const tensorflow::Tensor& src,
                       TfLiteTensor* tensor) {
  int num_dims = src.dims();
  TfLiteIntArray* shape = TfLiteIntArrayCreate(num_dims);
  for (int j = 0; j < num_dims; ++j) {
    // We need to cast from TensorFlow's int64 to TF Lite's int32. Let's
    // make sure there's no overflow.
    if (src.dim_size(j) >= std::numeric_limits<int>::max()) {
      context->ReportError(context,
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
    case kTfLiteInt16:
      return TF_INT16;
    case kTfLiteInt32:
      return TF_INT32;
    case kTfLiteUInt8:
      return TF_UINT8;
    case kTfLiteInt64:
      return TF_INT64;
    case kTfLiteComplex64:
      return TF_COMPLEX64;
    case kTfLiteString:
      return TF_STRING;
    case kTfLiteBool:
      return TF_BOOL;
  }
}

}  // namespace eager
}  // namespace tflite
