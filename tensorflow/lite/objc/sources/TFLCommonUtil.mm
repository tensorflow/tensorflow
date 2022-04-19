// Copyright 2022 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "TFLCommonUtil.h"

#import "TFLQuantizationParameters+Internal.h"
#import "tensorflow/lite/objc/apis/TFLTensor.h"

#include "tensorflow/lite/c/c_api.h"

NS_ASSUME_NONNULL_BEGIN

TFLTensorDataType TFLTensorDataTypeFromCTensor(const TfLiteTensor *cTensor) {
  TfLiteType cTensorType = TfLiteTensorType(cTensor);
  switch (cTensorType) {
    case kTfLiteFloat32:
      return TFLTensorDataTypeFloat32;
    case kTfLiteFloat16:
      return TFLTensorDataTypeFloat16;
    case kTfLiteFloat64:
      return TFLTensorDataTypeFloat64;
    case kTfLiteInt32:
      return TFLTensorDataTypeInt32;
    case kTfLiteUInt8:
      return TFLTensorDataTypeUInt8;
    case kTfLiteInt8:
      return TFLTensorDataTypeInt8;
    case kTfLiteInt64:
      return TFLTensorDataTypeInt64;
    case kTfLiteBool:
      return TFLTensorDataTypeBool;
    case kTfLiteInt16:
      return TFLTensorDataTypeInt16;
    case kTfLiteNoType:
    case kTfLiteString:
    case kTfLiteComplex64:
    case kTfLiteComplex128:
    case kTfLiteUInt16:
    case kTfLiteUInt32:
    case kTfLiteUInt64:
    case kTfLiteResource:
    case kTfLiteVariant:
      // kTfLiteString, kTfLiteUInt64, kTfLiteComplex64, kTfLiteComplex128,
      // kTfLiteResource and kTfLiteVariant are not supported in TensorFlow Lite
      // Objc API.
      return TFLTensorDataTypeNoType;
  }
}

NSString *__nullable TFLTensorNameFromCTensor(const TfLiteTensor *cTensor) {
  const char *cName = TfLiteTensorName(cTensor);
  if (cName == nullptr) return nil;
  return [NSString stringWithUTF8String:cName];
}

TFLQuantizationParameters *__nullable
TFLQuantizationParamsFromCTensor(const TfLiteTensor *cTensor) {
  TfLiteQuantizationParams cParams = TfLiteTensorQuantizationParams(cTensor);
  TFLQuantizationParameters *quantizationParams;

  // TODO(b/119735362): Update this check once the TfLiteQuantizationParams struct has a mode.
  if (cParams.scale != 0.0) {
    quantizationParams = [[TFLQuantizationParameters alloc] initWithScale:cParams.scale
                                                                zeroPoint:cParams.zero_point];
  }
  return quantizationParams;
}

NS_ASSUME_NONNULL_END
