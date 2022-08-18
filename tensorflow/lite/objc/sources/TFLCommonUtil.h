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

#import <Foundation/Foundation.h>

#import "tensorflow/lite/objc/apis/TFLTensor.h"

NS_ASSUME_NONNULL_BEGIN

typedef struct TfLiteTensor TfLiteTensor;

@class TFLQuantizationParameters;

/** Gets the tensor data type from a c tensor. */
FOUNDATION_EXTERN TFLTensorDataType TFLTensorDataTypeFromCTensor(const TfLiteTensor *cTensor);

/** Gets the tensor name from a c tensor. */
FOUNDATION_EXTERN NSString *__nullable TFLTensorNameFromCTensor(const TfLiteTensor *cTensor);

/** Gets the quantization parameters from a c tensor. */
FOUNDATION_EXTERN TFLQuantizationParameters *__nullable
TFLQuantizationParamsFromCTensor(const TfLiteTensor *cTensor);

NS_ASSUME_NONNULL_END
