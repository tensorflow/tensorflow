// Copyright 2018 Google Inc. All rights reserved.
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

#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLTensor (Internal)

/**
 * Initializes a `TFLTensor` with the given name, data type, shape, and quantization parameters.
 *
 * @param name Name of the tensor.
 * @param dataType Data type of the tensor.
 * @param shape Shape of the tensor.
 * @param byteSize Size of the tensor data in number of bytes.
 * @param quantizationParameters Quantization parameters of the tensor. `nil` if the tensor does not
 *     use quantization.
 *
 * @return A new instance of `TFLTensor` with the given name, data type, shape, and quantization
 *     parameters.
 */
- (instancetype)initWithName:(NSString *)name
                    dataType:(TFLTensorDataType)dataType
                       shape:(NSArray<NSNumber *> *)shape
                    byteSize:(NSUInteger)byteSize
      quantizationParameters:(nullable TFLQuantizationParameters *)quantizationParameters;

@end

NS_ASSUME_NONNULL_END
