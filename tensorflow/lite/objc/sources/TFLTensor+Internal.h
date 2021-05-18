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

#import "tensorflow/lite/objc/apis/TFLTensor.h"

@class TFLInterpreter;

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLTensorType
 * This enum specifies input or output tensor types.
 */
typedef NS_ENUM(NSUInteger, TFLTensorType) {
  /** Input tensor type. */
  TFLTensorTypeInput,

  /** Output tensor type. */
  TFLTensorTypeOutput,
};

@interface TFLTensor (Internal)

/** Input or output tensor type. */
@property(nonatomic, readonly) TFLTensorType type;

/** Index of the tensor. */
@property(nonatomic, readonly) NSUInteger index;

/**
 * Initializes a `TFLTensor` with the given interpreter, name, data type, and quantization
 * parameters.
 *
 * @param interpreter Interpreter backing the tensor.
 * @param type Input or output tensor type.
 * @param index Index of the tensor.
 * @param name Name of the tensor.
 * @param dataType Data type of the tensor.
 * @param quantizationParameters Quantization parameters of the tensor. `nil` if the tensor does not
 *     use quantization.
 *
 * @return A new instance of `TFLTensor` with the given name, data type, shape, and quantization
 *     parameters.
 */
- (instancetype)initWithInterpreter:(TFLInterpreter *)interpreter
                               type:(TFLTensorType)type
                              index:(NSUInteger)index
                               name:(NSString *)name
                           dataType:(TFLTensorDataType)dataType
             quantizationParameters:(nullable TFLQuantizationParameters *)quantizationParameters;

/**
 * Returns the string name of the given input or output tensor type.
 *
 * @param type Input or output tensor type.
 *
 * @return The string name of the given input or output tensor type.
 */
+ (NSString *)stringForTensorType:(TFLTensorType)type;

@end

NS_ASSUME_NONNULL_END
