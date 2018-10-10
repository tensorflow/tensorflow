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

#import <Foundation/Foundation.h>

@class TFLQuantizationParameters;

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLTensorDataType
 * This enum specifies supported TensorFlow Lite tensor data types.
 */
typedef NS_ENUM(NSUInteger, TFLTensorDataType) {
  /** Tensor data type not available. This indicates an error with the model. */
  TFLTensorDataTypeNoType,

  /** 32-bit single precision floating point. */
  TFLTensorDataTypeFloat32,

  /** 32-bit signed integer. */
  TFLTensorDataTypeInt32,

  /** 8-bit unsigned integer. */
  TFLTensorDataTypeUInt8,

  /** 64-bit signed integer. */
  TFLTensorDataTypeInt64,

  /** Boolean. */
  TFLTensorDataTypeBool,

  /** 16-bit signed integer. */
  TFLTensorDataTypeInt16,
};

/**
 * An input or output tensor in a TensorFlow Lite model.
 */
@interface TFLTensor : NSObject

/** Name of the tensor. */
@property(nonatomic, readonly, copy) NSString *name;

/** Data type of the tensor. */
@property(nonatomic, readonly) TFLTensorDataType dataType;

/**
 * Shape of the tensor, an array of positive unsigned integer(s) containing the size of each
 * dimension. For example: the shape of [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] is
 * [2, 2, 3].
 */
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;

/** Number of bytes for the tensor data. */
@property(nonatomic, readonly) NSUInteger byteSize;

/** Parameters for asymmetric quantization. `nil` if the tensor does not use quantization. */
@property(nonatomic, readonly, nullable) TFLQuantizationParameters *quantizationParameters;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
