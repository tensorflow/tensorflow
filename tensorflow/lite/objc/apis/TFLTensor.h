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

  /** 16-bit half precision floating point. */
  TFLTensorDataTypeFloat16,

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

  /** 8-bit signed integer. */
  TFLTensorDataTypeInt8,

  /** 64-bit double precision floating point. */
  TFLTensorDataTypeFloat64,
};

/**
 * An input or output tensor in a TensorFlow Lite model.
 *
 * @warning Each `TFLTensor` instance is associated with its provider, either a `TFLInterpreter` or
 * a `TFLSignatureRunner` instance. Multiple `TFLTensor` instances of the same TensorFlow Lite model
 * are associated with the same provider instance. As long as a `TFLTensor` instance is still in
 * use, its associated provider instance will not be deallocated.
 */
@interface TFLTensor : NSObject

/** Name of the tensor. */
@property(nonatomic, readonly, copy) NSString *name;

/** Data type of the tensor. */
@property(nonatomic, readonly) TFLTensorDataType dataType;

/** Parameters for asymmetric quantization. `nil` if the tensor does not use quantization. */
@property(nonatomic, readonly, nullable) TFLQuantizationParameters *quantizationParameters;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

/**
 * Copies the given data into an input tensor. This is allowed only for an input tensor and only
 * before the interpreter or the signature runner is invoked; otherwise an error will be returned.
 *
 * @param data The data to set. The byte size of the data must match what's required by the input
 *     tensor.
 * @param error An optional error parameter populated when there is an error in copying the data.
 *
 * @return Whether the data was copied into the input tensor successfully. Returns NO if an error
 *     occurred.
 */
- (BOOL)copyData:(NSData *)data error:(NSError **)error;

/**
 * Retrieves a copy of data in the tensor. For an output tensor, the data is only available after
 * the interpreter or signature runner invocation has successfully completed; otherwise an error
 * will be returned.
 *
 * @param error An optional error parameter populated when there is an error in retrieving the data.
 *
 * @return A copy of data in the tensor. `nil` if there is an error in retrieving the data or the
 *     data is not available.
 */
- (nullable NSData *)dataWithError:(NSError **)error;

/**
 * Retrieves the shape of the tensor, an array of positive unsigned integers containing the size
 * of each dimension. For example: the shape of [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] is
 * [2, 2, 3] (i.e. an array of 2 arrays of 2 arrays of 3 numbers).
 *
 * @param error An optional error parameter populated when there is an error in retrieving the
 *     shape.
 *
 * @return The shape of the tensor. `nil` if there is an error in retrieving the shape.
 */
- (nullable NSArray<NSNumber *> *)shapeWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
