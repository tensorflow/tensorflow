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

@class TFLInterpreterOptions;
@class TFLTensor;

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLInterpreterErrorCode
 * This enum specifies various error codes related to `TFLInterpreter`.
 */
typedef NS_ENUM(NSUInteger, TFLInterpreterErrorCode) {
  /** Provided tensor index is invalid. */
  TFLInterpreterErrorCodeInvalidTensorIndex,

  /** Input data has invalid byte size. */
  TFLInterpreterErrorCodeInvalidInputByteSize,

  /** Provided shape is invalid. It must be a non-empty array of positive unsigned integers. */
  TFLInterpreterErrorCodeInvalidShape,

  /** Provided model cannot be loaded. */
  TFLInterpreterErrorCodeFailedToLoadModel,

  /** Failed to create `TFLInterpreter`. */
  TFLInterpreterErrorCodeFailedToCreateInterpreter,

  /** Failed to invoke `TFLInterpreter`. */
  TFLInterpreterErrorCodeFailedToInvoke,

  /** Failed to retrieve a tensor. */
  TFLInterpreterErrorCodeFailedToGetTensor,

  /** Failed to resize an input tensor. */
  TFLInterpreterErrorCodeFailedToResizeInputTensor,

  /** Failed to copy data into an input tensor. */
  TFLInterpreterErrorCodeFailedToCopyDataToInputTensor,

  /** Failed to get data from an output tensor. */
  TFLInterpreterErrorCodeFailedToGetDataFromOutputTensor,

  /** Failed to allocate memory for tensors. */
  TFLInterpreterErrorCodeFailedToAllocateTensors,

  /** Operaton not allowed without allocating memory for tensors first. */
  TFLInterpreterErrorCodeAllocateTensorsRequired,

  /** Operaton not allowed without invoking the interpreter first. */
  TFLInterpreterErrorCodeInvokeInterpreterRequired,
};

/**
 * A TensorFlow Lite model interpreter.
 */
@interface TFLInterpreter : NSObject

/** The total number of input tensors. 0 if the interpreter creation failed. */
@property(nonatomic, readonly) NSUInteger inputTensorCount;

/** The total number of output tensors. 0 if the interpreter creation failed. */
@property(nonatomic, readonly) NSUInteger outputTensorCount;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path and the
 * default interpreter options.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `TFLInterpreter` with the given model and the default interpreter
 *     options.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path and
 * options.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param options Options to use for configuring the TensorFlow Lite interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and options.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath
                          options:(TFLInterpreterOptions *)options NS_DESIGNATED_INITIALIZER;

/**
 * Invokes the interpreter to run inference.
 *
 * @param error An optional error parameter populated when there is an error in invoking the
 *     interpreter.
 *
 * @return Whether the invocation is successful. Returns NO if an error occurred.
 */
- (BOOL)invokeWithError:(NSError **)error;

/**
 * Returns the input tensor at the given index.
 *
 * @param index The index of an input tensor.
 * @param error An optional error parameter populated when there is an error in looking up the input
 *     tensor.
 *
 * @return The input tensor at the given index. `nil` if there is an error.
 */
- (nullable TFLTensor *)inputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Returns the output tensor at the given index.
 *
 * @param index The index of an output tensor.
 * @param error An optional error parameter populated when there is an error in looking up the
 *     output tensor.
 *
 * @return The output tensor at the given index. `nil` if there is an error.
 */
- (nullable TFLTensor *)outputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Resizes the input tensor at the given index to the specified shape (an array of positive unsigned
 * integers).
 *
 * @param index The index of an input tensor.
 * @param shape Shape that the given input tensor should be resized to. It should be an array of
 *     positive unsigned integer(s) containing the size of each dimension.
 * @param error An optional error parameter populated when there is an error in resizing the input
 *     tensor.
 *
 * @return Whether the input tensor was resized successfully. Returns NO if an error occurred.
 */
- (BOOL)resizeInputTensorAtIndex:(NSUInteger)index
                         toShape:(NSArray<NSNumber *> *)shape
                           error:(NSError **)error;

/**
 * Copies the given data into the input tensor at the given index. This is allowed only before the
 * interpreter is invoked.
 *
 * @param data The data to set. The byte size of the data must match what's required by the given
 *     input tensor.
 * @param index The index of an input tensor.
 * @param error An optional error parameter populated when there is an error in setting the data.
 *
 * @return Whether the data was set into the input tensor successfully. Returns NO if an error
 *     occurred.
 */
- (BOOL)copyData:(NSData *)data toInputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Gets the data from the output tensor at the given index. The interpreter invocation has to
 * complete before the data can be retrieved from an output tensor.
 *
 * @param index The index of an output tensor.
 * @param error An optional error parameter populated when there is an error in getting the data.
 *
 * @return The data of the output tensor at the given index. `nil` if there is an error.
 */
- (nullable NSData *)dataFromOutputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Allocates memory for tensors.
 *
 * @param error An optional error parameter populated when there is an error in allocating memory.
 *
 * @return Whether memory allocation is successful. Returns NO if an error occurred.
 */
- (BOOL)allocateTensorsWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
