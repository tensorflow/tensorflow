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

  /** Invalid tensor. */
  TFLInterpreterErrorCodeInvalidTensor,

  /** Failed to resize an input tensor. */
  TFLInterpreterErrorCodeFailedToResizeInputTensor,

  /** Failed to copy data into an input tensor. */
  TFLInterpreterErrorCodeFailedToCopyDataToInputTensor,

  /** Copying data into an output tensor not allowed. */
  TFLInterpreterErrorCodeCopyDataToOutputTensorNotAllowed,

  /** Failed to get data from a tensor. */
  TFLInterpreterErrorCodeFailedToGetDataFromTensor,

  /** Failed to allocate memory for tensors. */
  TFLInterpreterErrorCodeFailedToAllocateTensors,

  /** Operation not allowed without allocating memory for tensors first. */
  TFLInterpreterErrorCodeAllocateTensorsRequired,

  /** Operation not allowed without invoking the interpreter first. */
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
 * @param error An optional error parameter populated when there is an error in initializing the
 *     interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and the default interpreter
 *     options. `nil` if there is an error in initializing the interpreter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path and
 * options.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param options Options to use for configuring the TensorFlow Lite interpreter.
 * @param error An optional error parameter populated when there is an error in initializing the
 *     interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and options. `nil` if there is an
 *     error in initializing the interpreter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

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
 * @return The input tensor at the given index. `nil` if there is an error. See the `TFLTensor`
 *     class documentation for more details on the life expectancy between the returned tensor and
 *     this interpreter.
 */
- (nullable TFLTensor *)inputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Returns the output tensor at the given index.
 *
 * @param index The index of an output tensor.
 * @param error An optional error parameter populated when there is an error in looking up the
 *     output tensor.
 *
 * @return The output tensor at the given index. `nil` if there is an error. See the `TFLTensor`
 *     class documentation for more details on the life expectancy between the returned tensor and
 *     this interpreter.
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
 * Allocates memory for tensors.
 *
 * @param error An optional error parameter populated when there is an error in allocating memory.
 *
 * @return Whether memory allocation is successful. Returns NO if an error occurred.
 */
- (BOOL)allocateTensorsWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
