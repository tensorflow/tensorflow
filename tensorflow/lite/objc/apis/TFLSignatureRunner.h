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

NS_ASSUME_NONNULL_BEGIN

@class TFLTensor;

/** Domain for errors in the signature runner API. */
FOUNDATION_EXPORT NSErrorDomain const TFLSignatureRunnerErrorDomain;

/**
 * @enum TFLSignatureRunnerErrorCode
 * This enum specifies various error codes related to `TFLSignatureRunner`.
 */
typedef NS_ENUM(NSUInteger, TFLSignatureRunnerErrorCode) {
  /** Input data has invalid byte size. */
  TFLSignatureRunnerErrorCodeInvalidInputByteSize,

  /** Provided shape is invalid. It must be a non-empty array of positive unsigned integers. */
  TFLSignatureRunnerErrorCodeInvalidShape,

  /** Failed to create `TFLSignatureRunner`. */
  TFLSignatureRunnerErrorCodeFailedToCreateSignatureRunner,

  /** Failed to invoke `TFLSignatureRunner`. */
  TFLSignatureRunnerErrorCodeFailedToInvoke,

  /** Failed to retrieve a tensor. */
  TFLSignatureRunnerErrorCodeFailedToGetTensor,

  /** Invalid tensor. */
  TFLSignatureRunnerErrorCodeInvalidTensor,

  /** Failed to resize an input tensor. */
  TFLSignatureRunnerErrorCodeFailedToResizeInputTensor,

  /** Failed to copy data into an input tensor. */
  TFLSignatureRunnerErrorCodeFailedToCopyDataToInputTensor,

  /** Copying data into an output tensor not allowed. */
  TFLSignatureRunnerErrorCodeCopyDataToOutputTensorNotAllowed,

  /** Failed to get data from a tensor. */
  TFLSignatureRunnerErrorCodeFailedToGetDataFromTensor,

  /** Failed to allocate memory for tensors. */
  TFLSignatureRunnerErrorCodeFailedToAllocateTensors,
};

/**
 * A TensorFlow Lite model signature runner. You can get a `TFLSignatureRunner` instance for a
 * signature from the `TFLInterpreter` and then use the SignatureRunner APIs.
 *
 * @note `TFLSignatureRunner` instances are *not* thread-safe.
 * @note Each `TFLSignatureRunner` instance is associated with a `TFLInterpreter` instance. As long
 *     as a `TFLSignatureRunner` instance is still in use, its associated `TFLInterpreter` instance
 *     will not be deallocated.
 */
@interface TFLSignatureRunner : NSObject

/** The signature key. */
@property(nonatomic, readonly) NSString *signatureKey;

/** An ordered list of the SignatureDefs input names. */
@property(nonatomic, readonly) NSArray<NSString *> *inputs;

/** An ordered list of the SignatureDefs output names. */
@property(nonatomic, readonly) NSArray<NSString *> *outputs;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/**
 * Returns the input tensor with the given input name in the signature.
 *
 * @param name The input name in the signature.
 * @param error An optional error parameter populated when there is an error in looking up the input
 *     tensor.
 *
 * @return The input tensor with the given input name. `nil` if there is an error. See the
 *     `TFLTensor` class documentation for more details on the life expectancy between the returned
 *     tensor and this signature runner.
 */
- (nullable TFLTensor *)inputTensorWithName:(NSString *)name error:(NSError **)error;

/**
 * Returns the output tensor with the given output name in the signature.
 *
 * @param name The output name in the signature.
 * @param error An optional error parameter populated when there is an error in looking up the
 *     output tensor.
 *
 * @return The output tensor with the given output name. `nil` if there is an error. See the
 *     `TFLTensor` class documentation for more details on the life expectancy between the returned
 *     tensor and this signature runner.
 */
- (nullable TFLTensor *)outputTensorWithName:(NSString *)name error:(NSError **)error;

/**
 * Resizes the input tensor with the given input name to the specified shape (an array of positive
 * unsigned integers).
 *
 * @param name The input name.
 * @param shape Shape that the given input tensor should be resized to. It should be an array of
 *     positive unsigned integer(s) containing the size of each dimension.
 * @param error An optional error parameter populated when there is an error in resizing the input
 *     tensor.
 *
 * @return Whether the input tensor was resized successfully. Returns NO if an error occurred.
 */
- (BOOL)resizeInputTensorWithName:(NSString *)name
                          toShape:(NSArray<NSNumber *> *)shape
                            error:(NSError **)error;

/**
 * Allocates memory for tensors.
 *
 * @note This call is *purely optional*. Tensor allocation will occur automatically during
 *     execution.
 *
 * @param error An optional error parameter populated when there is an error in allocating memory.
 *
 * @return Whether memory allocation is successful. Returns NO if an error occurred.
 */
- (BOOL)allocateTensorsWithError:(NSError **)error;

/**
 * Invoke the signature with given input data.
 *
 * @param inputs A map from input name to the input data. The input data will be copied into the
 *     input tensor.
 * @param error An optional error parameter populated when there is an error in invoking the
 * signature.
 *
 * @return Whether the invocation is successful. Returns NO if an error occurred.
 */
- (BOOL)invokeWithInputs:(NSDictionary<NSString *, NSData *> *)inputs Error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
