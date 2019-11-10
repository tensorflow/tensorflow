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

#import "tensorflow/lite/experimental/objc/apis/TFLInterpreter.h"

@class TFLTensor;

NS_ASSUME_NONNULL_BEGIN

@interface TFLInterpreter (Internal)

/**
 * Copies the given data into the input tensor at the given index. This is allowed only before the
 * interpreter is invoked.
 *
 * @param data The data to set. The byte size of the data must match what's required by the input
 *     tensor at the given index.
 * @param index An input tensor index.
 * @param error An optional error parameter populated when there is an error in setting the data.
 *
 * @return Whether the data was copied into the input tensor at the given index successfully.
 *     Returns NO if an error occurred.
 */
- (BOOL)copyData:(NSData *)data toInputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Retrieves a copy of the data from the given tensor. For an output tensor, the interpreter
 * invocation has to complete before the data can be retrieved.
 *
 * @param tensor A tensor.
 * @param error An optional error parameter populated when there is an error in getting the data.
 *
 * @return The data of the given tensor. `nil` if there is an error or data is not available.
 */
- (nullable NSData *)dataFromTensor:(TFLTensor *)tensor error:(NSError **)error;

/**
 * Retrieves the shape of the given tensor, an array of positive unsigned integer(s) containing the
 * size of each dimension. For example: shape of [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] is
 * [2, 2, 3].
 *
 * @param tensor An input or output tensor.
 * @param error An optional error parameter populated when there is an error in retrieving the
 *     shape.
 *
 * @return The shape of the tensor. `nil` if there is an error in retrieving the shape.
 */
- (nullable NSArray<NSNumber *> *)shapeOfTensor:(TFLTensor *)tensor error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
