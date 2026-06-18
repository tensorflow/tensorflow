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

#import "tensorflow/lite/objc/apis/TFLSignatureRunner.h"

#import "TFLTensorDataAccessor.h"

@class TFLInterpreter;

NS_ASSUME_NONNULL_BEGIN

@interface TFLSignatureRunner (Internal) <TFLTensorDataAccessor>

/**
 * Initializes a new TensorFlow Lite signature runner instance with the given interpreter and
 * signature key.
 *
 * @param interpreter The TensorFlow Lite model interpreter.
 * @param signatureKey The signature key.
 * @param error An optional error parameter populated when there is an error in initializing the
 * signature runner.
 *
 * @return A new instance of `TFLSignatureRunner` with the given model and options. `nil` if there
 * is an error in initializing the signature runner.
 */
- (nullable instancetype)initWithInterpreter:(TFLInterpreter *)interpreter
                                signatureKey:(NSString *)signatureKey
                                       error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
