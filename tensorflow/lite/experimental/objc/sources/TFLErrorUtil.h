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

#import "tensorflow/lite/experimental/objc/apis/TFLInterpreter.h"

NS_ASSUME_NONNULL_BEGIN

/** Helper utility for error reporting. */
@interface TFLErrorUtil : NSObject

/**
 * Creates and saves an interpreter error with the given error code and description.
 *
 * @param code Error code.
 * @param description Error description.
 * @param error Pointer to where to save the created error. If `nil`, no error will be saved.
 */
+ (void)saveInterpreterErrorWithCode:(TFLInterpreterErrorCode)code
                         description:(NSString *)description
                               error:(NSError **)error;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
