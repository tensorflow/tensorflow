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

#import "TFLErrorUtil.h"

NS_ASSUME_NONNULL_BEGIN

/** Error domain of TensorFlow Lite interpreter related errors. */
static NSString *const TFLInterpreterErrorDomain = @"org.tensorflow.lite.interpreter";

@implementation TFLErrorUtil

#pragma mark - Public

+ (void)saveInterpreterErrorWithCode:(TFLInterpreterErrorCode)code
                         description:(NSString *)description
                               error:(NSError **)error {
  [self setError:error withDomain:TFLInterpreterErrorDomain code:code description:description];
}

+ (void)setError:(NSError **)error
      withDomain:(NSErrorDomain)domain
            code:(NSInteger)code
     description:(NSString *)description {
  if (error) {
    *error = [NSError errorWithDomain:domain
                                 code:code
                             userInfo:@{NSLocalizedDescriptionKey : description}];
  }
}

@end

NS_ASSUME_NONNULL_END
