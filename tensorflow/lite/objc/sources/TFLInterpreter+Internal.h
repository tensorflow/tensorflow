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

#import "tensorflow/lite/objc/apis/TFLInterpreter.h"

#import "TFLTensorDataAccessor.h"

typedef struct TfLiteInterpreter TfLiteInterpreter;

NS_ASSUME_NONNULL_BEGIN

@interface TFLInterpreter (Internal) <TFLTensorDataAccessor>

/** TfLiteInterpreter backed by C API. */
@property(nonatomic, readonly) TfLiteInterpreter *interpreter;

@end

NS_ASSUME_NONNULL_END
