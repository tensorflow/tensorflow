/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TfLiteExternalDelegateOptions is a structure of key/value options to create
// an external delegate.
#define kExternalDelegateMaxOptions 256
typedef struct TfLiteExternalDelegateOptions {
  const char* lib_path;
  int count;
  const char* keys[kExternalDelegateMaxOptions];
  const char* values[kExternalDelegateMaxOptions];
  TfLiteStatus (*insert)(struct TfLiteExternalDelegateOptions* options,
                         const char* key, const char* value);
} TfLiteExternalDelegateOptions;

// Insert key/value to the options.
TfLiteStatus TfLiteExternalDelegateOptionsInsert(
    TfLiteExternalDelegateOptions* options, const char* key, const char* value);

// Populates TfLiteExternalDelegateOptions with the given shared library path.
TfLiteExternalDelegateOptions TfLiteExternalDelegateOptionsDefault(
    const char* lib_path);

// Creates a new delegate instance that need to be destroyed with
// `TfLiteExternalDelegateDelete` when delegate is no longer used by TFLite.
TfLiteDelegate* TfLiteExternalDelegateCreate(
    const TfLiteExternalDelegateOptions* options);

// Destroys a delegate created with `TfLiteExternalDelegateCreate` call.
void TfLiteExternalDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_H_
