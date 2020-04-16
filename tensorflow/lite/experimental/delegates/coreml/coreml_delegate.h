/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
typedef enum {
  // Create Core ML delegate only on devices with Apple Neural Engine.
  // Returns nullptr otherwise.
  TfLiteCoreMlDelegateDevicesWithNeuralEngine,
  // Always create Core ML delegate
  TfLiteCoreMlDelegateAllDevices
} TfLiteCoreMlDelegateEnabledDevices;

typedef struct {
  // Only create delegate when Neural Engine is available on the device.
  TfLiteCoreMlDelegateEnabledDevices enabled_devices;
} TfLiteCoreMlDelegateOptions;

// Return a delegate that uses CoreML for ops execution.
// Must outlive the interpreter.
TfLiteDelegate* TfLiteCoreMlDelegateCreate(
    const TfLiteCoreMlDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_H_
