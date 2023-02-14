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
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_COREML_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_COREML_DELEGATE_H_

#include "tensorflow/lite/core/c/common.h"

// LINT.IfChange

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
  // Specifies target Core ML version for model conversion.
  // Core ML 3 come with a lot more ops, but some ops (e.g. reshape) is not
  // delegated due to input rank constraint.
  // if not set to one of the valid versions, the delegate will use highest
  // version possible in the platform.
  // Valid versions: (2, 3)
  int coreml_version;
  // This sets the maximum number of Core ML delegates created.
  // Each graph corresponds to one delegated node subset in the
  // TFLite model. Set this to 0 to delegate all possible partitions.
  int max_delegated_partitions;
  // This sets the minimum number of nodes per partition delegated with
  // Core ML delegate. Defaults to 2.
  int min_nodes_per_partition;
#ifdef TFLITE_DEBUG_DELEGATE
  // This sets the index of the first node that could be delegated.
  int first_delegate_node_index;
  // This sets the index of the last node that could be delegated.
  int last_delegate_node_index;
#endif
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

// LINT.ThenChange(README.md)

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_COREML_DELEGATE_H_
