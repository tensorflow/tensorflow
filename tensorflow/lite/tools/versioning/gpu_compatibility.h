/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_GPU_COMPATIBILITY_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_GPU_COMPATIBILITY_H_

#include "absl/status/status.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

// Bitwise flags to specify GPU compatibility. Multiple flags can be set.
enum class GpuCompatibilityFlags {
  kStandard = 0,
  kEnhancedBroadcast = 1 << 0,  // Set when backend supports enhanced broadcast.
};

// Check if the given op signature is compatible with GPU delegate.
absl::Status CheckGpuDelegateCompatibility(
    const OpSignature& op_sig,
    GpuCompatibilityFlags flags = GpuCompatibilityFlags::kStandard);

// Check if the given operator in a TFLite flatbuffer model is compatible with
// GPU delegate.
// WARNING: It's not fully implemented and still under development. Only use the
// function for an experimental feature.
// WARNING: This is an experimental API and subject to change.
absl::Status CheckGpuDelegateCompatibility(const OperatorCode* op_code,
                                           const Operator* op,
                                           const SubGraph* subgraph,
                                           const Model* model);

// Check if the given TfLiteNode op is compatible with GPU delegate.
absl::Status CheckGpuDelegateCompatibility(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration,
    GpuCompatibilityFlags flags = GpuCompatibilityFlags::kStandard);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_GPU_COMPATIBILITY_H_
