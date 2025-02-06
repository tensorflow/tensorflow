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
#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_

#include "tensorflow/compiler/mlir/lite/tools/versioning/op_signature.h"  // iwyu pragma: export
#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// Generate OpSignature with the given TfLiteContext, TfLiteNode and
// TfLiteRegistration.
// The function can be used by a compatibility checker of a delegate such as
// TFLiteOperationParser::IsSupported() in the GPU delegate.
OpSignature GetOpSignature(const TfLiteContext* context, const TfLiteNode* node,
                           const TfLiteRegistration* registration);
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_
