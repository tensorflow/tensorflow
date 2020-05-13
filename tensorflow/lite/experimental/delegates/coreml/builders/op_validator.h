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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_

#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace delegates {
namespace coreml {
// Follow the ordering of TfLiteBuiltinOperator enum.
bool IsConcatenationOpSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node, TfLiteContext* context);
bool IsConvolutionOpSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context);
bool IsDepthwiseConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);
bool IsFullyConnectedOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context);
bool IsReshapeOpSupported(const TfLiteRegistration* registration,
                          const TfLiteNode* node, TfLiteContext* context,
                          int coreml_version);
bool IsResizeBilinearOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context);
bool IsTransposeConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);
}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_
