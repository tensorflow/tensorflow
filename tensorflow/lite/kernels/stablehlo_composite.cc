/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_composite {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode*) {
  TF_LITE_KERNEL_LOG(context,
                     "STABLEHLO_COMPOSITE must be inlined before execution");
  return kTfLiteError;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode*) {
  TF_LITE_KERNEL_LOG(context,
                     "STABLEHLO_COMPOSITE must be inlined before execution");
  return kTfLiteError;
}

}  // namespace stablehlo_composite

TfLiteRegistration* Register_STABLEHLO_COMPOSITE() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/stablehlo_composite::Prepare,
                                 /*invoke=*/stablehlo_composite::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
