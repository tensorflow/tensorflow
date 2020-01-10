/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace flex {

// Return the registration object used to initialize and execute ops that will
// be delegated to TensorFlow's Eager runtime. This TF Lite op is created by
// the flex delegate to handle execution of a supported subgraph. The usual
// flow is that the delegate informs the interpreter of supported nodes in a
// graph, and each supported subgraph is replaced with one instance of this
// kernel.
TfLiteRegistration GetKernel();

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_
