/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_VERIFIER_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_VERIFIER_H_

#include <stdio.h>

namespace tflite {

// Verifies the integrity of a Tensorflow Lite flatbuffer model file.
// Currently, it verifies:
// * The file is following a legit flatbuffer schema.
// * The model is in supported version.
bool Verify(const void* buf, size_t len);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_VERIFIER_H_
