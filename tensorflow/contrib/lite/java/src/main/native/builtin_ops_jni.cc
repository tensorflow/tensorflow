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

#include "tensorflow/contrib/lite/kernels/register.h"

namespace tflite {

// The JNI code in interpreter_jni.cc expects a CreateOpResolver() function in
// the tflite namespace. This one instantiates a BuiltinOpResolver, with all the
// builtin ops. For smaller binary sizes users should avoid linking this in, and
// should provide a custom make CreateOpResolver() instead.
std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
      new tflite::ops::builtin::BuiltinOpResolver());
}

}  // namespace tflite
