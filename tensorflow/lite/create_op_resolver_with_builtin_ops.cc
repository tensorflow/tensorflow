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

#include <memory>

#include "tensorflow/lite/core/create_op_resolver.h"
#include "tensorflow/lite/core/kernels/register.h"

namespace tflite {

// This function instantiates a  BuiltinOpResolverWithoutDefaultDelegates, with
// all the builtin ops but without applying any TfLite delegates by default
// (like the XNNPACK delegate). For smaller binary sizes users should avoid
// linking this in, and should provide a CreateOpResolver() with selected ops
// instead.
std::unique_ptr<MutableOpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
      new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
}

}  // namespace tflite
