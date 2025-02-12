/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_REGISTER_H_
#define TENSORFLOW_LITE_KERNELS_REGISTER_H_

/// For documentation, see third_party/tensorflow/lite/core/kernels/register.h

#include "tensorflow/lite/core/kernels/register.h"  // IWYU pragma: export

namespace tflite {
namespace ops {
namespace builtin {
using BuiltinOpResolver = ::tflite::ops::builtin::BuiltinOpResolver;
using BuiltinOpResolverWithoutDefaultDelegates =
    ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates;

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_REGISTER_H_
