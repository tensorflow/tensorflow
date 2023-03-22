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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/create_op_resolver.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_CREATE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_CORE_CREATE_OP_RESOLVER_H_

#include <memory>

#include "tensorflow/lite/mutable_op_resolver.h"
// The following include is not needed but is kept for now to not break
// compatibility for existing clients; it should be removed with the next
// non-backwards compatible version of TFLite.
#include "tensorflow/lite/op_resolver.h"

namespace tflite {
std::unique_ptr<MutableOpResolver> CreateOpResolver();
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_CREATE_OP_RESOLVER_H_
