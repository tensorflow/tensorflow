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
#ifndef TENSORFLOW_LITE_KERNELS_REGISTER_H_
#define TENSORFLOW_LITE_KERNELS_REGISTER_H_

#include "tensorflow/lite/model.h"  // Legacy.
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace builtin {

// This built-in op resolver provides a list of TfLite delegates that could be
// applied by TfLite interpreter by default.
class BuiltinOpResolver : public MutableOpResolver {
 public:
  // NOTE: we *deliberately* don't define any virtual functions here to avoid
  // behavior changes when users pass a derived instance by value or assign a
  // derived instance to a variable of this class. See "object slicing"
  // (https://en.wikipedia.org/wiki/Object_slicing)) for details.
  BuiltinOpResolver();
};

// TfLite interpreter could apply a TfLite delegate by default. To completely
// disable this behavior, one could choose to use the following class
// BuiltinOpResolverWithoutDefaultDelegates.
class BuiltinOpResolverWithoutDefaultDelegates : public BuiltinOpResolver {
 public:
  BuiltinOpResolverWithoutDefaultDelegates() : BuiltinOpResolver() {
    delegate_creators_.clear();
    opaque_delegate_creators_.clear();
  }
};

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_REGISTER_H_
