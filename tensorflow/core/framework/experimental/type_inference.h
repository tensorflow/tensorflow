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
#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPE_INFERENCE_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPE_INFERENCE_H_

#include <unordered_map>

#include "tensorflow/core/framework/full_type.pb.h"

namespace tensorflow {

namespace full_type {

// A raw integer type for testing.
using Type = size_t;
// A short name for numeric types i.e. tf.float32 --> f4.
std::string ShortName(Type type);
// A long type name.
std::string Name(Type type);
// For testing. This allows passing in extra types that don't exist in FT.
Type ReturnType(Type t1, Type t2);
// Check what type `t1` and `t2` are promotable to, and return it.
FullTypeDef ReturnType(FullTypeDef t1, FullTypeDef t2);

// TODO(aselle): These shouldn't be necessary in the long run.
enum EXTRA_TYPES {
  TFT_BOOL_WEAK = 16000,
  TFT_FLOAT_WEAK = 16001,
  TFT_INT_WEAK = 16002,
  TFT_COMPLEX_WEAK = 16003,
};

}  // namespace full_type
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPE_INFERENCE_H_
