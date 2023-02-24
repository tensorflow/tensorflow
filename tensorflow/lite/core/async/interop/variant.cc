/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/interop/variant.h"

#include <utility>

namespace tflite {
namespace interop {

Variant::Variant() {
  type = kInvalid;
  val.i = 0;
}

bool Variant::operator==(const Variant& other) const {
  if (type != other.type) return false;
  switch (type) {
    case kInvalid:
      // Treats uninitialized variant equals.
      return true;
    case kInt:
      return val.i == other.val.i;
    case kSizeT:
      return val.s == other.val.s;
    case kString:
      return (val.c == other.val.c) || (strcmp(val.c, other.val.c) == 0);
  }
}

bool Variant::operator!=(const Variant& other) const {
  return !(*this == other);
}

}  // namespace interop
}  // namespace tflite
