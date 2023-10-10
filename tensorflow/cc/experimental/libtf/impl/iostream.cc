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
// Specializations of ostream::operator<< for API values. These are defined here
// so that they don't need to be linked in executables that need to be kept
// small (and don't use the functionality).
#include <iostream>

#include "tensorflow/cc/experimental/libtf/impl/none.h"
#include "tensorflow/cc/experimental/libtf/impl/string.h"
#include "tensorflow/cc/experimental/libtf/impl/tensor_spec.h"

namespace tf {
namespace libtf {
namespace impl {

std::ostream& operator<<(std::ostream& o, const None& none) {
  return o << "None";
}

std::ostream& operator<<(std::ostream& o, const String& str) {
  return o << str.str();
}

std::ostream& operator<<(std::ostream& o, const TensorSpec& x) {
  o << "TensorSpec(shape = " << x.shape.DebugString() << ", dtype = " << x.dtype
    << ")";
  return o;
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
