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
#include "tensorflow/cc/experimental/libtf/value.h"

#include <string>
#include <unordered_set>

namespace tf {
namespace libtf {
namespace impl {

const char* InternString(const char* s) {
  static auto* table = new std::unordered_set<std::string>;
  auto it = table->find(s);
  if (it != table->end()) {
    return it->c_str();
  }
  auto ret = table->insert(s);
  return ret.first->c_str();
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
