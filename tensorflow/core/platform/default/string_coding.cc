/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/default/string_coding.h"

namespace tensorflow {
namespace port {

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out) {
  return std::unique_ptr<StringListEncoder>(new StringListEncoder(out));
}

std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in) {
  return std::unique_ptr<StringListDecoder>(new StringListDecoder(in));
}

}  // namespace port
}  // namespace tensorflow
