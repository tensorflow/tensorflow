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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_config.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

CppConfig::CppConfig(const string &category, const string &name_space)
    : category(category),
      unit(str_util::Lowercase(category)),
      namespaces(absl::StrSplit(name_space, "::")) {}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
