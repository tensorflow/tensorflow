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
#include "tensorflow/c/experimental/ops/gen/common/view_util.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

std::string Call(const std::string& object, const std::string& method,
                 std::vector<std::string> arguments, const char* oper) {
  return absl::Substitute("$0$1$2($3)", object, oper, method,
                          absl::StrJoin(arguments, ", "));
}

std::string Call(const std::string& function,
                 std::vector<std::string> arguments) {
  return absl::Substitute("$0($1)", function, absl::StrJoin(arguments, ", "));
}

std::string Quoted(const std::string& s) {
  return absl::Substitute("\"$0\"", s);
}

}  // namespace generator
}  // namespace tensorflow
