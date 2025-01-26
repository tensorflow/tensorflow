// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_PLUGIN_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_PLUGIN_COMMON_H_

#include <string>
#include <vector>

// Simple compiled result def holds byte code and per op data.
struct LiteRtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

namespace litert::example {}  // namespace litert::example

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_PLUGIN_COMMON_H_
