/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/array.h"

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

char Array::ID = 0;

std::vector<Array*> MakeArrayPointerList(
    absl::Span<const tsl::RCReference<Array>> arrays) {
  std::vector<Array*> result;
  result.reserve(arrays.size());
  for (const auto& array : arrays) {
    result.push_back(array.get());
  }
  return result;
}

}  // namespace ifrt
}  // namespace xla
