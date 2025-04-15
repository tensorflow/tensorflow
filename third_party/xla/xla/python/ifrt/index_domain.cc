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

#include "xla/python/ifrt/index_domain.h"

#include <ostream>
#include <string>

#include "absl/strings/str_cat.h"

namespace xla {
namespace ifrt {

std::string IndexDomain::DebugString() const {
  return absl::StrCat("IndexDomain(origin=", origin_.DebugString(),
                      ",shape=", shape_.DebugString(), ")");
}

std::ostream& operator<<(std::ostream& os, const IndexDomain& index_domain) {
  return os << index_domain.DebugString();
}

}  // namespace ifrt
}  // namespace xla
