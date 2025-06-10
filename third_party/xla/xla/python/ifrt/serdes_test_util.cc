/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/serdes_test_util.h"

#include <vector>

#include "xla/python/ifrt/serdes_any_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {
namespace test_util {

std::vector<SerDesVersion> AllSupportedSerDesVersions() {
  std::vector<SerDesVersion> versions;
  versions.reserve(SerDesVersion::current().version() -
                   SerDesAnyVersionAccessor::GetMinimum().version() + 1);
  for (int version = SerDesAnyVersionAccessor::GetMinimum().version();
       version <= SerDesVersion::current().version(); ++version) {
    versions.push_back(SerDesAnyVersionAccessor::Get(version));
  }
  return versions;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
