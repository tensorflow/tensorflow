/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_ATTRIBUTE_MAP_UTIL_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_ATTRIBUTE_MAP_UTIL_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/attribute_map.h"

namespace xla {
namespace ifrt {

// Converts a PjRt attribute map into an IFRT attribute map.
AttributeMap FromPjRtAttributeMap(
    absl::flat_hash_map<std::string, xla::PjRtValueType> attributes);

// Converts an IFRT attribute map into a PjRt attribute map.
absl::flat_hash_map<std::string, xla::PjRtValueType> ToPjRtAttributeMap(
    AttributeMap attributes);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_ATTRIBUTE_MAP_UTIL_H_
