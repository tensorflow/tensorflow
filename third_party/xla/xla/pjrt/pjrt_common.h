/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_COMMON_H_
#define XLA_PJRT_PJRT_COMMON_H_

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"

namespace xla {

// bool comes before int64_t because when pybind11 tries to convert a Python
// object to a C++ type, it will try to convert it to the first type in the list
// of possible types that it can be converted to (b/309163973).
using PjRtValueType =
    std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;

xla::PjRtValueTypeProto PjRtValueTypeToProto(const PjRtValueType& value);

PjRtValueType PjRtValueTypeFromProto(const xla::PjRtValueTypeProto& value);

template <typename Id>
using PjRtIdContainer = absl::InlinedVector<Id, 4>;

template <typename Id>
PjRtIdContainer<Id> MakeContinuousIds(int start, int size) {
  PjRtIdContainer<Id> container;
  container.reserve(size);
  for (int i = 0; i < size; ++i) {
    container.push_back(Id(start + i));
  }
  return container;
}

using PjRtPlatformId = uint64_t;

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMMON_H_
