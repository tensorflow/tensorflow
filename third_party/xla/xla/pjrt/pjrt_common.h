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

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// bool comes before int64_t because when pybind11 tries to convert a Python
// object to a C++ type, it will try to convert it to the first type in the list
// of possible types that it can be converted to (b/309163973).
using PjRtValueType =
    std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;

// The strong-typed integer classes to better disambiguate different IDs for
// PJRT devices.
TSL_LIB_GTL_DEFINE_INT_TYPE(PjRtGlobalDeviceId, int32_t);
TSL_LIB_GTL_DEFINE_INT_TYPE(PjRtLocalDeviceId, int32_t);
TSL_LIB_GTL_DEFINE_INT_TYPE(PjRtLocalHardwareId, int32_t);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMMON_H_
