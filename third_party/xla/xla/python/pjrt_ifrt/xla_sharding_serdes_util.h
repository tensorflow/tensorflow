/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SERDES_UTIL_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SERDES_UTIL_H_

#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// Returns true if `op_sharding` or any nested tuple element carries a V3
// `named_sharding`.
inline bool ContainsNamedSharding(const xla::OpSharding& op_sharding) {
  if (op_sharding.has_named_sharding()) {
    return true;
  }
  for (const xla::OpSharding& element : op_sharding.tuple_shardings()) {
    if (ContainsNamedSharding(element)) {
      return true;
    }
  }
  return false;
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SERDES_UTIL_H_
