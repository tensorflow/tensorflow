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

#ifndef XLA_CORE_COLLECTIVES_RANK_ID_H_
#define XLA_CORE_COLLECTIVES_RANK_ID_H_

#include <cstdint>

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Strongly-typed integer type for defining the rank of the process in the
// collective clique.
TSL_LIB_GTL_DEFINE_INT_TYPE(RankId, int64_t);

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_RANK_ID_H_
