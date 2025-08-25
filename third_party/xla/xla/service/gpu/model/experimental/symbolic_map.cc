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

#include "xla/service/gpu/model/experimental/symbolic_map.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

SymbolicMap::SymbolicMap(int64_t num_dimensions, int64_t num_symbols,
                         std::vector<SymbolicExpr> exprs)
    : num_dimensions_(num_dimensions),
      num_symbols_(num_symbols),
      exprs_(std::move(exprs)) {}

}  // namespace gpu
}  // namespace xla
