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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_

#include <cstdint>
#include <vector>

#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

// Maps a set of input variables to a set of output SymbolicExpr trees.
class SymbolicMap {
 public:
  SymbolicMap(int64_t num_dimensions, int64_t num_symbols,
              std::vector<SymbolicExpr> exprs);

  int64_t GetNumDims() const { return num_dimensions_; }
  int64_t GetNumSymbols() const { return num_symbols_; }
  int64_t GetNumResults() const { return exprs_.size(); }
  const std::vector<SymbolicExpr>& GetResults() const { return exprs_; }
  SymbolicExpr GetResult(unsigned idx) const { return exprs_[idx]; }

 private:
  int64_t num_dimensions_;
  int64_t num_symbols_;
  std::vector<SymbolicExpr> exprs_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_
