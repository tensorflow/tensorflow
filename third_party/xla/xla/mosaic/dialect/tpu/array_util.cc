/* Copyright 2025 The JAX Authors.

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

#include "xla/mosaic/dialect/tpu/array_util.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tpu::internal {

bool sliceIsEmpty(const absl::Span<const int64_t> starts,
                  const absl::Span<const int64_t> limits) {
  for (auto [s, l] : llvm::zip_equal(starts, limits)) {
    CHECK_LE(s, l);
    if (s == l) {
      return true;
    }
  }
  return false;
}

bool incrementSliceIndex(const MutableArrayRef<int64_t> idx,
                         const absl::Span<const int64_t> starts,
                         const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, starts.size());
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = starts[i];
  }
  return false;
}

}  // namespace mlir::tpu::internal
