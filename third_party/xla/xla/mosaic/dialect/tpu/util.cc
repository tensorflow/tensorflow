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

#include "xla/mosaic/dialect/tpu/util.h"

#include <optional>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space,
                    std::optional<CoreType> type) {
  auto memory_space =
      dyn_cast_or_null<tpu::MemorySpaceAttr>(ty.getMemorySpace());
  if (!memory_space || memory_space.getValue() != space) {
    return false;
  }
  return !type.has_value() || memory_space.getCoreType() == type;
}

}  // namespace mlir::tpu
