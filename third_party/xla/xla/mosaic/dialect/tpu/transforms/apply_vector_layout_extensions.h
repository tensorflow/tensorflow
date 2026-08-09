/* Copyright 2024 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_

#include <functional>

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu::extensions {

const llvm::StringMap<
    std::function<LogicalResult(ApplyVectorLayoutContext &, Operation &,
                                ArrayRef<Layout>, ArrayRef<Layout>)>> &
rules();

}  // namespace mlir::tpu::extensions

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
