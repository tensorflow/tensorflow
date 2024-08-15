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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {

// Custom parser to parse IndexingMapAttr.
mlir::FailureOr<mlir::Attribute> ParseIndexingMapAttr(mlir::AsmParser& parser);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_
