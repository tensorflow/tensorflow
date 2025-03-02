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

#ifndef XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_BUFFER_H_
#define XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_BUFFER_H_

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"

namespace mlir::stablehlo {

std::unique_ptr<xla::PjRtBuffer> CreateMlirBufferFromLiteral(
    const xla::LiteralSlice& literal, xla::PjRtMemorySpace* memory_space);
std::unique_ptr<xla::PjRtBuffer> CreateMlirBufferFromAttribute(
    DenseElementsAttr attribute, xla::PjRtMemorySpace* memory_space);
std::unique_ptr<xla::PjRtBuffer> CreateMlirBufferUninitizlied(
    const xla::Shape& shape, xla::PjRtMemorySpace* memory_space);
absl::StatusOr<DenseElementsAttr> GetAttributeFromBuffer(
    xla::PjRtBuffer* buffer);
DenseElementsAttr CloneIntoContext(DenseElementsAttr attr,
                                   MLIRContext& context);

}  // namespace mlir::stablehlo

#endif  // XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_BUFFER_H_
