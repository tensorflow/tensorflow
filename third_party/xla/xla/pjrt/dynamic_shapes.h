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

#ifndef XLA_PJRT_DYNAMIC_SHAPES_H_
#define XLA_PJRT_DYNAMIC_SHAPES_H_

#include <cstddef>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

enum class PjRtDynamicShapeKind {
  kNotSupported,
  kPrefix,  // Prepended before the payload.
  kSuffix,  // Appended after the payload.
};

// Offsets and bounds for extracting dynamic shape metadata.
struct PjRtShapeAndMetadataTransferRequirements {
  static PjRtShapeAndMetadataTransferRequirements Get(
      const xla::Shape& shape, PjRtDynamicShapeKind kind);

  size_t size = 0;
  size_t metadata_alignment = 0;
  size_t metadata_offset = 0;
  size_t metadata_size = 0;
  size_t array_offset = 0;
  size_t array_size = 0;
};

// Reads dynamic shape metadata into a static Shape.
absl::StatusOr<xla::Shape> ReadDynamicShapeMetadata(
    absl::Span<const uint8_t> metadata, const xla::Shape& shape,
    PjRtDynamicShapeKind kind);

// Returns a view sliced to exclude dynamic shape metadata.
absl::StatusOr<PjRtRawBufferRef> RemoveDynamicShapeMetadataIfPresent(
    PjRtRawBufferRef raw_buffer, const xla::Shape& device_shape,
    const xla::Shape& logical_shape, PjRtDynamicShapeKind kind);

// Reads dynamic shape metadata into an output AsyncValueRef.
void ReadDynamicShape(PjRtRawBufferRef raw_buffer,
                      tsl::AsyncValueRef<xla::Shape> output_shape,
                      xla::Shape shape, PjRtDynamicShapeKind kind);

}  // namespace xla

#endif  // XLA_PJRT_DYNAMIC_SHAPES_H_
