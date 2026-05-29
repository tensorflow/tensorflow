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

#include "xla/pjrt/dynamic_shapes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/mem.h"

namespace xla {

PjRtShapeAndMetadataTransferRequirements
PjRtShapeAndMetadataTransferRequirements::Get(const xla::Shape& shape,
                                              PjRtDynamicShapeKind kind) {
  PjRtShapeAndMetadataTransferRequirements requirements;
  int64_t array_size = shape.IsToken() ? 0 : xla::ShapeUtil::ArraySize(shape);
  requirements.size = array_size;
  requirements.metadata_alignment = 1;
  requirements.metadata_offset = 0;
  requirements.metadata_size = 0;
  requirements.array_offset = 0;
  requirements.array_size = array_size;

  if (shape.IsToken() || shape.is_static()) {
    return requirements;
  }

  int64_t metadata_size = 0;

  switch (kind) {
    case PjRtDynamicShapeKind::kPrefix:
      if (shape.has_layout()) {
        metadata_size = shape.layout().dynamic_shape_metadata_prefix_bytes();
      }
      requirements.size = array_size + metadata_size;
      requirements.metadata_alignment = sizeof(int32_t);
      requirements.metadata_offset = 0;
      requirements.metadata_size = metadata_size;
      requirements.array_offset = metadata_size;
      break;

    case PjRtDynamicShapeKind::kSuffix:
      metadata_size = sizeof(int32_t) * shape.dimensions().size();
      requirements.size = array_size + metadata_size;
      requirements.metadata_alignment = sizeof(int32_t);
      requirements.metadata_offset = array_size;
      requirements.metadata_size = metadata_size;
      requirements.array_offset = 0;
      break;

    case PjRtDynamicShapeKind::kNotSupported:
    default:
      break;
  }
  return requirements;
}

absl::StatusOr<xla::Shape> ReadDynamicShapeMetadata(
    absl::Span<const uint8_t> metadata, const xla::Shape& shape,
    PjRtDynamicShapeKind kind) {
  if (kind == PjRtDynamicShapeKind::kNotSupported) {
    return absl::InvalidArgumentError("Dynamic shapes are not supported.");
  }

  if (metadata.size() < sizeof(int32_t) * shape.dimensions().size()) {
    return absl::InvalidArgumentError(
        "Metadata does not contain sufficient dimension bytes.");
  }

  const int32_t* runtime_sizes =
      reinterpret_cast<const int32_t*>(metadata.data());
  xla::Shape output_shape = shape;

  for (int i = 0; i < shape.dimensions().size(); ++i) {
    output_shape.set_dimensions(i, runtime_sizes[i]);
  }

  output_shape.clear_dynamic_dimensions();

  if (!xla::ShapeUtil::DynamicShapeIsCompatible(output_shape, shape)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Output dynamic shape (%s) incompatible with original shape (%s)",
        output_shape.ToString(true), shape.ToString(true)));
  }

  return output_shape;
}

absl::StatusOr<PjRtRawBufferRef> RemoveDynamicShapeMetadataIfPresent(
    PjRtRawBufferRef raw_buffer, const xla::Shape& device_shape,
    const xla::Shape& logical_shape, PjRtDynamicShapeKind kind) {
  auto device_requirements =
      PjRtShapeAndMetadataTransferRequirements::Get(device_shape, kind);
  if (device_requirements.metadata_size == 0) {
    return raw_buffer;
  }
  auto logical_requirements =
      PjRtShapeAndMetadataTransferRequirements::Get(logical_shape, kind);

  return raw_buffer->Slice(device_requirements.array_offset,
                           logical_requirements.array_size);
}

void ReadDynamicShape(PjRtRawBufferRef raw_buffer,
                      tsl::AsyncValueRef<xla::Shape> output_shape,
                      xla::Shape shape, PjRtDynamicShapeKind kind) {
  auto requirements =
      PjRtShapeAndMetadataTransferRequirements::Get(shape, kind);
  if (requirements.metadata_size == 0) {
    output_shape.SetError(
        absl::InvalidArgumentError("No dynamic metadata found."));
    return;
  }

  if (void* host_ptr = raw_buffer->GetHostPointer(); host_ptr != nullptr) {
    const uint8_t* metadata_base =
        static_cast<const uint8_t*>(host_ptr) + requirements.metadata_offset;
    auto result = ReadDynamicShapeMetadata(
        absl::MakeSpan(metadata_base, requirements.metadata_size), shape, kind);
    if (!result.ok()) {
      output_shape.SetError(result.status());
    } else {
      *output_shape = std::move(*result);
      output_shape.SetStateConcrete();
    }
    return;
  }

  void* scratch = tsl::port::AlignedMalloc(
      requirements.metadata_size,
      static_cast<std::align_val_t>(requirements.metadata_alignment));
  if (scratch == nullptr) {
    output_shape.SetError(absl::ResourceExhaustedError("AlignedMalloc failed"));
    return;
  }

  auto future = raw_buffer->CopyRawDeviceToHost(
      scratch, requirements.metadata_offset, requirements.metadata_size);

  future.OnReady(
      [scratch, output_shape, shape, kind,
       size = requirements.metadata_size](const absl::Status& status) mutable {
        if (!status.ok()) {
          output_shape.SetError(status);
        } else {
          auto result = ReadDynamicShapeMetadata(
              absl::MakeSpan(static_cast<const uint8_t*>(scratch), size), shape,
              kind);
          if (!result.ok()) {
            output_shape.SetError(result.status());
          } else {
            *output_shape = std::move(*result);
            output_shape.SetStateConcrete();
          }
        }
        tsl::port::AlignedFree(scratch);
      });
}

}  // namespace xla
