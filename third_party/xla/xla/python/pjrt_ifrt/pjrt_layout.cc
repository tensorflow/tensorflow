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

#include "xla/python/pjrt_ifrt/pjrt_layout.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char PjRtLayout::ID = 0;

namespace {

// Computes the byte size of a shard shape using a concrete `PjRtLayout`.
absl::StatusOr<std::optional<int64_t>> ComputeByteSize(
    DType dtype, const Shape& shard_shape,
    const absl_nonnull std::shared_ptr<const xla::PjRtLayout>& pjrt_layout) {
  auto bit_size = dtype.bit_size();
  if (!bit_size.has_value()) {
    return std::nullopt;
  }
  TF_ASSIGN_OR_RETURN(auto xla_primitive_type, ToPrimitiveType(dtype));
  auto xla_shape =
      xla::ShapeUtil::MakeShape(xla_primitive_type, shard_shape.dims());
  *xla_shape.mutable_layout() = pjrt_layout->xla_layout();
  return xla::ShapeUtil::ArraySize(xla_shape);
}

}  // namespace

absl_nonnull std::unique_ptr<PjRtLayout> PjRtLayout::Create(
    absl_nonnull std::shared_ptr<const xla::PjRtLayout> pjrt_layout) {
  return absl::WrapUnique<PjRtLayout>(new PjRtLayout(std::move(pjrt_layout)));
}

absl::StatusOr<std::optional<int64_t>> PjRtLayout::ByteSize(
    DType dtype, const Shape& shard_shape) const {
  return ComputeByteSize(dtype, shard_shape, pjrt_layout_);
}

absl::StatusOr<std::optional<int64_t>> PjRtLayout::ByteSize(
    DType dtype, const Shape& shape, const ShardingRef& sharding,
    const absl_nullable std::shared_ptr<const xla::PjRtLayout>& pjrt_layout) {
  std::optional<absl::StatusOr<Shape>> shard_shape;
  const Shape* shard_shape_ptr;
  if (sharding->IsFullyReplicated()) {
    shard_shape_ptr = &shape;
  } else {
    shard_shape = sharding->GetShardShape(shape);
    if (!shard_shape->ok()) {
      return {std::nullopt};
    }
    shard_shape_ptr = &**shard_shape;
  }
  if (pjrt_layout == nullptr) {
    Device* device = sharding->devices()->devices().front();
    TF_ASSIGN_OR_RETURN(
        const std::shared_ptr<const xla::PjRtLayout>
            concrete_default_pjrt_layout,
        device->client()->GetDefaultPjRtLayout(
            dtype, shard_shape_ptr->dims(), device, sharding->memory_kind()));
    return ComputeByteSize(dtype, *shard_shape_ptr,
                           concrete_default_pjrt_layout);
  }
  return ComputeByteSize(dtype, *shard_shape_ptr, pjrt_layout);
}

bool PjRtLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  if (const auto* other_pjrt = llvm::dyn_cast<PjRtLayout>(&other);
      other_pjrt != nullptr) {
    return *pjrt_layout_ == *other_pjrt->pjrt_layout_;
  }
  return false;
}

std::string PjRtLayout::ToString() const {
  return absl::StrCat("PjRtLayout(", pjrt_layout_->ToString(), ")");
}

absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const CustomLayoutRef& layout) {
  if (const auto* pjrt_layout = llvm::dyn_cast<PjRtLayout>(layout.get())) {
    return pjrt_layout->pjrt_layout();
  }
  if (const auto* compact_layout =
          llvm::dyn_cast<CompactLayout>(layout.get())) {
    xla::Layout layout;
    int num_dims = compact_layout->major_to_minor().size();
    layout.mutable_minor_to_major()->reserve(num_dims);
    for (int i = num_dims - 1; i >= 0; i--) {
      layout.add_minor_to_major(compact_layout->major_to_minor()[i]);
    }
    return std::make_shared<xla::PjRtLayout>(std::move(layout));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported layout type: ", *layout));
}

absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape, Device* device,
             MemoryKind memory_kind, const LayoutRef& layout) {
  if (layout == nullptr) {
    return device->client()->GetDefaultPjRtLayout(dtype, shard_shape.dims(),
                                                  device, memory_kind);
  }
  return ToPjRtLayout(dtype, shard_shape, layout);
}

}  // namespace ifrt
}  // namespace xla
