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

#include "xla/python/ifrt/layout.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char Layout::ID = 0;
char CompactLayout::ID = 0;

absl::StatusOr<CustomLayoutRef> Layout::FromProto(
    const LayoutProto& layout_proto) {
  return Deserialize<Layout>(layout_proto.serialized_layout(),
                             /*options=*/nullptr);
}

absl::StatusOr<LayoutProto> Layout::ToProto() const {
  LayoutProto layout_proto;
  TF_ASSIGN_OR_RETURN(*layout_proto.mutable_serialized_layout(),
                      Serialize(*this, /*options=*/nullptr));
  return layout_proto;
}

absl::StatusOr<absl_nonnull std::unique_ptr<CompactLayout>>
CompactLayout::Create(absl::Span<const int> major_to_minor) {
  tsl::core::Bitmap bitmap(major_to_minor.size());
  for (int i : major_to_minor) {
    if (i < 0 || i >= major_to_minor.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("CompactLayout expects major_to_minor with elements in "
                       "range [0, ",
                       major_to_minor.size(), "), but got major_to_minor=[",
                       absl::StrJoin(major_to_minor, ","), "]"));
    }
    bitmap.set(i);
  }
  if (!bitmap.IsAllSet()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CompactLayout expects major_to_minor with all elements "
                     "in range [0, ",
                     major_to_minor.size(), "), but got major_to_minor=[",
                     absl::StrJoin(major_to_minor, ","), "]"));
  }
  return absl::WrapUnique<CompactLayout>(new CompactLayout(
      MajorToMinor(major_to_minor.begin(), major_to_minor.end())));
}

absl_nonnull std::unique_ptr<CompactLayout> CompactLayout::CreateCOrder(
    int num_shard_shape_dims) {
  MajorToMinor major_to_minor(num_shard_shape_dims);
  absl::c_iota(major_to_minor, 0);
  return absl::WrapUnique<CompactLayout>(
      new CompactLayout(std::move(major_to_minor)));
}

absl::StatusOr<std::optional<int64_t>> CompactLayout::ByteSize(
    DType dtype, const Shape& shard_shape) const {
  if (major_to_minor_.size() != shard_shape.dims().size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CompactLayout expects Shape with the same number of "
                     "dimensions as major_to_minor [",
                     absl::StrJoin(major_to_minor_, ","),
                     "], but got shard_shape=", shard_shape));
  }
  auto bit_size = dtype.bit_size();
  if (!bit_size.has_value()) {
    return std::nullopt;
  }
  // All elements are packed at the bit level. The last byte may contain a small
  // padding.
  return (shard_shape.num_elements() * *bit_size + 7) / 8;
}

bool CompactLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  if (const auto* other_compact = llvm::dyn_cast<CompactLayout>(&other);
      other_compact != nullptr) {
    return major_to_minor_ == other_compact->major_to_minor_;
  }
  return false;
}

std::string CompactLayout::ToString() const {
  return absl::StrCat("CompactLayout(major_to_minor=[",
                      absl::StrJoin(major_to_minor_, ","), "])");
}

absl::StatusOr<bool> EquivalentLayouts(DType dtype1, const Shape& shape1,
                                       const ShardingRef& sharding1,
                                       const LayoutRef& layout1, DType dtype2,
                                       const Shape& shape2,
                                       const ShardingRef& sharding2,
                                       const LayoutRef& layout2) {
  if (layout1 == nullptr && layout2 == nullptr) {
    // TODO(hyeontaek): Track a default layout domain in `Device` to check if
    // two default layouts will be the same. For now, we resolve them to
    // concrete layouts and compare them.
    Device* device1 = sharding1->devices()->devices().front();
    Device* device2 = sharding2->devices()->devices().front();
    if (dtype1 == dtype2 && shape1 == shape2 && device1 == device2 &&
        sharding1->memory_kind() == sharding2->memory_kind()) {
      // Assume that layouts will resolve to the same concrete layout if all
      // metadata is the same.
      return true;
    }
    // TODO(hyeontaek): Change to IFRT `Layout` comparison once
    // `Client::GetDefaultLayout()` returns a `CustomLayoutRef`.
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const xla::PjRtLayout> pjrt_layout1,
        device1->client()->GetDefaultLayout(dtype1, shape1.dims(), device1,
                                            sharding1->memory_kind()));
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const xla::PjRtLayout> pjrt_layout2,
        device2->client()->GetDefaultLayout(dtype2, shape2.dims(), device2,
                                            sharding2->memory_kind()));
    return *pjrt_layout1 == *pjrt_layout2;
  }
  if (layout1 != nullptr && layout2 != nullptr) {
    return *layout1 == *layout2;
  }
  return false;
}

}  // namespace ifrt
}  // namespace xla
