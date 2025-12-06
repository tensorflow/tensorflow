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

#include "xla/python/pjrt_ifrt/pjrt_layout_migration_util.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

absl::StatusOr<CustomLayoutRef> GetDefaultLayoutUsingDefaultPjRtLayout(
    const Client* client, DType dtype, const Shape& shape,
    const ShardingRef& sharding) {
  if (sharding->devices()->empty()) {
    return absl::InvalidArgumentError(
        "Getting a default layout requires at least one device in sharding");
  }

  absl::StatusOr<Shape> shard_shape = sharding->GetShardShape(shape);
  if (shard_shape.ok()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const xla::PjRtLayout> pjrt_layout,
        client->GetDefaultPjRtLayout(dtype, shard_shape->dims(),
                                     sharding->devices()->devices().front(),
                                     sharding->memory_kind()));
    return PjRtLayout::Create(std::move(pjrt_layout));
  }
  // If the shard shape could not be determined, assume that the shard shape has
  // the same number of dimensions as the global shape.
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const xla::PjRtLayout> pjrt_layout,
      client->GetDefaultPjRtLayout(dtype, shape.dims(),
                                   sharding->devices()->devices().front(),
                                   sharding->memory_kind()));
  return PjRtLayout::Create(std::move(pjrt_layout));
}

CustomLayoutRef GetArrayLayoutUsingPjRtLayout(const Array* array) {
  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> pjrt_layout =
      array->pjrt_layout();
  if (pjrt_layout.ok()) {
    return PjRtLayout::Create(*std::move(pjrt_layout));
  }
  // Upon an error, fall back to the default layout.
  const ShardingRef& sharding = array->shared_ptr_sharding();
  absl::StatusOr<CustomLayoutRef> default_layout =
      array->client()->GetDefaultLayout(array->dtype(), array->shape(),
                                        sharding);
  if (default_layout.ok()) {
    return *std::move(default_layout);
  }

  // The fallback of using a descending layout below is expected to be taken
  // only with an incomplete implementation of `GetDefaultLayout()` (which is
  // not implemented using `GetDefaultLayoutUsingDefaultPjRtLayout()`). All
  // `Array` is expected to have some `Layout`, which implies that a call to
  // `GetDefaultLayout()` with array properties should be able to return some
  // `Layout` as well, only returning an error when there is a major runtime
  // error that makes it not meaningful to obtain a correct layout anyway (e.g.,
  // the IFRT Proxy client disconnected from the server.

  absl::StatusOr<Shape> shard_shape = sharding->GetShardShape(array->shape());
  if (shard_shape.ok()) {
    return PjRtLayout::Create(std::make_shared<const xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(shard_shape->dims().size())));
  }
  // If the shard shape could not be determined, assume that the shard shape
  // has the same number of dimensions as the global shape.
  return PjRtLayout::Create(std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(array->shape().dims().size())));
}

}  // namespace ifrt
}  // namespace xla
