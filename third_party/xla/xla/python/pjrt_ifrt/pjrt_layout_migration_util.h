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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_MIGRATION_UTIL_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_MIGRATION_UTIL_H_

#include "absl/status/statusor.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

// Helper function for implementing `Client::GetDefaultLayout()` by using
// `Client::GetDefaultPjRtLayout()` internally.
//
// TODO(hyeontaek): Remove this function once all IFRT clients are migrated to
// implement `Client::GetDefaultLayout()`.
absl::StatusOr<CustomLayoutRef> GetDefaultLayoutUsingDefaultPjRtLayout(
    const Client* client, DType dtype, const Shape& shape,
    const ShardingRef& sharding);

// Helper function for implementing `Array::layout()` by using
// `Array::pjrt_layout()` internally.
//
// TODO(hyeontaek): Remove this function once all IFRT clients are migrated to
// implement `Array::layout()`.
CustomLayoutRef GetArrayLayoutUsingPjRtLayout(const Array* array);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_MIGRATION_UTIL_H_
