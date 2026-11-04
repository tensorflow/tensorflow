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

#ifndef XLA_PYTHON_PJRT_IFRT_ARRAY_TEST_UTIL_H_
#define XLA_PYTHON_PJRT_IFRT_ARRAY_TEST_UTIL_H_

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Creates a sharded array from the given literal. `sharding` must support
// index domain calculation.
absl::StatusOr<ArrayRef> MakeArrayFromLiteral(
    Client* absl_nonnull client, const xla::LiteralBase& literal,
    const ShardingRef& sharding,
    std::shared_ptr<const xla::PjRtLayout> layout = nullptr);

// Copies a sharded array into a literal. The array's sharding must support
// index domain calculation.
absl::StatusOr<xla::Literal> CopyArrayToLiteral(ArrayRef array);

// Creates a literal with integral `element_type` and `dims` filled with
// sequential numbers starting from `start`.
absl::StatusOr<xla::Literal> CreateIotaLiteral(xla::PrimitiveType element_type,
                                               absl::Span<const int64_t> dims,
                                               int start = 0);

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_ARRAY_TEST_UTIL_H_
