/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_TYPES_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_TYPES_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

struct ArrayHandle {
  uint64_t handle;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ArrayHandle& h) {
    absl::Format(&sink, "arr_%v", h.handle);
  }
};

absl::StatusOr<ArrayCopySemantics> FromArrayCopySemanticsProto(
    proto::ArrayCopySemantics s);
proto::ArrayCopySemantics ToArrayCopySemanticsProto(ArrayCopySemantics s);

absl::StatusOr<SingleDeviceShardSemantics> FromSingleDeviceShardSemanticsProto(
    proto::SingleDeviceShardSemantics s);
proto::SingleDeviceShardSemantics ToSingleDeviceShardSemanticsProto(
    SingleDeviceShardSemantics s);

absl::StatusOr<xla::PjRtValueType> FromVariantProto(
    const proto::Variant& variant_proto);
absl::StatusOr<proto::Variant> ToVariantProto(const xla::PjRtValueType& value);

std::vector<int64_t> FromByteStridesProto(const proto::ByteStrides& strides);
proto::ByteStrides ToByteStridesProto(absl::Span<const int64_t> strides);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_TYPES_H_
