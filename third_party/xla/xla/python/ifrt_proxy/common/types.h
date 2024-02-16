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
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
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

DType FromDTypeProto(proto::DType dtype_proto);
proto::DType ToDTypeProto(DType dtype);

Shape FromShapeProto(const proto::Shape& shape_proto);
proto::Shape ToShapeProto(const Shape& shape);

absl::StatusOr<std::shared_ptr<const Sharding>> FromShardingProto(
    DeviceList::LookupDeviceFunc lookup_device,
    const proto::Sharding& sharding_proto);
absl::StatusOr<proto::Sharding> ToShardingProto(const Sharding& sharding);

absl::StatusOr<ArrayCopySemantics> FromArrayCopySemanticsProto(
    proto::ArrayCopySemantics s);
proto::ArrayCopySemantics ToArrayCopySemanticsProto(ArrayCopySemantics s);

absl::StatusOr<xla::PjRtValueType> FromVariantProto(
    const proto::Variant& variant_proto);
absl::StatusOr<proto::Variant> ToVariantProto(const xla::PjRtValueType& value);

std::vector<int64_t> FromByteStridesProto(const proto::ByteStrides& strides);
proto::ByteStrides ToByteStridesProto(absl::Span<const int64_t> strides);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_TYPES_H_
