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

#include "xla/python/pjrt_ifrt/array_test_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace test_util {

absl::StatusOr<ArrayRef> MakeArrayFromLiteral(
    Client* absl_nonnull client, const xla::LiteralBase& literal,
    const ShardingRef& sharding,
    std::shared_ptr<const xla::PjRtLayout> layout) {
  ABSL_ASSIGN_OR_RETURN(const DType dtype, ToDType(literal.shape().element_type()));
  const Shape shape(literal.shape().dimensions());

  ABSL_ASSIGN_OR_RETURN(const std::vector<IndexDomain> index_domains,
                   sharding->IndexDomains(
                       shape, SingleDeviceShardSemantics::kAddressableShards));

  Client::MakeArraysFromHostBufferShardsSpec spec = {
      /*buffers=*/{},
      /*array_spec=*/
      {
          /*dtype=*/dtype,
          /*shape=*/shape,
          /*sharding=*/sharding,
          /*layout=*/std::move(layout),
      },
  };
  spec.buffers.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    const Index& offset = index_domains[i].origin();
    const Shape& shard_shape = index_domains[i].shape();

    const Index limit = offset + Index(shard_shape.dims());
    auto sliced = std::make_shared<xla::Literal>(
        literal.Slice(offset.elements(), limit.elements()));
    VLOG(2) << "Slice #" << i << "(" << index_domains[i]
            << "): " << sliced->ToString();

    Client::HostBuffer host_buffer = {
        /*data=*/sliced->untyped_data(),
        /*dtype=*/dtype,
        /*shape=*/shard_shape,
        /*byte_strides=*/std::nullopt,
        /*on_done=*/[sliced]() {},
    };
    spec.buffers.push_back({{i}, std::move(host_buffer)});
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<ArrayRef> arrays,
      client->MakeArraysFromHostBufferShards(
          absl::MakeSpan(&spec, 1),
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes));
  return std::move(arrays[0]);
}

absl::StatusOr<xla::Literal> CopyArrayToLiteral(ArrayRef array) {
  ABSL_ASSIGN_OR_RETURN(const xla::PrimitiveType element_type,
                   ToPrimitiveType(array->dtype()));
  const auto xla_shape =
      xla::ShapeUtil::MakeShape(element_type, array->shape().dims());

  ABSL_ASSIGN_OR_RETURN(
      const std::vector<IndexDomain> index_domains,
      array->sharding().IndexDomains(
          array->shape(), SingleDeviceShardSemantics::kAddressableShards));
  ABSL_ASSIGN_OR_RETURN(std::vector<ArrayRef> shards,
                   array->DisassembleIntoSingleDeviceArrays(
                       ArrayCopySemantics::kReuseInput,
                       SingleDeviceShardSemantics::kAddressableShards));

  ABSL_ASSIGN_OR_RETURN(xla::Literal literal, xla::Literal::Make(xla_shape));
  absl::flat_hash_set<IndexDomain> seen;

  for (int i = 0; i < shards.size(); ++i) {
    const Index& offset = index_domains[i].origin();
    const Shape& shard_shape = index_domains[i].shape();

    ABSL_ASSIGN_OR_RETURN(xla::Literal slice,
                     xla::Literal::Make(xla::ShapeUtil::MakeShape(
                         element_type, shard_shape.dims())));
    tsl::Future<> future = shards[i]->CopyToHostBuffer(
        slice.untyped_data(), std::nullopt, ArrayCopySemantics::kAlwaysCopy);
    ABSL_RETURN_IF_ERROR(future.Await());
    VLOG(2) << "Slice #" << i << " (" << index_domains[i]
            << "): " << slice.ToString();

    if (seen.insert(index_domains[i]).second) {
      ABSL_RETURN_IF_ERROR(literal.CopySliceFrom(
          slice, Index::Zeros(shard_shape.dims().size()).elements(),
          offset.elements(), shard_shape.dims()));
    } else {
      Index limits = offset + Index(shard_shape.dims());
      const xla::Literal expected =
          literal.Slice(offset.elements(), limits.elements());
      if (slice != expected) {
        return absl::InternalError(
            absl::StrCat("Inconsistent replication in ", index_domains[i], ": ",
                         slice.ToString(), " vs. ", expected.ToString()));
      }
    }
  }
  return literal;
}

absl::StatusOr<xla::Literal> CreateIotaLiteral(xla::PrimitiveType element_type,
                                               absl::Span<const int64_t> dims,
                                               int start) {
  if (!xla::primitive_util::IsIntegralType(element_type)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CreateIotaLiteral only supports integral types; received: ",
        xla::primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
  ABSL_ASSIGN_OR_RETURN(const xla::Shape shape,
                   xla::ShapeUtil::MakeValidatedShape(element_type, dims));
  ABSL_ASSIGN_OR_RETURN(xla::Literal literal, xla::Literal::Make(shape));
  ABSL_RETURN_IF_ERROR(xla::primitive_util::IntegralTypeSwitch(
      [&](auto primitive_type_constant) -> absl::Status {
        using T = xla::primitive_util::NativeTypeOf<primitive_type_constant>;
        T value(start);
        return literal.Populate<T>([&](absl::Span<const int64_t> indices) {
          T current = value;
          value = T(static_cast<int>(value) + 1);
          return current;
        });
      },
      literal.shape().element_type()));
  return literal;
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
