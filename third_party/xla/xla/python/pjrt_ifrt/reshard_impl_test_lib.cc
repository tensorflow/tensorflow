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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

absl::StatusOr<ArrayRef> MakeArrayFromLiteral(Client* absl_nonnull client,
                                              const xla::LiteralBase& literal,
                                              const ShardingRef& sharding) {
  TF_ASSIGN_OR_RETURN(const DType dtype,
                      ToDType(literal.shape().element_type()));
  const Shape shape(literal.shape().dimensions());

  TF_ASSIGN_OR_RETURN(
      const std::vector<IndexDomain> index_domains,
      sharding->IndexDomains(shape,
                             SingleDeviceShardSemantics::kAddressableShards));

  Client::MakeArraysFromHostBufferShardsSpec spec = {
      /*buffers=*/{},
      /*array_spec=*/
      {
          /*dtype=*/dtype,
          /*shape=*/shape,
          /*sharding=*/sharding,
      },
  };
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

  TF_ASSIGN_OR_RETURN(
      std::vector<ArrayRef> arrays,
      client->MakeArraysFromHostBufferShards(
          absl::MakeSpan(&spec, 1),
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes));
  return arrays[0];
}

absl::StatusOr<xla::Literal> CopyArrayToLiteral(ArrayRef array) {
  TF_ASSIGN_OR_RETURN(const xla::PrimitiveType element_type,
                      ToPrimitiveType(array->dtype()));
  const auto xla_shape =
      xla::ShapeUtil::MakeShape(element_type, array->shape().dims());

  TF_ASSIGN_OR_RETURN(
      const std::vector<IndexDomain> index_domains,
      array->sharding().IndexDomains(
          array->shape(), SingleDeviceShardSemantics::kAddressableShards));
  TF_ASSIGN_OR_RETURN(std::vector<ArrayRef> shards,
                      array->DisassembleIntoSingleDeviceArrays(
                          ArrayCopySemantics::kReuseInput,
                          SingleDeviceShardSemantics::kAddressableShards));

  TF_ASSIGN_OR_RETURN(xla::Literal literal, xla::Literal::Make(xla_shape));
  absl::flat_hash_set<IndexDomain> seen;

  for (int i = 0; i < shards.size(); ++i) {
    const Index& offset = index_domains[i].origin();
    const Shape& shard_shape = index_domains[i].shape();

    TF_ASSIGN_OR_RETURN(xla::Literal slice,
                        xla::Literal::Make(xla::ShapeUtil::MakeShape(
                            element_type, shard_shape.dims())));
    tsl::Future<> future = shards[i]->CopyToHostBuffer(
        slice.untyped_data(), std::nullopt, ArrayCopySemantics::kAlwaysCopy);
    TF_RETURN_IF_ERROR(future.Await());
    VLOG(2) << "Slice #" << i << " (" << index_domains[i]
            << "): " << slice.ToString();

    if (seen.insert(index_domains[i]).second) {
      TF_RETURN_IF_ERROR(literal.CopySliceFrom(
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
                                               absl::Span<const int64_t> dims) {
  TF_ASSIGN_OR_RETURN(
      xla::Literal literal,
      xla::Literal::Make(xla::ShapeUtil::MakeShape(element_type, dims)));
  TF_RETURN_IF_ERROR(xla::primitive_util::IntegralTypeSwitch(
      [&](auto primitive_type_constant) -> absl::Status {
        using T = xla::primitive_util::NativeTypeOf<primitive_type_constant>;
        T value(0);
        return literal.Populate<T>(
            [&](absl::Span<const int64_t> indices) { return value++; });
      },
      literal.shape().element_type()));
  return literal;
}

class ReshardTest : public testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, test_util::GetClient());
  }

  std::shared_ptr<Client> client_;
};

TEST_F(ReshardTest, BatchedWithDifferentSharding) {
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          CreateIotaLiteral(xla::PrimitiveType::S32, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  std::vector<ArrayRef> src_arrays;
  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        MakeArrayFromLiteral(
            client_.get(), literal,
            HloSharding::Create(src_device_list, MemoryKind(),
                                xla::HloSharding::IotaTile({4, 2}))));
  }

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                          client_->MakeDeviceList(client_->devices()));
  std::vector<ArraySpec> array_specs = {
      {
          /*dtype=*/src_arrays[0]->dtype(),
          /*shape=*/src_arrays[0]->shape(),
          /*sharding=*/
          HloSharding::Create(dst_device_list, MemoryKind(),
                              xla::HloSharding::Replicate()),
      },
      {
          /*dtype=*/src_arrays[1]->dtype(),
          /*shape=*/src_arrays[1]->shape(),
          /*sharding=*/
          HloSharding::Create(dst_device_list, MemoryKind(),
                              xla::HloSharding::IotaTile({2, 4})),
      },
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      client_->ReshardArrays(absl::MakeSpan(src_arrays), array_specs,
                             ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_arrays[1]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_F(ReshardTest, BatchedWithDifferentDeviceLists) {
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          CreateIotaLiteral(xla::PrimitiveType::S32, {4, 8}));

  std::vector<ArrayRef> src_arrays;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(0, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        MakeArrayFromLiteral(
            client_.get(), literal,
            HloSharding::Create(src_device_list, MemoryKind(),
                                xla::HloSharding::IotaTile({2, 2}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(4, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        MakeArrayFromLiteral(
            client_.get(), literal,
            HloSharding::Create(src_device_list, MemoryKind(),
                                xla::HloSharding::IotaTile({2, 2}))));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list_0_4,
      client_->MakeDeviceList(client_->devices().subspan(0, 4)));
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list_4_4,
      client_->MakeDeviceList(client_->devices().subspan(4, 4)));
  std::vector<ArraySpec> array_specs = {
      {
          /*dtype=*/src_arrays[0]->dtype(),
          /*shape=*/src_arrays[0]->shape(),
          /*sharding=*/
          HloSharding::Create(std::move(device_list_0_4), MemoryKind(),
                              xla::HloSharding::Replicate()),
      },
      {
          /*dtype=*/src_arrays[1]->dtype(),
          /*shape=*/src_arrays[1]->shape(),
          /*sharding=*/
          HloSharding::Create(std::move(device_list_4_4), MemoryKind(),
                              xla::HloSharding::IotaTile({2, 2})),
      },
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      client_->ReshardArrays(absl::MakeSpan(src_arrays), array_specs,
                             ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_arrays[1]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_F(ReshardTest, PoisonedInput) {
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          CreateIotaLiteral(xla::PrimitiveType::S32, {4, 8}));
  const absl::Status error = absl::InternalError("injected error");

  std::vector<ArrayRef> src_arrays;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(0, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        MakeArrayFromLiteral(
            client_.get(), literal,
            HloSharding::Create(src_device_list, MemoryKind(),
                                xla::HloSharding::IotaTile({2, 2}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(4, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        auto arrays,
        client_->MakeErrorArrays(
            error, {{
                       /*dtype=*/DType(DType::kS32),
                       /*shape=*/Shape({4, 8}),
                       /*sharding=*/
                       HloSharding::Create(src_device_list, MemoryKind(),
                                           xla::HloSharding::IotaTile({2, 2})),
                   }}));
    src_arrays.push_back(std::move(arrays[0]));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list_0_4,
      client_->MakeDeviceList(client_->devices().subspan(0, 4)));
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list_4_4,
      client_->MakeDeviceList(client_->devices().subspan(4, 4)));
  std::vector<ArraySpec> array_specs = {
      {
          /*dtype=*/src_arrays[0]->dtype(),
          /*shape=*/src_arrays[0]->shape(),
          /*sharding=*/
          HloSharding::Create(std::move(device_list_0_4), MemoryKind(),
                              xla::HloSharding::Replicate()),
      },
      {
          /*dtype=*/src_arrays[1]->dtype(),
          /*shape=*/src_arrays[1]->shape(),
          /*sharding=*/
          HloSharding::Create(std::move(device_list_4_4), MemoryKind(),
                              xla::HloSharding::IotaTile({2, 2})),
      },
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      client_->ReshardArrays(absl::MakeSpan(src_arrays), array_specs,
                             ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(dst_arrays[1]->GetReadyFuture().Await(),
              absl_testing::StatusIs(error.code(), HasSubstr(error.message())));
}

TEST_F(ReshardTest, DifferentDestinationLayout) {
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          CreateIotaLiteral(xla::PrimitiveType::S32, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      MakeArrayFromLiteral(
          client_.get(), literal,
          HloSharding::Create(src_device_list, MemoryKind(),
                              xla::HloSharding::IotaTile({4, 2}))));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                          client_->MakeDeviceList(client_->devices()));
  ArraySpec dst_array_spec = {
      /*dtype=*/src_array->dtype(),
      /*shape=*/src_array->shape(),
      /*sharding=*/
      HloSharding::Create(dst_device_list, MemoryKind(),
                          xla::HloSharding::Replicate()),
      /*layout=*/
      std::make_shared<xla::PjRtLayout>(
          xla::LayoutUtil::MakeAscendingLayout(2)),
  };

  // Make sure that the destination layout is actually different from the source
  // layout in order to ensure the test coverage.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> src_layout,
                          src_array->pjrt_layout());
  if (src_layout == nullptr) {
    TF_ASSERT_OK_AND_ASSIGN(
        Shape shard_shape,
        src_array->sharding().GetShardShape(src_array->shape()));
    TF_ASSERT_OK_AND_ASSIGN(
        src_layout, client_->GetDefaultPjRtLayout(
                        src_array->dtype(), shard_shape.dims(),
                        src_array->sharding().devices()->devices().front(),
                        src_array->sharding().memory_kind()));
  }
  ASSERT_NE(src_layout->xla_layout(), dst_array_spec.layout->xla_layout());

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      client_->ReshardArrays(absl::MakeSpan(&src_array, 1), {dst_array_spec},
                             ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);

  // Verify that the destination array is created with the user-provided layout.
  TF_ASSERT_OK_AND_ASSIGN(const auto dst_layout, dst_array->pjrt_layout());
  ASSERT_NE(dst_layout, nullptr);
  EXPECT_EQ(dst_layout->xla_layout(), dst_array_spec.layout->xla_layout());

  EXPECT_THAT(CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

class ReshardMemoryKindTest : public ReshardTest,
                              public testing::WithParamInterface<MemoryKind> {};

TEST_P(ReshardMemoryKindTest, Int4) {
  const MemoryKind memory_kind = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          CreateIotaLiteral(xla::PrimitiveType::S4, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      MakeArrayFromLiteral(
          client_.get(), literal,
          HloSharding::Create(src_device_list, memory_kind,
                              xla::HloSharding::IotaTile({4, 2}))));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                          client_->MakeDeviceList(client_->devices()));
  ArraySpec dst_array_spec = {
      /*dtype=*/src_array->dtype(),
      /*shape=*/src_array->shape(),
      /*sharding=*/
      HloSharding::Create(dst_device_list, memory_kind,
                          xla::HloSharding::IotaTile({2, 4})),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      client_->ReshardArrays(absl::MakeSpan(&src_array, 1), {dst_array_spec},
                             ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);
  EXPECT_THAT(CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

auto AllMemoryKinds() {
  return testing::Values(MemoryKind("device"), MemoryKind("pinned_host"),
                         MemoryKind("unpinned_host"));
}

INSTANTIATE_TEST_SUITE_P(
    AllMemoryKinds, ReshardMemoryKindTest, AllMemoryKinds(),
    [](const testing::TestParamInfo<ReshardMemoryKindTest::ParamType>& info)
        -> std::string { return absl::StrCat(info.param); });

struct ReshardTestParam {
  absl::string_view name;

  Shape shape;

  xla::HloSharding src_sharding;
  std::vector<int> src_device_indices;

  xla::HloSharding dst_sharding;
  std::vector<int> dst_device_indices;
};

class ReshardParameterizedTest
    : public ReshardTest,
      public testing::WithParamInterface<
          std::tuple<ReshardTestParam, MemoryKind, MemoryKind>> {};

TEST_P(ReshardParameterizedTest, RoundTrip) {
  const auto& [param, src_memory_kind, dst_memory_kind] = GetParam();

  absl::InlinedVector<Device*, 1> src_devices;
  src_devices.reserve(param.src_device_indices.size());
  for (const int index : param.src_device_indices) {
    src_devices.push_back(client_->devices()[index]);
  }
  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(src_devices));
  const ShardingRef src_sharding = HloSharding::Create(
      std::move(src_device_list), src_memory_kind, param.src_sharding);

  absl::InlinedVector<Device*, 1> dst_devices;
  dst_devices.reserve(param.dst_device_indices.size());
  for (const int index : param.dst_device_indices) {
    dst_devices.push_back(client_->devices()[index]);
  }
  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                          client_->MakeDeviceList(dst_devices));
  const ShardingRef dst_sharding = HloSharding::Create(
      std::move(dst_device_list), dst_memory_kind, param.dst_sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal,
      CreateIotaLiteral(xla::PrimitiveType::S32, param.shape.dims()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      MakeArrayFromLiteral(client_.get(), literal, src_sharding));

  // Reshard from source to destination.
  ArrayRef dst_array;
  {
    SCOPED_TRACE(absl::StrCat(*src_sharding, " -> ", *dst_sharding));

    ArraySpec array_spec = {
        /*dtype=*/src_array->dtype(),
        /*shape=*/src_array->shape(),
        /*sharding=*/dst_sharding,
    };
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<ArrayRef> dst_arrays,
        client_->ReshardArrays(absl::MakeSpan(&src_array, 1), {array_spec},
                               ArrayCopySemantics::kDonateInput));
    ASSERT_EQ(dst_arrays.size(), 1);
    dst_array = std::move(dst_arrays[0]);

    EXPECT_EQ(dst_array->sharding(), *array_spec.sharding);
    EXPECT_THAT(CopyArrayToLiteral(dst_array),
                absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
  }

  // Reshard from destination back to source.
  {
    SCOPED_TRACE(absl::StrCat(*dst_sharding, " -> ", *src_sharding));

    ArraySpec array_spec = {
        /*dtype=*/dst_array->dtype(),
        /*shape=*/dst_array->shape(),
        /*sharding=*/src_sharding,
    };
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<ArrayRef> src_arrays,
        client_->ReshardArrays(absl::MakeSpan(&dst_array, 1), {array_spec},
                               ArrayCopySemantics::kDonateInput));
    ASSERT_EQ(src_arrays.size(), 1);
    src_array = std::move(src_arrays[0]);

    EXPECT_EQ(src_array->sharding(), *array_spec.sharding);
    EXPECT_THAT(CopyArrayToLiteral(src_array),
                absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SameDeviceCount, ReshardParameterizedTest,
    testing::Combine(     //
        testing::Values(  //
            ReshardTestParam{
                /*name=*/"ReplicateToReplicate",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ReplicateToReplicateDeviceLayout",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ReplicateToTile",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::IotaTile({4, 2}),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"TileToTile",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::IotaTile({2, 4}),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::IotaTile({4, 2}),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"TileToPartialTile",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::IotaTile({4, 2}),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/
                xla::HloSharding::PartialTile(xla::TileAssignment({1, 4, 2})),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ZeroSized",
                /*shape=*/Shape({0, 4}),
                /*src_sharding=*/xla::HloSharding::IotaTile({4, 2}),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::IotaTile({2, 4}),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            }),
        AllMemoryKinds(), AllMemoryKinds()),
    ([](const testing::TestParamInfo<ReshardParameterizedTest::ParamType>&
            info) {
      const auto& [param, src_memory_kind, dst_memory_kind] = info.param;
      return absl::StrCat(param.name, "_", src_memory_kind, "_to_",
                          dst_memory_kind);
    }));

INSTANTIATE_TEST_SUITE_P(
    DifferentDeviceCount, ReshardParameterizedTest,
    testing::Combine(     //
        testing::Values(  //
            ReshardTestParam{
                /*name=*/"ReplicateToReplicate",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ReplicateToReplicateDeviceLayout",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ReplicateToTile",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1},
                /*dst_sharding=*/xla::HloSharding::IotaTile({4, 2}),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"TileToTile",
                /*shape=*/Shape({4, 4, 4}),
                /*src_sharding=*/xla::HloSharding::IotaTile({2, 2, 2}),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::IotaTile({1, 2, 1}),
                /*dst_device_indices=*/{4, 0},
            },
            ReshardTestParam{
                /*name=*/"TileToPartialTile",
                /*shape=*/Shape({4, 8}),
                /*src_sharding=*/xla::HloSharding::IotaTile({2, 1}),
                /*src_device_indices=*/{1, 0},
                /*dst_sharding=*/
                xla::HloSharding::PartialTile(xla::TileAssignment({1, 4, 2})),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
            ReshardTestParam{
                /*name=*/"ZeroSized",
                /*shape=*/Shape({0, 4}),
                /*src_sharding=*/xla::HloSharding::IotaTile({2, 1}),
                /*src_device_indices=*/{0, 1},
                /*dst_sharding=*/xla::HloSharding::IotaTile({2, 4}),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            }),
        AllMemoryKinds(), AllMemoryKinds()),
    ([](const testing::TestParamInfo<ReshardParameterizedTest::ParamType>&
            info) {
      const auto& [param, src_memory_kind, dst_memory_kind] = info.param;
      return absl::StrCat(param.name, "_", src_memory_kind, "_to_",
                          dst_memory_kind);
    }));

}  // namespace
}  // namespace ifrt
}  // namespace xla
