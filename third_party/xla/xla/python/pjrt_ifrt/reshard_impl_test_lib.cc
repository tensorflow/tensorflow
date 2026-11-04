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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt/value_util.h"
#include "xla/python/pjrt_ifrt/array_test_util.h"
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

enum class ReshardMethod {
  kArray,
  kBundle,
};

template <typename Sink>
void AbslStringify(Sink& sink, ReshardMethod method) {
  switch (method) {
    case ReshardMethod::kArray:
      sink.Append("Array");
      break;
    case ReshardMethod::kBundle:
      sink.Append("Bundle");
      break;
  }
}

absl::StatusOr<std::vector<ArrayRef>> ReshardArrays(
    ReshardMethod method, Client* client, absl::Span<ArrayRef> src_arrays,
    absl::Span<const ArraySpec> array_specs, ArrayCopySemantics semantics) {
  switch (method) {
    case ReshardMethod::kArray:
      return client->ReshardArrays(src_arrays, array_specs, semantics);
    case ReshardMethod::kBundle: {
      std::vector<ValueRef> src_values = ToValues(src_arrays);
      ABSL_ASSIGN_OR_RETURN(BundleRef bundle,
                       client->Bundle(absl::MakeSpan(src_values),
                                      ArrayCopySemantics::kReuseInput));
      ABSL_ASSIGN_OR_RETURN(BundleRef resharded_bundle,
                       bundle->ReshardArrays(array_specs, semantics));
      ABSL_ASSIGN_OR_RETURN(
          std::vector<ValueRef> retrieved_values,
          resharded_bundle->GetValues(ArrayCopySemantics::kReuseInput));
      return ToArrays(absl::MakeSpan(retrieved_values));
    }
  }
}

class ReshardTestBase : public testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, test_util::GetClient());
  }

  std::shared_ptr<Client> client_;
};

class ReshardTest : public ReshardTestBase,
                    public testing::WithParamInterface<
                        std::tuple<ReshardMethod, xla::PrimitiveType>> {};

TEST_P(ReshardTest, BatchedWithDifferentSharding) {
  const auto [method, primitive_type] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          test_util::CreateIotaLiteral(primitive_type, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  std::vector<ArrayRef> src_arrays;
  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        test_util::MakeArrayFromLiteral(
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(src_arrays),
                    array_specs, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_arrays[1]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_P(ReshardTest, BatchedWithDifferentDeviceLists) {
  const auto [method, primitive_type] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          test_util::CreateIotaLiteral(primitive_type, {4, 8}));

  std::vector<ArrayRef> src_arrays;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(0, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        test_util::MakeArrayFromLiteral(
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
        test_util::MakeArrayFromLiteral(
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(src_arrays),
                    array_specs, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_arrays[1]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_P(ReshardTest, PoisonedInput) {
  const auto [method, primitive_type] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          test_util::CreateIotaLiteral(primitive_type, {4, 8}));
  TF_ASSERT_OK_AND_ASSIGN(const DType dtype, ToDType(primitive_type));
  const absl::Status error = absl::InternalError("injected error");

  std::vector<ArrayRef> src_arrays;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const DeviceListRef src_device_list,
        client_->MakeDeviceList(client_->devices().subspan(0, 4)));
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays.emplace_back(),
        test_util::MakeArrayFromLiteral(
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
                       /*dtype=*/dtype,
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(src_arrays),
                    array_specs, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 2);

  EXPECT_EQ(dst_arrays[0]->sharding(), *array_specs[0].sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_arrays[0]),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));

  EXPECT_EQ(dst_arrays[1]->sharding(), *array_specs[1].sharding);
  EXPECT_THAT(dst_arrays[1]->GetReadyFuture().Await(),
              absl_testing::StatusIs(error.code(), HasSubstr(error.message())));
}

TEST_P(ReshardTest, DifferentDestinationLayout) {
  if (client_->platform_id() == xla::CpuId()) {
    GTEST_SKIP() << "PjRt CPU does not support custom layouts";
  }

  const auto [method, primitive_type] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                          test_util::CreateIotaLiteral(primitive_type, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      test_util::MakeArrayFromLiteral(
          client_.get(), literal,
          HloSharding::Create(src_device_list, MemoryKind(),
                              xla::HloSharding::IotaTile({4, 2}))));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                          client_->MakeDeviceList(client_->devices()));

  xla::Layout layout = xla::LayoutUtil::MakeAscendingLayout(2);
  if (auto ifrt_topology = client_->GetTopologyForDevices(dst_device_list);
      ifrt_topology.ok()) {
    const xla::PjRtTopologyDescription* topology =
        (*ifrt_topology)->description().get();
    TF_ASSERT_OK_AND_ASSIGN(xla::Shape shape,
                            topology->MakeCanonicalShapeForMemorySpace(
                                topology->GetDefaultMemorySpaceKindId(),
                                xla::ShapeUtil::MakeShape(
                                    primitive_type, src_array->shape().dims()),
                                &layout));
    layout = shape.layout();
  }

  ArraySpec dst_array_spec = {
      /*dtype=*/src_array->dtype(),
      /*shape=*/src_array->shape(),
      /*sharding=*/
      HloSharding::Create(dst_device_list, MemoryKind(),
                          xla::HloSharding::Replicate()),
      /*layout=*/std::make_shared<const xla::PjRtLayout>(std::move(layout)),
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(&src_array, 1),
                    {dst_array_spec}, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);

  // Verify that the destination array is created with the user-provided layout.
  TF_ASSERT_OK_AND_ASSIGN(const auto dst_layout, dst_array->pjrt_layout());
  ASSERT_NE(dst_layout, nullptr);
  EXPECT_EQ(dst_layout->xla_layout(), dst_array_spec.layout->xla_layout());

  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_P(ReshardTest, DifferentSourceLayout) {
  if (client_->platform_id() == xla::CpuId()) {
    GTEST_SKIP() << "PjRt CPU does not support custom layouts";
  }

  const auto [method, primitive_type] = GetParam();
  ASSERT_OK_AND_ASSIGN(const xla::Literal literal,
                       test_util::CreateIotaLiteral(primitive_type, {4, 8}));

  ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                       client_->MakeDeviceList(client_->devices()));

  xla::Layout layout = xla::LayoutUtil::MakeAscendingLayout(2);
  if (auto ifrt_topology = client_->GetTopologyForDevices(src_device_list);
      ifrt_topology.ok()) {
    const xla::PjRtTopologyDescription* topology =
        (*ifrt_topology)->description().get();
    ASSERT_OK_AND_ASSIGN(xla::Shape shape,
                         topology->MakeCanonicalShapeForMemorySpace(
                             topology->GetDefaultMemorySpaceKindId(),
                             xla::ShapeUtil::MakeShape(
                                 primitive_type, literal.shape().dimensions()),
                             &layout));
    layout = shape.layout();
  }
  auto src_layout = std::make_shared<const xla::PjRtLayout>(std::move(layout));

  ASSERT_OK_AND_ASSIGN(ArrayRef src_array,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), literal,
                           HloSharding::Create(src_device_list, MemoryKind(),
                                               xla::HloSharding::Replicate()),
                           src_layout));

  ASSERT_OK_AND_ASSIGN(const DeviceListRef dst_device_list,
                       client_->MakeDeviceList(client_->devices()));

  ArraySpec dst_array_spec = {
      /*dtype=*/src_array->dtype(),
      /*shape=*/src_array->shape(),
      /*sharding=*/
      HloSharding::Create(dst_device_list, MemoryKind(),
                          xla::HloSharding::Replicate()),
      /*layout=*/nullptr,
  };

  // Make sure that the source layout is actually different from the destination
  // layout in order to ensure the test coverage.
  ASSERT_OK_AND_ASSIGN(const auto actual_src_layout, src_array->pjrt_layout());
  ASSERT_NE(actual_src_layout, nullptr);
  EXPECT_EQ(actual_src_layout->xla_layout(), src_layout->xla_layout());

  ASSERT_OK_AND_ASSIGN(
      Shape shard_shape,
      dst_array_spec.sharding->GetShardShape(dst_array_spec.shape));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const xla::PjRtLayout> default_dst_layout,
      client_->GetDefaultPjRtLayout(
          dst_array_spec.dtype, shard_shape.dims(),
          dst_array_spec.sharding->devices()->devices().front(),
          dst_array_spec.sharding->memory_kind()));
  ASSERT_NE(actual_src_layout->xla_layout(), default_dst_layout->xla_layout());

  ASSERT_OK_AND_ASSIGN(
      std::vector<ArrayRef> dst_arrays,
      ReshardArrays(method, client_.get(), absl::MakeSpan(&src_array, 1),
                    {dst_array_spec}, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);

  // Verify that the destination array is created with the default layout.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> dst_layout,
                       dst_array->pjrt_layout());
  if (dst_layout == nullptr) {
    ASSERT_OK_AND_ASSIGN(dst_layout,
                         client_->GetDefaultPjRtLayout(
                             dst_array->dtype(), shard_shape.dims(),
                             dst_array->sharding().devices()->devices().front(),
                             dst_array->sharding().memory_kind()));
  }
  EXPECT_EQ(dst_layout->xla_layout(), default_dst_layout->xla_layout());

  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

INSTANTIATE_TEST_SUITE_P(
    ReshardTest, ReshardTest,
    testing::Combine(
        testing::Values(ReshardMethod::kArray, ReshardMethod::kBundle),
        testing::Values(xla::PrimitiveType::S32, xla::PrimitiveType::S64)),
    ([](const testing::TestParamInfo<ReshardTest::ParamType>& info) {
      const auto& [method, primitive_type] = info.param;
      return absl::StrCat(
          method, "_",
          xla::primitive_util::LowercasePrimitiveTypeName(primitive_type));
    }));

class ReshardMemoryKindTest : public ReshardTestBase,
                              public testing::WithParamInterface<
                                  std::tuple<ReshardMethod, MemoryKind>> {};

TEST_P(ReshardMemoryKindTest, Int4) {
  const auto& [method, memory_kind] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S4, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      test_util::MakeArrayFromLiteral(
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(&src_array, 1),
                    {dst_array_spec}, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

TEST_P(ReshardMemoryKindTest, Int64) {
  const auto& [method, memory_kind] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S64, {4, 8}));

  TF_ASSERT_OK_AND_ASSIGN(const DeviceListRef src_device_list,
                          client_->MakeDeviceList(client_->devices()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      test_util::MakeArrayFromLiteral(
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
      ReshardArrays(method, client_.get(), absl::MakeSpan(&src_array, 1),
                    {dst_array_spec}, ArrayCopySemantics::kDonateInput));
  ASSERT_EQ(dst_arrays.size(), 1);

  const ArrayRef& dst_array = dst_arrays[0];
  EXPECT_EQ(dst_array->sharding(), *dst_array_spec.sharding);
  EXPECT_THAT(test_util::CopyArrayToLiteral(dst_array),
              absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
}

auto AllMemoryKinds() {
  return testing::Values(MemoryKind("device"), MemoryKind("pinned_host"),
                         MemoryKind("unpinned_host"));
}

INSTANTIATE_TEST_SUITE_P(
    AllMemoryKinds, ReshardMemoryKindTest,
    testing::Combine(testing::Values(ReshardMethod::kArray,
                                     ReshardMethod::kBundle),
                     AllMemoryKinds()),
    ([](const testing::TestParamInfo<ReshardMemoryKindTest::ParamType>& info)
         -> std::string {
      const auto& [method, memory_kind] = info.param;
      return absl::StrCat(method, "_", memory_kind);
    }));

struct ReshardTestParam {
  absl::string_view name;

  Shape shape;

  xla::HloSharding src_sharding;
  std::vector<int> src_device_indices;

  xla::HloSharding dst_sharding;
  std::vector<int> dst_device_indices;
};

class ReshardParameterizedTest
    : public ReshardTestBase,
      public testing::WithParamInterface<
          std::tuple<ReshardMethod, ReshardTestParam, MemoryKind, MemoryKind,
                     xla::PrimitiveType>> {};

TEST_P(ReshardParameterizedTest, RoundTrip) {
  const auto& [method, param, src_memory_kind, dst_memory_kind,
               primitive_type] = GetParam();

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
      test_util::CreateIotaLiteral(primitive_type, param.shape.dims()));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef src_array,
      test_util::MakeArrayFromLiteral(client_.get(), literal, src_sharding));

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
        ReshardArrays(method, client_.get(), absl::MakeSpan(&src_array, 1),
                      {array_spec}, ArrayCopySemantics::kDonateInput));
    ASSERT_EQ(dst_arrays.size(), 1);
    dst_array = std::move(dst_arrays[0]);

    EXPECT_EQ(dst_array->sharding(), *array_spec.sharding);
    EXPECT_THAT(test_util::CopyArrayToLiteral(dst_array),
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
        ReshardArrays(method, client_.get(), absl::MakeSpan(&dst_array, 1),
                      {array_spec}, ArrayCopySemantics::kDonateInput));
    ASSERT_EQ(src_arrays.size(), 1);
    src_array = std::move(src_arrays[0]);

    EXPECT_EQ(src_array->sharding(), *array_spec.sharding);
    EXPECT_THAT(test_util::CopyArrayToLiteral(src_array),
                absl_testing::IsOkAndHolds(Eq(std::cref(literal))));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SameDeviceCount, ReshardParameterizedTest,
    testing::Combine(  //
        testing::Values(ReshardMethod::kArray, ReshardMethod::kBundle),
        testing::Values(  //
            ReshardTestParam{
                /*name=*/"Scalar",
                /*shape=*/Shape({}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
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
        AllMemoryKinds(), AllMemoryKinds(),
        testing::Values(xla::PrimitiveType::S32, xla::PrimitiveType::S64)),
    ([](const testing::TestParamInfo<ReshardParameterizedTest::ParamType>&
            info) {
      const auto& [method, param, src_memory_kind, dst_memory_kind,
                   primitive_type] = info.param;
      return absl::StrCat(
          method, "_", param.name, "_", src_memory_kind, "_to_",
          dst_memory_kind, "_",
          xla::primitive_util::LowercasePrimitiveTypeName(primitive_type));
    }));

INSTANTIATE_TEST_SUITE_P(
    DifferentDeviceCount, ReshardParameterizedTest,
    testing::Combine(  //
        testing::Values(ReshardMethod::kArray, ReshardMethod::kBundle),
        testing::Values(  //
            ReshardTestParam{
                /*name=*/"Scalar",
                /*shape=*/Shape({}),
                /*src_sharding=*/xla::HloSharding::Replicate(),
                /*src_device_indices=*/{0, 1},
                /*dst_sharding=*/xla::HloSharding::Replicate(),
                /*dst_device_indices=*/{0, 1, 2, 3, 4, 5, 6, 7},
            },
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
            },
            ReshardTestParam{
                /*name=*/"TileGather",
                /*shape=*/Shape({12, 1, 32, 128}),
                /*src_sharding=*/xla::HloSharding::IotaTile({4, 1, 1, 1}),
                /*src_device_indices=*/{0, 1, 2, 3},
                /*dst_sharding=*/xla::HloSharding::IotaTile({1, 1, 1, 1}),
                /*dst_device_indices=*/{0},
            }),
        AllMemoryKinds(), AllMemoryKinds(),
        testing::Values(xla::PrimitiveType::S32, xla::PrimitiveType::S64)),
    ([](const testing::TestParamInfo<ReshardParameterizedTest::ParamType>&
            info) {
      const auto& [method, param, src_memory_kind, dst_memory_kind,
                   primitive_type] = info.param;
      return absl::StrCat(
          method, "_", param.name, "_", src_memory_kind, "_to_",
          dst_memory_kind, "_",
          xla::primitive_util::LowercasePrimitiveTypeName(primitive_type));
    }));

}  // namespace
}  // namespace ifrt
}  // namespace xla
