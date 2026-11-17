/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

using XlaShardingSerDesTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class XlaShardingSerDesTest
    : public testing::TestWithParam<XlaShardingSerDesTestParam> {
 public:
  XlaShardingSerDesTest()
      : version_(std::get<0>(GetParam())), fixture_(std::get<1>(GetParam())) {}

  SerDesVersion version() const { return version_; }

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  SerDesVersion version_;
  test_util::DeviceTestFixture fixture_;
};

TEST_P(XlaShardingSerDesTest, HloShardingRoundTrip) {
  auto device_list = GetDevices({0, 1});
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  auto sharding = HloSharding::Create(device_list, MemoryKind("abc"),
                                      /*xla_hlo_sharding=*/xla_hlo_sharding);

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<HloSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(client())));

  EXPECT_THAT(out_sharding->devices()->devices(),
              ElementsAreArray(sharding->devices()->devices()));
  EXPECT_EQ(out_sharding->xla_hlo_sharding(), sharding->xla_hlo_sharding());
}

TEST_P(XlaShardingSerDesTest, RejectHloShardingV3Serialization) {
  DeviceListRef device_list = GetDevices({0, 1});
  xla::HloSharding xla_hlo_sharding =
      xla::HloSharding::ToV3Sharding(xla::HloSharding::IotaTile({2, 1}));
  std::unique_ptr<HloSharding> sharding = HloSharding::Create(
      device_list, MemoryKind("abc"), /*xla_hlo_sharding=*/xla_hlo_sharding);

  auto options = std::make_unique<SerializeOptions>(version());
  EXPECT_THAT(
      Serialize(*sharding, std::move(options)),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "XLA HloShardingV3 format is not supported for serialization")));
}

TEST_P(XlaShardingSerDesTest, RejectHloShardingV3Deserialization) {
  DeviceListRef device_list = GetDevices({0, 1});
  xla::HloSharding xla_hlo_sharding = xla::HloSharding::IotaTile({2, 1});
  std::unique_ptr<HloSharding> sharding = HloSharding::Create(
      device_list, MemoryKind("abc"), /*xla_hlo_sharding=*/xla_hlo_sharding);
  ASSERT_OK_AND_ASSIGN(
      auto serialized,
      Serialize(*sharding, std::make_unique<SerializeOptions>(version())));

  HloShardingProto proto;
  ASSERT_TRUE(proto.ParseFromString(serialized.data()));
  xla::OpSharding v3_op_sharding =
      xla::HloSharding::ToV3Sharding(xla_hlo_sharding).ToProto();
  *proto.mutable_xla_op_sharding() = v3_op_sharding;
  serialized.set_data(proto.SerializeAsCord());

  EXPECT_THAT(
      Deserialize<HloSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(client())),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("XLA HloShardingV3 format is not supported for "
                         "deserialization")));

  // Also reject V3 shardings nested inside a tuple OpSharding.
  xla::OpSharding* tuple_op_sharding = proto.mutable_xla_op_sharding();
  tuple_op_sharding->Clear();
  tuple_op_sharding->set_type(xla::OpSharding::TUPLE);
  *tuple_op_sharding->add_tuple_shardings() = v3_op_sharding;
  serialized.set_data(proto.SerializeAsCord());

  EXPECT_THAT(
      Deserialize<HloSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(client())),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("XLA HloShardingV3 format is not supported for "
                         "deserialization")));
}

TEST_P(XlaShardingSerDesTest, RejectEmptyDeviceListOnDeserialize) {
  HloShardingProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());
  GetDevices({})->ToProto(*proto.mutable_devices(), version());
  *proto.mutable_xla_op_sharding() = xla::HloSharding::Replicate().ToProto();
  Serialized wrapper;
  wrapper.set_type_name("xla::ifrt::HloSharding");
  wrapper.set_data(proto.SerializeAsString());

  EXPECT_THAT(
      Deserialize<HloSharding>(
          wrapper, std::make_unique<DeserializeShardingOptions>(client())),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires at least 1 device")));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, XlaShardingSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/2, /*num_addressable_devices=*/2})),
    [](const testing::TestParamInfo<XlaShardingSerDesTestParam>& info) {
      return absl::StrCat("version_",
                          std::get<0>(info.param).version_number().value(),
                          "_num_devices_", std::get<1>(info.param).num_devices,
                          "_num_addressable_devices_",
                          std::get<1>(info.param).num_addressable_devices);
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
