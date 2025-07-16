/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/array_spec.h"

#include <memory>
#include <tuple>

#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ArraySpecTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class ArraySpecTest : public testing::TestWithParam<ArraySpecTestParam> {
 public:
  ArraySpecTest()
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

TEST_P(ArraySpecTest, SupportsAbslHash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      ArraySpec{DType(DType::kS32), Shape({4, 2}),
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 2}),
                                             /*shard_shape=*/Shape({2, 2}))},
      ArraySpec{DType(DType::kS32), Shape({4, 2}),
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 2}),
                                             /*shard_shape=*/Shape({2, 2})),
                std::make_shared<xla::PjRtLayout>(
                    xla::LayoutUtil::MakeDescendingLayout(2))},
  }));
}

TEST_P(ArraySpecTest, ToFromProto) {
  auto device_list = GetDevices({0, 1});
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  Shape shard_shape({2, 2});
  ArraySpec spec{/*dtype=*/dtype, /*shape=*/shape,
                 /*sharding=*/
                 ConcreteEvenSharding::Create(device_list, MemoryKind(),
                                              /*shape=*/shape,
                                              /*shard_shape=*/shard_shape)};

  TF_ASSERT_OK_AND_ASSIGN(const ArraySpecProto proto, spec.ToProto(version()));
  TF_ASSERT_OK_AND_ASSIGN(const ArraySpec array_spec_copy,
                          ArraySpec::FromProto(client(), proto));

  EXPECT_EQ(array_spec_copy.dtype, dtype);
  EXPECT_EQ(array_spec_copy.shape, shape);

  const auto* sharding =
      llvm::dyn_cast<ConcreteEvenSharding>(array_spec_copy.sharding.get());
  ASSERT_NE(sharding, nullptr);
  EXPECT_EQ(*sharding->devices(), *spec.sharding->devices());
  EXPECT_EQ(sharding->memory_kind(), spec.sharding->memory_kind());
  EXPECT_EQ(sharding->shape(), shape);
  EXPECT_EQ(sharding->shard_shape(), shard_shape);
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, ArraySpecTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/2,
                         /*num_addressable_devices=*/2})));

}  // namespace
}  // namespace ifrt
}  // namespace xla
