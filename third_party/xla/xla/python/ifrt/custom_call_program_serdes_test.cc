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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/custom_call_program.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program_serdes.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::MatchesRegex;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

class CustomCallProgramSerDesTest : public test_util::ShardingTest {};

TEST_P(CustomCallProgramSerDesTest, RoundTrip) {
  Shape shape0({10, 20});
  Shape shard_shape0({5, 20});
  DeviceList devices = GetDevices({0, 1});
  std::shared_ptr<const Sharding> sharding0 =
      ConcreteEvenSharding::Create(devices, MemoryKind(),
                                   /*shape=*/shape0,
                                   /*shard_shape=*/shard_shape0);

  Shape shape1({});
  Shape shard_shape1({});
  std::shared_ptr<const Sharding> sharding1 =
      ConcreteEvenSharding::Create(devices, MemoryKind(),
                                   /*shape=*/shape1,
                                   /*shard_shape=*/shard_shape1);

  CustomCallProgram orig(
      /*type=*/"test type",
      /*name=*/"test name",
      /*serialized_program_text=*/"test\0program\0text\0",
      /*devices=*/std::move(devices),
      /*input_specs=*/
      {
          ArraySpec{/*dtype=*/DType(DType::kF32), /*shape=*/shape0,
                    /*sharding=*/sharding0},
      },
      /*output_specs=*/
      {
          ArraySpec{/*dtype=*/DType(DType::kF32), /*shape=*/shape1,
                    /*sharding=*/sharding1},
      });

  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(orig));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallProgram> deserialized_program,
      Deserialize<CustomCallProgram>(
          serialized, std::make_unique<DeserializeProgramOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_EQ(deserialized_program->type, "test type");
  EXPECT_EQ(deserialized_program->name, "test name");
  EXPECT_EQ(deserialized_program->serialized_program_text,
            "test\0program\0text\0");

  EXPECT_EQ(deserialized_program->devices, orig.devices);

  ASSERT_THAT(deserialized_program->input_specs, SizeIs(1));
  EXPECT_EQ(deserialized_program->input_specs.front().dtype,
            DType(DType::kF32));
  EXPECT_EQ(deserialized_program->input_specs.front().shape, shape0);
  const auto* deserialized_sharding0 = llvm::dyn_cast<ConcreteEvenSharding>(
      deserialized_program->input_specs.front().sharding.get());
  ASSERT_NE(deserialized_sharding0, nullptr);
  EXPECT_EQ(deserialized_sharding0->devices(), sharding0->devices());
  EXPECT_EQ(deserialized_sharding0->shape(), shape0);
  EXPECT_EQ(deserialized_sharding0->shard_shape(), shard_shape0);

  ASSERT_THAT(deserialized_program->output_specs, SizeIs(1));
  EXPECT_EQ(deserialized_program->output_specs.front().dtype,
            DType(DType::kF32));
  EXPECT_EQ(deserialized_program->output_specs.front().shape, shape1);
  const auto* deserialized_sharding1 = llvm::dyn_cast<ConcreteEvenSharding>(
      deserialized_program->output_specs.front().sharding.get());
  ASSERT_NE(deserialized_sharding1, nullptr);
  EXPECT_EQ(deserialized_sharding1->devices(), sharding1->devices());
  EXPECT_EQ(deserialized_sharding1->shape(), shape1);
  EXPECT_EQ(deserialized_sharding1->shard_shape(), shard_shape1);
}

INSTANTIATE_TEST_SUITE_P(NumDevices, CustomCallProgramSerDesTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/2,
                             /*num_addressable_devices=*/2}));

TEST(CustomCallCompileOptionsSerDesTest, RoundTrip) {
  CustomCallCompileOptions orig;
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(orig));
  TF_EXPECT_OK(
      Deserialize<CustomCallCompileOptions>(serialized, /*options=*/nullptr)
          .status());
}

TEST(CustomCallCompileOptionsSerDesTest, InvalidSerialized) {
  CustomCallCompileOptions orig;
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(orig));
  serialized.set_data("abc");
  EXPECT_THAT(
      Deserialize<CustomCallCompileOptions>(serialized, /*options=*/nullptr),
      StatusIs(absl::StatusCode::kInvalidArgument,
               MatchesRegex("Invalid serialized CustomCallCompileOptions.*")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
