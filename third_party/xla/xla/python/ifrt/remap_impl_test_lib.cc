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

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

// Creates an array with (base_values.size()) shards. The constructed array
// shape is [2 * base_values.size(), 3]. Each shard has a shape of [2, 3] and
// has a content reshaped from an iota starting from `base_values[i]` for shard
// i. Shard i is placed on addressable device (device_indices[i]).
absl::StatusOr<tsl::RCReference<Array>> CreateArray(
    Client* client, absl::Span<const int32_t> base_values,
    absl::Span<const int> device_indices) {
  TF_RET_CHECK(base_values.size() == device_indices.size());

  DType dtype(DType::kS32);
  Shape shape({2 * static_cast<int64_t>(base_values.size()), 3});
  Shape shard_shape({2, 3});

  std::vector<tsl::RCReference<Array>> shards;
  shards.reserve(base_values.size());
  DeviceList::Devices devices;
  devices.reserve(device_indices.size());

  for (int i = 0; i < base_values.size(); ++i) {
    std::vector<int32_t> data(6);
    std::iota(data.begin(), data.end(), base_values[i]);

    Device* device = client->addressable_devices().at(device_indices[i]);
    devices.push_back(device);
    std::shared_ptr<const Sharding> sharding =
        SingleDeviceSharding::Create(device, MemoryKind());

    TF_ASSIGN_OR_RETURN(
        shards.emplace_back(),
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shard_shape,
            /*byte_strides=*/std::nullopt, std::move(sharding),
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));
  }

  std::shared_ptr<const Sharding> assembled_sharding =
      ConcreteEvenSharding::Create(DeviceList(std::move(devices)), MemoryKind(),
                                   /*shape=*/shape,
                                   /*shard_shape=*/std::move(shard_shape));
  return client->AssembleArrayFromSingleDeviceArrays(
      std::move(shape), std::move(assembled_sharding), absl::MakeSpan(shards),
      ArrayCopySemantics::kDonateInput);
}

// Checks the shards of an array. The expected array shape is [2 *
// base_values.size(), 3]. Each shard has an expected shape of [2, 3], whose
// content is an iota starting from `base_values[i]` for shard i. Shard i is
// expected to be placed on addressable device (device_indices[i]).
void AssertArrayContent(Client* client, Array* array,
                        absl::Span<const int32_t> base_values,
                        absl::Span<const int> device_indices) {
  DType expected_dtype(DType::kS32);
  Shape expected_shape({2 * static_cast<int64_t>(base_values.size()), 3});
  Shape expected_shard_shape({2, 3});
  EXPECT_EQ(array->dtype(), expected_dtype);
  EXPECT_EQ(array->shape(), expected_shape);
  const auto* actual_sharding =
      llvm::dyn_cast<ConcreteEvenSharding>(array->shared_ptr_sharding().get());
  ASSERT_NE(actual_sharding, nullptr);
  EXPECT_EQ(actual_sharding->shape(), expected_shape);
  EXPECT_EQ(actual_sharding->shard_shape(), expected_shard_shape);

  TF_ASSERT_OK_AND_ASSIGN(auto shards, array->DisassembleIntoSingleDeviceArrays(
                                           ArrayCopySemantics::kReuseInput));
  ASSERT_THAT(shards, SizeIs(base_values.size()));
  for (int i = 0; i < shards.size(); ++i) {
    EXPECT_EQ(shards[i]->dtype(), expected_dtype);
    EXPECT_EQ(shards[i]->shape(), expected_shard_shape);
    const auto* actual_shard_sharding = llvm::dyn_cast<SingleDeviceSharding>(
        shards[i]->shared_ptr_sharding().get());
    ASSERT_NE(actual_shard_sharding, nullptr);
    Device* expected_device =
        client->addressable_devices().at(device_indices[i]);
    EXPECT_THAT(actual_shard_sharding->devices(), ElementsAre(expected_device));

    std::vector<int32_t> expected_data(6);
    std::iota(expected_data.begin(), expected_data.end(), base_values[i]);

    std::vector<int32_t> actual_data(6);
    TF_ASSERT_OK(shards[i]
                     ->CopyToHostBuffer(actual_data.data(),
                                        /*byte_strides=*/std::nullopt,
                                        ArrayCopySemantics::kAlwaysCopy)
                     .Await());
    EXPECT_THAT(actual_data, ElementsAreArray(expected_data));
  }
};

TEST(RemapImplTest, ExtractSingleShard) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({8, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {0, 1, 2, 3}).value(),
                    MemoryKind(), /*shape=*/Shape({8, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {1}).value(),
                    MemoryKind(), /*shape=*/Shape({2, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  // arrays[0].shards[1:2:1] is mapped into out_arrays[0].shards[0:1:1].
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{1, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  TF_ASSERT_OK(plan.Validate());

  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray(client.get(), /*base_values=*/{0, 6, 100, 106},
                  /*device_indices=*/{0, 1, 2, 3}));

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kReuseInput));
    ASSERT_THAT(out_arrays, SizeIs(1));
    // `out_arrays[0].shards[0] == arrays[0].shards[1]`.
    AssertArrayContent(client.get(), out_arrays[0].get(), /*base_values=*/{6},
                       /*device_indices=*/{1});
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kDonateInput));
    ASSERT_THAT(out_arrays, SizeIs(1));
    // `out_arrays[0].shards[0] == arrays[0].shards[1]`.
    AssertArrayContent(client.get(), out_arrays[0].get(), /*base_values=*/{6},
                       /*device_indices=*/{1});
  }
}

TEST(RemapImplTest, InterleaveArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {0, 1}).value(),
                    MemoryKind(), /*shape=*/Shape({4, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {2, 3}).value(),
                    MemoryKind(), /*shape=*/Shape({4, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({8, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {0, 2, 1, 3}).value(),
                    MemoryKind(), /*shape=*/Shape({8, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  // arrays[0].shards[0:2:1] is mapped into out_arrays[0].shards[0:4:2].
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->reserve(2);
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 4, 2}}});
  // arrays[1].shards[0:2:1] is mapped into out_arrays[0].shards[1:4:2].
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1, /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 2, 1}},
                         /*to=*/{RemapPlan::Interval{1, 4, 2}}});
  TF_ASSERT_OK(plan.Validate());

  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(arrays.emplace_back(),
                          CreateArray(client.get(), /*base_values=*/{0, 6},
                                      /*device_indices=*/{0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(arrays.emplace_back(),
                          CreateArray(client.get(), /*base_values=*/{100, 106},
                                      /*device_indices=*/{2, 3}));

  EXPECT_THAT(
      client->RemapArrays(plan, absl::MakeSpan(arrays),
                          ArrayCopySemantics::kReuseInput),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("kDonateInput is required if multiple inputs are used")));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                           ArrayCopySemantics::kDonateInput));

  ASSERT_THAT(out_arrays, SizeIs(1));
  // `out_arrays[0].shards[0] == arrays[0].shards[0]`
  // `out_arrays[0].shards[1] == arrays[1].shards[0]`
  // `out_arrays[0].shards[2] == arrays[0].shards[1]`
  // `out_arrays[0].shards[3] == arrays[1].shards[1]`
  AssertArrayContent(client.get(), out_arrays[0].get(),
                     /*base_values=*/{0, 100, 6, 106},
                     /*device_indices=*/{0, 2, 1, 3});
}

TEST(RemapImplTest, DeinterleaveArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({8, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {0, 2, 1, 3}).value(),
                    MemoryKind(), /*shape=*/Shape({8, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {0, 1}).value(),
                    MemoryKind(), /*shape=*/Shape({4, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(
                    test_util::GetDevices(client.get(), {2, 3}).value(),
                    MemoryKind(), /*shape=*/Shape({4, 3}),
                    /*shard_shape=*/Shape({2, 3}))});
  // arrays[0].shards[0:4:2] is mapped into out_arrays[0].shards[0:2:1].
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->reserve(2);
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 4, 2}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  // arrays[0].shards[1:4:2] is mapped into out_arrays[1].shards[0:2:1].
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/1,
                         /*from=*/{RemapPlan::Interval{1, 4, 2}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  TF_ASSERT_OK(plan.Validate());

  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray(client.get(), /*base_values=*/{0, 100, 6, 106},
                  /*device_indices=*/{0, 2, 1, 3}));

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kReuseInput));

    ASSERT_THAT(out_arrays, SizeIs(2));
    // `out_arrays[0].shards[0] == arrays[0].shards[0]`
    // `out_arrays[0].shards[1] == arrays[0].shards[2]`
    // `out_arrays[1].shards[0] == arrays[0].shards[1]`
    // `out_arrays[1].shards[1] == arrays[0].shards[3]`
    AssertArrayContent(client.get(), out_arrays[0].get(),
                       /*base_values=*/{0, 6},
                       /*device_indices=*/{0, 1});
    AssertArrayContent(client.get(), out_arrays[1].get(),
                       /*base_values=*/{100, 106},
                       /*device_indices=*/{2, 3});
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kDonateInput));

    ASSERT_THAT(out_arrays, SizeIs(2));
    // `out_arrays[0].shards[0] == arrays[0].shards[0]`
    // `out_arrays[0].shards[1] == arrays[0].shards[2]`
    // `out_arrays[1].shards[0] == arrays[0].shards[1]`
    // `out_arrays[1].shards[1] == arrays[0].shards[3]`
    AssertArrayContent(client.get(), out_arrays[0].get(),
                       /*base_values=*/{0, 6},
                       /*device_indices=*/{0, 1});
    AssertArrayContent(client.get(), out_arrays[1].get(),
                       /*base_values=*/{100, 106},
                       /*device_indices=*/{2, 3});
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
