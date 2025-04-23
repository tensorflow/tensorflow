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

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

// Returns a shape for an array whose first dimension is fully sharded across
// `num_shards` devices. For example, [2, 3] with num_shards=5 becomes [10, 3].
absl::StatusOr<Shape> GetShape(int64_t num_shards, Shape shard_shape) {
  TF_RET_CHECK(!shard_shape.dims().empty());
  Shape::Dimensions dims(shard_shape.dims().begin(), shard_shape.dims().end());
  dims.front() *= num_shards;
  return Shape(std::move(dims));
}

// Returns an array spec that expresses an array whose first dimension is fully
// sharded across `device_indices`. For example, [2, 3] with
// device_indices.size()=5 becomes [10, 3].
absl::StatusOr<ArraySpec> CreateArraySpec(Client* client,
                                          absl::Span<const int> device_indices,
                                          Shape shard_shape = Shape({2, 3}),
                                          DType dtype = DType(DType::kS32)) {
  TF_ASSIGN_OR_RETURN(DeviceListRef device_list,
                      test_util::GetAddressableDevices(client, device_indices));
  TF_ASSIGN_OR_RETURN(Shape shape,
                      GetShape(device_indices.size(), shard_shape));
  return ArraySpec{/*dtype=*/dtype,
                   /*shape=*/shape,
                   /*sharding=*/
                   ConcreteEvenSharding::Create(device_list, MemoryKind(),
                                                shape, shard_shape)};
}

// Creates an array with (base_values.size()) shards. Each shard of the
// constructed array has a shape of `shard_shape` and it is fully sharded on
// first dimension. For example, if shard_shape=[2, 3] then the array shaps is
// [2 * base_values.size(), 3]. The contents of shard i is reshaped from an iota
// starting from `base_values[i]` and placed on addressable device
// `device_indices[i]`.

template <typename ValueType>
struct CppTypeToDType;

template <>
struct CppTypeToDType<int32_t> {
  static constexpr DType::Kind kDType = DType::kS32;
};

template <>
struct CppTypeToDType<float> {
  static constexpr DType::Kind kDType = DType::kF32;
};

template <typename ValueType>
absl::StatusOr<tsl::RCReference<Array>> CreateArray(
    Client* client, absl::Span<const ValueType> base_values,
    absl::Span<const int> device_indices, Shape shard_shape = Shape({2, 3})) {
  TF_RET_CHECK(base_values.size() == device_indices.size());

  DType dtype(CppTypeToDType<ValueType>::kDType);
  TF_ASSIGN_OR_RETURN(Shape shape, GetShape(base_values.size(), shard_shape));

  std::vector<tsl::RCReference<Array>> shards;
  shards.reserve(base_values.size());
  absl::InlinedVector<xla::ifrt::Device*, 1> devices;
  devices.reserve(device_indices.size());

  for (int i = 0; i < base_values.size(); ++i) {
    std::vector<ValueType> data(shard_shape.num_elements());
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
      ConcreteEvenSharding::Create(client->MakeDeviceList(devices),
                                   MemoryKind(),
                                   /*shape=*/shape,
                                   /*shard_shape=*/std::move(shard_shape));
  absl::Span<tsl::RCReference<Array>> arrays = absl::MakeSpan(shards);
  return client->AssembleArrayFromSingleDeviceArrays(
      arrays.at(0)->dtype(), std::move(shape), std::move(assembled_sharding),
      arrays, ArrayCopySemantics::kDonateInput,
      SingleDeviceShardSemantics::kAddressableShards);
}

// Checks the shards and contents of an array, same as what CreateArray would
// generate given the same arguments.
template <typename ValueType>
void AssertArrayContent(Client* client, Array* array,
                        absl::Span<const ValueType> base_values,
                        absl::Span<const int> device_indices,
                        Shape expected_shard_shape = Shape({2, 3})) {
  DType expected_dtype(CppTypeToDType<ValueType>::kDType);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape,
                          GetShape(base_values.size(), expected_shard_shape));
  EXPECT_EQ(array->dtype(), expected_dtype);
  EXPECT_EQ(array->shape(), expected_shape);
  const auto* actual_sharding =
      llvm::dyn_cast<ConcreteEvenSharding>(array->shared_ptr_sharding().get());
  ASSERT_NE(actual_sharding, nullptr);
  EXPECT_EQ(actual_sharding->shape(), expected_shape);
  EXPECT_EQ(actual_sharding->shard_shape(), expected_shard_shape);

  TF_ASSERT_OK_AND_ASSIGN(auto shards,
                          array->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kReuseInput,
                              SingleDeviceShardSemantics::kAddressableShards));
  ASSERT_THAT(shards, SizeIs(base_values.size()));
  for (int i = 0; i < shards.size(); ++i) {
    EXPECT_EQ(shards[i]->dtype(), expected_dtype);
    EXPECT_EQ(shards[i]->shape(), expected_shard_shape);
    const auto* actual_shard_sharding = llvm::dyn_cast<SingleDeviceSharding>(
        shards[i]->shared_ptr_sharding().get());
    ASSERT_NE(actual_shard_sharding, nullptr);
    Device* expected_device =
        client->addressable_devices().at(device_indices[i]);
    EXPECT_THAT(actual_shard_sharding->devices()->devices(),
                ElementsAre(expected_device));

    std::vector<ValueType> expected_data(expected_shard_shape.num_elements());
    std::iota(expected_data.begin(), expected_data.end(), base_values[i]);

    std::vector<ValueType> actual_data(shards[i]->shape().num_elements());
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
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{1}).value());
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
      CreateArray<int32_t>(client.get(), /*base_values=*/{0, 6},
                           /*device_indices=*/{0, 1}));

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kReuseInput));
    ASSERT_THAT(out_arrays, SizeIs(1));
    // `out_arrays[0].shards[0] == arrays[0].shards[1]`.
    AssertArrayContent<int32_t>(client.get(), out_arrays[0].get(),
                                /*base_values=*/{6},
                                /*device_indices=*/{1});
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                             ArrayCopySemantics::kDonateInput));
    ASSERT_THAT(out_arrays, SizeIs(1));
    // `out_arrays[0].shards[0] == arrays[0].shards[1]`.
    AssertArrayContent<int32_t>(client.get(), out_arrays[0].get(),
                                /*base_values=*/{6},
                                /*device_indices=*/{1});
  }
}

TEST(RemapImplTest, InterleaveArraysDonate) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1}).value());
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{2, 3}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 2, 1, 3}).value());
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
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{0, 6},
                           /*device_indices=*/{0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{100, 106},
                           /*device_indices=*/{2, 3}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_arrays, client->RemapArrays(plan, absl::MakeSpan(arrays),
                                           ArrayCopySemantics::kDonateInput));

  ASSERT_THAT(out_arrays, SizeIs(1));
  // `out_arrays[0].shards[0] == arrays[0].shards[0]`
  // `out_arrays[0].shards[1] == arrays[1].shards[0]`
  // `out_arrays[0].shards[2] == arrays[0].shards[1]`
  // `out_arrays[0].shards[3] == arrays[1].shards[1]`
  AssertArrayContent<int32_t>(client.get(), out_arrays[0].get(),
                              /*base_values=*/{0, 100, 6, 106},
                              /*device_indices=*/{0, 2, 1, 3});
}

TEST(RemapImplTest, InterleaveArraysReuse) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1}).value());
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{2, 3}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 2, 1, 3}).value());
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
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{0, 6},
                           /*device_indices=*/{0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{100, 106},
                           /*device_indices=*/{2, 3}));

  EXPECT_THAT(client->RemapArrays(plan, absl::MakeSpan(arrays),
                                  ArrayCopySemantics::kReuseInput),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("kDonateInput is required if multiple inputs "
                                 "are mapped to one output")));
}

TEST(RemapImplTest, DeinterleaveArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  RemapPlan plan;
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 2, 1, 3}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{2, 3}).value());
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
      CreateArray<int32_t>(client.get(), /*base_values=*/{0, 100, 6, 106},
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
    AssertArrayContent<int32_t>(client.get(), out_arrays[0].get(),
                                /*base_values=*/{0, 6},
                                /*device_indices=*/{0, 1});
    AssertArrayContent<int32_t>(client.get(), out_arrays[1].get(),
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
    AssertArrayContent<int32_t>(client.get(), out_arrays[0].get(),
                                /*base_values=*/{0, 6},
                                /*device_indices=*/{0, 1});
    AssertArrayContent<int32_t>(client.get(), out_arrays[1].get(),
                                /*base_values=*/{100, 106},
                                /*device_indices=*/{2, 3});
  }
}

TEST(RemapImplTest, BatchMappingIdentity) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client,
                          test_util::GetClient());
  Shape first_shard_shape({2, 3});
  Shape second_shard_shape({3, 5});
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec all_device_spec,
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1, 2, 3},
                      first_shard_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec first_two_device_spec,
      CreateArraySpec(client.get(), /*device_indices=*/{0, 1},
                      second_shard_shape));

  RemapPlan plan;
  plan.input_specs.push_back(all_device_spec);
  plan.input_specs.push_back(first_two_device_spec);
  plan.output_specs.push_back(all_device_spec);
  plan.output_specs.push_back(first_two_device_spec);
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 4, 1}},
                         /*to=*/{RemapPlan::Interval{0, 4, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,
                         /*out_array=*/1,
                         /*from=*/{RemapPlan::Interval{0, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  TF_ASSERT_OK(plan.Validate());

  std::vector<tsl::RCReference<Array>> inputs;
  TF_ASSERT_OK_AND_ASSIGN(
      inputs.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{10, 20, 30, 40},
                           /*device_indices=*/{0, 1, 2, 3}, first_shard_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      inputs.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{50, 60},
                           /*device_indices=*/{0, 1}, second_shard_shape));
  for (ArrayCopySemantics copy_semantics : std::vector<ArrayCopySemantics>{
           ArrayCopySemantics::kReuseInput, ArrayCopySemantics::kDonateInput}) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<tsl::RCReference<Array>> outputs,
        client->RemapArrays(plan, absl::MakeSpan(inputs), copy_semantics));
    ASSERT_THAT(outputs, SizeIs(2));
    AssertArrayContent<int32_t>(client.get(), outputs[0].get(),
                                /*base_values=*/{10, 20, 30, 40},
                                /*device_indices=*/{0, 1, 2, 3},
                                first_shard_shape);
    AssertArrayContent<int32_t>(client.get(), outputs[1].get(),
                                /*base_values=*/{50, 60},
                                /*device_indices=*/{0, 1}, second_shard_shape);
  }
}

// For a specific output, kDonateInput allows mapping multiple inputs to this
// output, whereas kReuseInput does not. See CheckArrayCopySemantics. As such,
// only test DeinterleaveArrays situation, not InterleaveArrays.
TEST(RemapImplTest, BatchMappingDeinterleave) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client,
                          test_util::GetClient());
  Shape first_shard_shape({2, 3});
  Shape second_shard_shape({3, 5});
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec first_input_spec,
      CreateArraySpec(client.get(), {0, 1, 2, 3}, first_shard_shape,
                      DType(DType::kF32)));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec first_output_spec_one,
      CreateArraySpec(client.get(), {0, 1}, first_shard_shape,
                      DType(DType::kF32)));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec first_output_spec_two,
      CreateArraySpec(client.get(), {2, 3}, first_shard_shape,
                      DType(DType::kF32)));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec second_input_spec,
      CreateArraySpec(client.get(), {0, 1}, second_shard_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec second_output_spec_one,
      CreateArraySpec(client.get(), {0}, second_shard_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      ArraySpec second_output_spec_two,
      CreateArraySpec(client.get(), {1}, second_shard_shape));

  RemapPlan plan;
  plan.input_specs.push_back(first_input_spec);
  plan.input_specs.push_back(second_input_spec);
  plan.output_specs.push_back(first_output_spec_one);
  plan.output_specs.push_back(first_output_spec_two);
  plan.output_specs.push_back(second_output_spec_one);
  plan.output_specs.push_back(second_output_spec_two);
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/1,
                         /*from=*/{RemapPlan::Interval{2, 4, 1}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,
                         /*out_array=*/2,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,
                         /*out_array=*/3,
                         /*from=*/{RemapPlan::Interval{1, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  TF_ASSERT_OK(plan.Validate());

  std::vector<tsl::RCReference<Array>> inputs;
  TF_ASSERT_OK_AND_ASSIGN(
      inputs.emplace_back(),
      CreateArray<float>(client.get(), /*base_values=*/{10, 20, 30, 40},
                         /*device_indices=*/{0, 1, 2, 3}, first_shard_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      inputs.emplace_back(),
      CreateArray<int32_t>(client.get(), /*base_values=*/{50, 60},
                           /*device_indices=*/{0, 1}, second_shard_shape));
  for (ArrayCopySemantics copy_semantics : std::vector<ArrayCopySemantics>{
           ArrayCopySemantics::kReuseInput, ArrayCopySemantics::kDonateInput}) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<tsl::RCReference<Array>> outputs,
        client->RemapArrays(plan, absl::MakeSpan(inputs), copy_semantics));
    ASSERT_THAT(outputs, SizeIs(4));
    AssertArrayContent<float>(client.get(), outputs[0].get(),
                              /*base_values=*/{10, 20},
                              /*device_indices=*/{0, 1}, first_shard_shape);
    AssertArrayContent<float>(client.get(), outputs[1].get(),
                              /*base_values=*/{30, 40},
                              /*device_indices=*/{2, 3}, first_shard_shape);
    AssertArrayContent<int32_t>(client.get(), outputs[2].get(),
                                /*base_values=*/{50},
                                /*device_indices=*/{0}, second_shard_shape);
    AssertArrayContent<int32_t>(client.get(), outputs[3].get(),
                                /*base_values=*/{60},
                                /*device_indices=*/{1}, second_shard_shape);
  }
}

TEST(RemapImplTest, DetectBadInput) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  // Trivial remap plan for a single device array on device 0.
  RemapPlan plan;
  plan.input_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0}).value());
  plan.output_specs.push_back(
      CreateArraySpec(client.get(), /*device_indices=*/{0}).value());
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  TF_ASSERT_OK(plan.Validate());

  {
    std::vector<tsl::RCReference<Array>> arrays;
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        CreateArray<int32_t>(client.get(), /*base_values=*/{0},
                             /*device_indices=*/{0}));
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        CreateArray<int32_t>(client.get(), /*base_values=*/{0},
                             /*device_indices=*/{0}));
    EXPECT_THAT(
        client->RemapArrays(plan, absl::MakeSpan(arrays),
                            ArrayCopySemantics::kReuseInput),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("RemapArrays expects 1 input arrays, but got 2")));
  }

  {
    std::vector<tsl::RCReference<Array>> arrays;
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        CreateArray<float>(client.get(), /*base_values=*/{0},
                           /*device_indices=*/{0}));
    EXPECT_THAT(
        client->RemapArrays(plan, absl::MakeSpan(arrays),
                            ArrayCopySemantics::kReuseInput),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("RemapArrays expects input #0 to have dtype")));
  }

  {
    std::vector<tsl::RCReference<Array>> arrays;
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        CreateArray<int32_t>(client.get(), /*base_values=*/{0},
                             /*device_indices=*/{0},
                             /*shard_shape=*/Shape({20, 30})));
    EXPECT_THAT(
        client->RemapArrays(plan, absl::MakeSpan(arrays),
                            ArrayCopySemantics::kReuseInput),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("RemapArrays expects input #0 to have shape")));
  }

  {
    std::vector<tsl::RCReference<Array>> arrays;
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        CreateArray<int32_t>(client.get(), /*base_values=*/{0},
                             /*device_indices=*/{1}));
    EXPECT_THAT(client->RemapArrays(plan, absl::MakeSpan(arrays),
                                    ArrayCopySemantics::kReuseInput),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RemapArrays expects input #0 to be on")));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
