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

#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

void CheckArrayContents(Array* lhs, Array* rhs) {
  ASSERT_EQ(lhs->dtype(), rhs->dtype());
  ASSERT_EQ(lhs->shape(), rhs->shape());

  ASSERT_TRUE(lhs->dtype().byte_size().has_value());
  int64_t expected_size =
      *lhs->dtype().byte_size() * lhs->shape().num_elements();

  std::vector<uint8_t> lhs_data(expected_size);
  std::vector<uint8_t> rhs_data(expected_size);

  tsl::Future<> lhs_future =
      lhs->CopyToHostBuffer(lhs_data.data(), /*byte_strides=*/std::nullopt,
                            ArrayCopySemantics::kAlwaysCopy);
  tsl::Future<> rhs_future =
      rhs->CopyToHostBuffer(rhs_data.data(), /*byte_strides=*/std::nullopt,
                            ArrayCopySemantics::kAlwaysCopy);

  ASSERT_OK(lhs_future.Await());
  ASSERT_OK(rhs_future.Await());
  EXPECT_THAT(lhs_data, ElementsAreArray(rhs_data));
}

TEST(BundleImplTest, Roundtrip) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_fill(data, 1.0f);

  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array1,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array2,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  std::vector<ValueRef> values = {array1, array2};
  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle,
      client->Bundle(absl::MakeSpan(values), ArrayCopySemantics::kReuseInput));

  EXPECT_EQ(bundle->num_values(), 2);

  ASSERT_OK_AND_ASSIGN(std::vector<ValueRef> retrieved_values,
                       bundle->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values.size(), 2);
  auto* retrieved_array0 = llvm::dyn_cast<Array>(retrieved_values[0].get());
  ASSERT_NE(retrieved_array0, nullptr);
  CheckArrayContents(retrieved_array0, array1.get());
  auto* retrieved_array1 = llvm::dyn_cast<Array>(retrieved_values[1].get());
  ASSERT_NE(retrieved_array1, nullptr);
  CheckArrayContents(retrieved_array1, array2.get());
}

TEST(BundleImplTest, ConcatBundles) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_fill(data, 1.0f);

  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array1,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array2,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  std::vector<ValueRef> values1 = {array1};
  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle1,
      client->Bundle(absl::MakeSpan(values1), ArrayCopySemantics::kReuseInput));

  std::vector<ValueRef> values2 = {array2};
  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle2,
      client->Bundle(absl::MakeSpan(values2), ArrayCopySemantics::kReuseInput));

  std::vector<BundleRef> bundles = {bundle1, bundle2};

  ASSERT_OK_AND_ASSIGN(BundleRef concat_bundle,
                       client->ConcatBundles(absl::MakeSpan(bundles),
                                             ArrayCopySemantics::kReuseInput));

  EXPECT_EQ(concat_bundle->num_values(), 2);

  ASSERT_OK_AND_ASSIGN(
      std::vector<ValueRef> retrieved_values,
      concat_bundle->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values.size(), 2);
  auto* retrieved_array0 = llvm::dyn_cast<Array>(retrieved_values[0].get());
  ASSERT_NE(retrieved_array0, nullptr);
  CheckArrayContents(retrieved_array0, array1.get());
  auto* retrieved_array1 = llvm::dyn_cast<Array>(retrieved_values[1].get());
  ASSERT_NE(retrieved_array1, nullptr);
  CheckArrayContents(retrieved_array1, array2.get());
}

TEST(BundleImplTest, Slice) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_fill(data, 1.0f);

  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array1,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array2,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array3,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  std::vector<ValueRef> values = {array1, array2, array3};
  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle,
      client->Bundle(absl::MakeSpan(values), ArrayCopySemantics::kReuseInput));

  ASSERT_OK_AND_ASSIGN(std::vector<BundleRef> slices,
                       bundle->Slice({1, 2}, ArrayCopySemantics::kReuseInput));

  ASSERT_EQ(slices.size(), 2);
  EXPECT_EQ(slices[0]->num_values(), 1);
  EXPECT_EQ(slices[1]->num_values(), 2);

  ASSERT_OK_AND_ASSIGN(std::vector<ValueRef> retrieved_values0,
                       slices[0]->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values0.size(), 1);
  auto* retrieved_array0 = llvm::dyn_cast<Array>(retrieved_values0[0].get());
  ASSERT_NE(retrieved_array0, nullptr);
  CheckArrayContents(retrieved_array0, array1.get());

  ASSERT_OK_AND_ASSIGN(std::vector<ValueRef> retrieved_values1,
                       slices[1]->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values1.size(), 2);
  auto* retrieved_array1_0 = llvm::dyn_cast<Array>(retrieved_values1[0].get());
  ASSERT_NE(retrieved_array1_0, nullptr);
  CheckArrayContents(retrieved_array1_0, array2.get());
  auto* retrieved_array1_1 = llvm::dyn_cast<Array>(retrieved_values1[1].get());
  ASSERT_NE(retrieved_array1_1, nullptr);
  CheckArrayContents(retrieved_array1_1, array3.get());
}

TEST(BundleImplTest, Alias) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  std::vector<ValueRef> values;
  values.reserve(10);
  for (int i = 0; i < 10; ++i) {
    DType dtype(DType::kF32);
    Shape shape({2, 3});
    std::vector<float> data(6);
    absl::c_fill(data, 1.0f);

    Device* device = client->addressable_devices().at(0);
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

    ASSERT_OK_AND_ASSIGN(
        values.emplace_back(),
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape,
            /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
  }

  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle,
      client->Bundle(absl::MakeSpan(values), ArrayCopySemantics::kReuseInput));

  EXPECT_EQ(bundle->num_values(), 10);

  values.resize(5);

  bundle = {};

  for (const auto& value : values) {
    EXPECT_FALSE(value->IsDeleted());
  }
}

TEST(BundleImplTest, CopyArrays) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_fill(data, 1.0f);

  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  std::vector<ValueRef> values = {array};

  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle,
      client->Bundle(absl::MakeSpan(values), ArrayCopySemantics::kReuseInput));

  Bundle::CopySpec spec;
  ASSERT_OK_AND_ASSIGN(spec.devices, client->MakeDeviceList({device}));
  std::vector<Bundle::CopySpec> specs = {spec};

  ASSERT_OK_AND_ASSIGN(
      BundleRef copied_bundle,
      bundle->CopyArrays({1}, specs, ArrayCopySemantics::kReuseInput));

  EXPECT_EQ(copied_bundle->num_values(), 1);

  ASSERT_OK_AND_ASSIGN(
      std::vector<ValueRef> retrieved_values,
      copied_bundle->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values.size(), 1);
  auto* copied_array = llvm::dyn_cast<Array>(retrieved_values[0].get());
  ASSERT_NE(copied_array, nullptr);
  EXPECT_EQ(copied_array->dtype(), dtype);
  EXPECT_EQ(copied_array->shape(), shape);
}

TEST(BundleImplTest, ReshardArrays) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_fill(data, 1.0f);

  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  ASSERT_OK_AND_ASSIGN(
      ArrayRef array,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  std::vector<ValueRef> values = {array};

  ASSERT_OK_AND_ASSIGN(
      BundleRef bundle,
      client->Bundle(absl::MakeSpan(values), ArrayCopySemantics::kReuseInput));

  std::vector<ArraySpec> specs = {{
      /*dtype=*/dtype,
      /*shape=*/shape,
      /*sharding=*/sharding,
      /*layout=*/nullptr,
  }};

  ASSERT_OK_AND_ASSIGN(
      BundleRef resharded_bundle,
      bundle->ReshardArrays(specs, ArrayCopySemantics::kReuseInput));
  EXPECT_EQ(resharded_bundle->num_values(), 1);

  ASSERT_OK_AND_ASSIGN(
      std::vector<ValueRef> retrieved_values,
      resharded_bundle->GetValues(ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(retrieved_values.size(), 1);
  auto* resharded_array = llvm::dyn_cast<Array>(retrieved_values[0].get());
  ASSERT_NE(resharded_array, nullptr);
  EXPECT_EQ(resharded_array->dtype(), dtype);
  EXPECT_EQ(resharded_array->shape(), shape);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
