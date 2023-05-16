/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <numeric>
#include <optional>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::SizeIs;

TEST(ArrayImplTest, MakeArrayFromHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
}

class ArrayImplWithHostBufferSemanticsTest
    : public testing::TestWithParam<Client::HostBufferSemantics> {};

TEST_P(ArrayImplWithHostBufferSemanticsTest,
       MakeArrayFromHostBufferCallsWithOnDoneWithHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Client::HostBufferSemantics semantics = GetParam();

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  absl::Notification done_with_host_buffer;
  auto on_done_with_host_buffer = [&]() { done_with_host_buffer.Notify(); };

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      std::move(on_done_with_host_buffer)));

  // Regardless of the host buffer semantics chosen, the host buffer must not be
  // used by the runtime once `on_done_with_host_buffer` has been called.
  if (semantics == Client::HostBufferSemantics::kZeroCopy) {
    // `on_done_with_host_buffer` is called only when the `Array` is destroyed
    // if the runtime implements `kZeroCopy`. A deadlock will occur if we keep
    // the `Array` instance.
    array.reset();

    // `done_with_host_buffer` is very likely to have been called after
    // sleeping. This method has false positives (sleeping was not long enough
    // for the callback to be called asynchronously), but it may greatly
    // increases the chance of detecting an incorrect implementation as a form
    // of test flakes.
    absl::SleepFor(absl::Seconds(3));
    ASSERT_TRUE(done_with_host_buffer.HasBeenNotified());
  } else {
    done_with_host_buffer.WaitForNotification();
  }
  data = nullptr;
  // There should be no use-after-free.
}

INSTANTIATE_TEST_CASE_P(
    AllHostBufferSemantics, ArrayImplWithHostBufferSemanticsTest,
    testing::Values(
        Client::HostBufferSemantics::kImmutableOnlyDuringCall,
        Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
        Client::HostBufferSemantics::kZeroCopy));

TEST(ArrayImplTest, MakeArrayFromHostBufferImmutableOnlyDuringCall) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  absl::Notification done_with_host_buffer;
  auto on_done_with_host_buffer = [&]() {
    // Sleeping facilitates testing if `MakeArrayFromHostBuffer()` calls
    // `on_done_with_host_buffer` synchronously before returning. This method
    // has false negatives (when a call to
    // `done_with_host_buffer.HasBeenNotified()` is delayed), but it may greatly
    // increases the chance of detecting an incorrect implementation as a form
    // of test flakes.
    absl::SleepFor(absl::Seconds(3));

    done_with_host_buffer.Notify();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      std::move(on_done_with_host_buffer)));

  // `on_done_with_host_buffer` should have been called before returning from
  // `MakeArrayFromHostBuffer`.
  ASSERT_TRUE(done_with_host_buffer.HasBeenNotified());
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferImmutableUntilTransferCompletes) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      client->MakeArrayFromHostBuffer(
          data->data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding,
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr));

  // Once the `Array` has become ready, the host buffer is not accessed.
  TF_ASSERT_OK(array->GetReadyFuture().Await());
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferZeroCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      client->MakeArrayFromHostBuffer(data->data(), dtype, shape,
                                      /*byte_strides=*/std::nullopt, sharding,
                                      Client::HostBufferSemantics::kZeroCopy,
                                      /*on_done_with_host_buffer=*/nullptr));

  // The `Array` may alias the host buffer, but once the transfer is done and
  // the `Array` is destroyed, the host buffer is not accessed. This test would
  // pass trivially on the implementations that downgrade `kZeroCopy`, if
  // `MakeArrayFromHostBufferImmutableUntilTransferCompletes` already passes.
  TF_ASSERT_OK(array->GetReadyFuture().Await());
  array.reset();
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  auto future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferWithByteStridesAndCopyToHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  // The input data layout is minor-to-major.
  std::vector<float> data = {0, 3, 1, 4, 2, 5};
  std::vector<int64_t> byte_strides = {4, 8};
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape, byte_strides, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  // The expected output data layout is major-to-minor.
  std::vector<float> expected_out_data = {0, 1, 2, 3, 4, 5};
  auto future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBufferWithByteStrides) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  // The input data layout is major-to-minor.
  std::vector<float> data = {0, 1, 2, 3, 4, 5};
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  // The requested output data layout is minor-to-major.
  std::vector<int64_t> byte_strides = {4, 8};
  std::vector<float> expected_out_data = {0, 3, 1, 4, 2, 5};
  auto future = array->CopyToHostBuffer(out_data.data(), byte_strides,
                                        ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(ArrayImplTest, AssembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device0 = client->addressable_devices().at(0);
  auto sharding0 = SingleDeviceSharding::Create(device0);
  Device* device1 = client->addressable_devices().at(1);
  auto sharding1 = SingleDeviceSharding::Create(device1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array0, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding0,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto array1, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding1,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));

  std::vector<tsl::RCReference<Array>> arrays({array0, array1});
  Shape assembled_shape({4, 3});
  auto assembled_sharding = OpaqueSharding::Create(
      DeviceList(DeviceList::Devices({array0->sharding().devices().front(),
                                      array1->sharding().devices().front()})));
  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      client->AssembleArrayFromSingleDeviceArrays(
          assembled_shape, assembled_sharding, absl::MakeSpan(arrays),
          ArrayCopySemantics::kAlwaysCopy));

  EXPECT_EQ(assembled_array->dtype(), dtype);
  EXPECT_EQ(assembled_array->shape(), assembled_shape);
  EXPECT_EQ(assembled_array->shared_ptr_sharding().get(),
            assembled_sharding.get());
}

TEST(ArrayImplTest, AssembleAndDisassembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device0 = client->addressable_devices().at(0);
  auto sharding0 = SingleDeviceSharding::Create(device0);
  Device* device1 = client->addressable_devices().at(1);
  auto sharding1 = SingleDeviceSharding::Create(device1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array0, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding0,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto array1, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding1,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));

  std::vector<tsl::RCReference<Array>> arrays({array0, array1});
  std::vector<Shape> single_device_shapes({shape, shape});
  Shape assembled_shape({4, 3});
  ShardingParam sharding_param(
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}});
  auto ifrt_device_list =
      DeviceList(DeviceList::Devices({array0->sharding().devices().front(),
                                      array1->sharding().devices().front()}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> sharding_param_sharding,
      ShardingParamSharding::Create(std::move(sharding_param),
                                    ifrt_device_list));
  auto assembled_shardings = {
      OpaqueSharding::Create(
          ifrt_device_list,
          OpaqueSharding::MakeDisassembleFuncFromShapes(single_device_shapes)),
      sharding_param_sharding};
  for (auto& assembled_sharding : assembled_shardings) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto assembled_array,
        client->AssembleArrayFromSingleDeviceArrays(
            assembled_shape, assembled_sharding, absl::MakeSpan(arrays),
            ArrayCopySemantics::kAlwaysCopy));

    TF_ASSERT_OK_AND_ASSIGN(auto single_device_arrays,
                            assembled_array->DisassembleIntoSingleDeviceArrays(
                                ArrayCopySemantics::kAlwaysCopy));

    ASSERT_THAT(single_device_arrays, SizeIs(2));
    EXPECT_EQ(single_device_arrays[0]->dtype(), array0->dtype());
    EXPECT_EQ(single_device_arrays[0]->shape(), array0->shape());
    EXPECT_THAT(single_device_arrays[0]->sharding().devices().devices(),
                ElementsAreArray(array0->sharding().devices().devices()));
    EXPECT_EQ(single_device_arrays[1]->dtype(), array1->dtype());
    EXPECT_EQ(single_device_arrays[1]->shape(), array1->shape());
    EXPECT_THAT(single_device_arrays[1]->sharding().devices().devices(),
                ElementsAreArray(array1->sharding().devices().devices()));
  }
}

TEST(ArrayImplTest, ReshardToSameSharding) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto reshared_array,
      array->Reshard(sharding, ArrayCopySemantics::kAlwaysCopy));

  std::vector<float> out_data(6);
  auto future = reshared_array->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, ReshardToDifferentDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));

  Device* new_device = client->addressable_devices().at(1);
  auto new_sharding = SingleDeviceSharding::Create(new_device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto reshared_array,
      array->Reshard(sharding, ArrayCopySemantics::kAlwaysCopy));

  std::vector<float> out_data(6);
  auto future = reshared_array->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, GetReadyFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  TF_EXPECT_OK(array->GetReadyFuture().Await());
}

TEST(ArrayImplTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  TF_EXPECT_OK(array->Delete().Await());
}

TEST(ArrayImplTest, IsDeleted) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto sharding = SingleDeviceSharding::Create(device);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  EXPECT_FALSE(array->IsDeleted());
  auto future = array->Delete();
  EXPECT_TRUE(array->IsDeleted());
  TF_EXPECT_OK(future.Await());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
