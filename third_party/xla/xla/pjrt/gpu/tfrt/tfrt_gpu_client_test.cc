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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla {

class DonationTransactionPeer {
 public:
  static absl::StatusOr<TfrtGpuBuffer::DonationTransaction> AcquireDonation(
      TfrtGpuBuffer* tfrt_buffer) {
    return tfrt_buffer->AcquireDonation();
  }
  static tsl::AsyncValueRef<bool> GetDonationEvent(TfrtGpuBuffer* tfrt_buffer) {
    return tfrt_buffer->GetDonationEvent();
  }
};

namespace {

using ::testing::HasSubstr;
using ::testing::status::IsOkAndHolds;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.Compile(xla_computation, compile_options);
}

TEST(TfrtGpuClientTest, GpuClientOptions) {
  GpuClientOptions options;
  options.platform_name = "cuda";
  options.allowed_devices = {0, 1};
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(options));
  EXPECT_THAT(client->platform_version(), HasSubstr("cuda"));
  EXPECT_EQ(client->device_count(), 2);
}

TEST(TfrtGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_EQ(memory_space->kind(), TfrtGpuDeviceMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(), TfrtGpuDeviceMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(TfrtGpuDeviceMemorySpace::kKind),
                IsOkAndHolds(memory_space));

    EXPECT_EQ(device->memory_spaces().size(), 2);
    auto* pinned = device->memory_spaces()[1];
    EXPECT_EQ(pinned->kind_id(), PinnedHostMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(PinnedHostMemorySpace::kKind),
                IsOkAndHolds(pinned));
  }
}

TEST(TfrtGpuClientTest, MemorySpacesUniqueIds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  absl::flat_hash_map<int, std::string> memories;
  for (auto* device : client->devices()) {
    for (auto* memory_space : device->memory_spaces()) {
      std::string debug_string(memory_space->DebugString());
      auto [it, inserted] = memories.insert({memory_space->id(), debug_string});
      EXPECT_TRUE(inserted) << "Duplicate ids for memory spaces '" << it->second
                            << "' and '" << debug_string << "'";
    }
  }
}

TEST(StreamExecutorGpuClientTest, PropagateError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);
  absl::Status input_error = absl::InvalidArgumentError("input error");
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

ENTRY %Add.6 (a.1: f32[], b.2: f32[]) -> (f32[], f32[]) {
  %a.1 = f32[] parameter(0)
  %b.2 = f32[] parameter(1)
  %add.3 = f32[] add(f32[] %a.1, f32[] %b.2)
  %add.4 = f32[] add(f32[] %add.3, f32[] %add.3)
  ROOT %tuple.5 = (f32[], f32[]) tuple(f32[] %add.3, f32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  // TF_ASSERT_OK_AND_ASSIGN(
  //     auto result,
  //     executable->Execute({{buffer.get(), buffer.get()}}, /*options=*/{}));

  // ASSERT_EQ(result.size(), 1);
  // ASSERT_EQ(result[0].size(), 1);
  // EXPECT_EQ(result[0][0]->GetReadyFuture().Await(), input_error);
}

TEST(TfrtGpuClientTest, AcquireDonation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  // Create TfrtGpuBuffer.
  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      tsl::MakeAvailableAsyncValueRef<GpuEvent>());
  auto memory_space = device->default_memory_space().value();
  auto tfrt_buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  auto donation_transaction =
      DonationTransactionPeer::AcquireDonation(tfrt_buffer.get());
  EXPECT_TRUE(donation_transaction.ok());
  std::move(*donation_transaction).Commit();
  EXPECT_EQ(donation_transaction->device_buffer(), nullptr);
  EXPECT_TRUE(
      DonationTransactionPeer::GetDonationEvent(tfrt_buffer.get()).get());
}

}  // namespace
}  // namespace xla
