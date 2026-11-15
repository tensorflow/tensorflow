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

#include "xla/pjrt/raw_buffer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/device_event.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"

namespace xla {

std::optional<absl::StatusOr<PjRtRawBufferRef>> MockFactory(
    PjRtBuffer* buffer) {
  if (buffer == nullptr) {
    return std::nullopt;
  }
  return absl::InvalidArgumentError("MockFactory");
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(MockFactory);

TEST(RawBufferTest, FactoryFallback) {
  auto status = PjRtRawBuffer::CreateRawAliasOfBuffer(nullptr).status();
  ASSERT_THAT(status,
              absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                                     testing::HasSubstr("null buffer.")));
}

TEST(RawBufferTest, FactoryError) {
  int dummy;
  auto status = PjRtRawBuffer::CreateRawAliasOfBuffer(
                    reinterpret_cast<PjRtBuffer*>(&dummy))
                    .status();
  ASSERT_THAT(status,
              absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                                     testing::HasSubstr("MockFactory")));
}

// A copy of the built-in raw buffer function table at a different address,
// whose accesses also carry copied function tables. To the client side this
// looks like a plugin from another binary, so it takes the C API path instead
// of recognizing its own async values.
const PJRT_RawBuffer_FunctionTable* ForeignRawBufferVtable(
    const PJRT_RawBuffer_FunctionTable* builtin) {
  static const PJRT_RawBuffer_FunctionTable* const kBuiltin = builtin;
  static const PJRT_RawBuffer_FunctionTable kForeign = [] {
    PJRT_RawBuffer_FunctionTable table = *kBuiltin;
    table.acquire_external_dma_access =
        +[](PJRT_RawBuffer* raw_buffer,
            PJRT_ExternalDmaAccess* access) -> PJRT_Error* {
      if (PJRT_Error* error =
              kBuiltin->acquire_external_dma_access(raw_buffer, access)) {
        return error;
      }
      static const PJRT_ExternalDmaAccess_FunctionTable* const kBuiltinAccess =
          access->vtable;
      static const PJRT_ExternalDmaAccess_FunctionTable kForeignAccess = [] {
        PJRT_ExternalDmaAccess_FunctionTable table = *kBuiltinAccess;
        // Hand out the event under a copied table too.
        table.get_event =
            +[](const PJRT_ExternalDmaAccess* access) -> PJRT_DeviceEvent {
          PJRT_DeviceEvent event = kBuiltinAccess->get_event(access);
          static const PJRT_DeviceEvent_FunctionTable kForeignEvent = [&] {
            PJRT_DeviceEvent_FunctionTable table = *event.vtable;
            table.parent = nullptr;
            return table;
          }();
          event.vtable = &kForeignEvent;
          return event;
        };
        return table;
      }();
      access->vtable = &kForeignAccess;
      return nullptr;
    };
    return table;
  }();
  return &kForeign;
}

// A raw buffer whose external DMA access is driven by the test. Calls through
// PjRtRawBufferInterface go through the C function table.
class FakeExternalDmaRawBuffer : public PjRtRawBuffer {
 public:
  // Counts its destruction in the owning buffer.
  class Access : public ExternalDmaAccess {
   public:
    Access(ExternalDmaInfo info, int* releases)
        : ExternalDmaAccess(info), releases_(releases) {}
    ~Access() override { ++*releases_; }

   private:
    int* releases_;
  };

  // By default the buffer pretends to live in another binary.
  explicit FakeExternalDmaRawBuffer(bool foreign = true) {
    if (foreign) {
      vtable = ForeignRawBufferVtable(vtable);
    }
  }

  tsl::AsyncValueRef<ExternalDmaAccess> AcquireExternalDmaAccess() override {
    pending_ = tsl::MakeUnconstructedAsyncValueRef<Access>();
    return pending_;
  }

  void SetInfo(ExternalDmaInfo info) {
    pending_.emplace(info, &releases_);
    pending_.reset();
  }
  void SetError(absl::Status status) {
    pending_.SetError(std::move(status));
    pending_.reset();
  }
  int releases() const { return releases_; }
  tsl::AsyncValue* pending() const { return pending_.GetAsyncValue(); }

  PJRT_Memory* c_memory_space() const override { return nullptr; }
  size_t GetOnDeviceSizeInBytes() const override { return 0; }
  Future<> CopyRawHostToDevice(const void* src, int64_t offset,
                               int64_t transfer_size) override {
    return Future<>(absl::UnimplementedError(""));
  }
  Future<> CopyRawDeviceToHost(void* dst, int64_t offset,
                               int64_t transfer_size) override {
    return Future<>(absl::UnimplementedError(""));
  }
  void* OpaqueDeviceMemoryDataPointer() const override { return nullptr; }
  absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset,
                                         int64_t size) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() override {
    return absl::UnimplementedError("");
  }
  void CopyTo(
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override {}
  void ScheduleCopyTo(
      PjRtDeviceEventRefVector transfer_dependency_events,
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override {}
  PjRtDeviceEventPtr GetRawBufferAsyncValue() override { return {}; }

 private:
  tsl::AsyncValueRef<Access> pending_;
  int releases_ = 0;
};

tsl::AsyncValueRef<PjRtRawBufferInterface::ExternalDmaAccess>
AcquireThroughCApi(PjRtRawBuffer* buffer) {
  return static_cast<PjRtRawBufferInterface*>(buffer)
      ->AcquireExternalDmaAccess();
}

TEST(RawBufferTest, ExternalDmaAccessForwardsInfo) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  auto access = AcquireThroughCApi(buffer.get());
  EXPECT_FALSE(access.IsAvailable());

  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{
      /*fd=*/7, /*offset=*/4096, /*size=*/1024});
  ASSERT_TRUE(access.IsConcrete());
  EXPECT_EQ(access->info().fd, 7);
  EXPECT_EQ(access->info().offset, 4096);
  EXPECT_EQ(access->info().size, 1024);

  EXPECT_EQ(buffer->releases(), 0);
  access.reset();
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessForwardsError) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  auto access = AcquireThroughCApi(buffer.get());

  buffer->SetError(absl::InvalidArgumentError("unreachable"));
  ASSERT_TRUE(access.IsError());
  EXPECT_THAT(access.GetError(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(buffer->releases(), 0);
}

TEST(RawBufferTest, ExternalDmaAccessDroppedBeforeInfo) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  AcquireThroughCApi(buffer.get()).reset();

  // Nobody holds the access, so it ends as soon as it is established.
  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{});
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessDroppedFromCallback) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  auto access = AcquireThroughCApi(buffer.get());
  access.AndThen([&access]() { access.reset(); });

  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{});
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessSameBinaryIsNotWrapped) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>(/*foreign=*/false);
  auto access = AcquireThroughCApi(buffer.get());
  // The backend's async value comes back as is: no wrapper, no extra event.
  EXPECT_EQ(access.GetAsyncValue(), buffer->pending());

  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{/*fd=*/7});
  EXPECT_EQ(access->info().fd, 7);
  access.reset();
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessCApi) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  PJRT_ExternalDmaAccess access;
  ASSERT_EQ(buffer->vtable->acquire_external_dma_access(buffer.get(), &access),
            nullptr);
  PjRtDeviceEventPtr event(access.vtable->get_event(&access));
  EXPECT_EQ(event.state(), PJRT_DeviceEvent_State_Unavailable);

  int fd = -1;
  uint64_t offset = 0;
  uint64_t size = 0;
  EXPECT_THAT(pjrt::PjrtErrorToStatus(
                  access.vtable->get_info(&access, &fd, &offset, &size)),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));

  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{/*fd=*/7});
  EXPECT_EQ(event.state(), PJRT_DeviceEvent_State_Ready);
  ASSERT_EQ(access.vtable->get_info(&access, &fd, &offset, &size), nullptr);
  EXPECT_EQ(fd, 7);

  EXPECT_EQ(buffer->releases(), 0);
  access.vtable->destroy(&access);
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessCApiError) {
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  PJRT_ExternalDmaAccess access;
  ASSERT_EQ(buffer->vtable->acquire_external_dma_access(buffer.get(), &access),
            nullptr);
  PjRtDeviceEventPtr event(access.vtable->get_event(&access));

  buffer->SetError(absl::InvalidArgumentError("unreachable"));
  EXPECT_EQ(event.state(), PJRT_DeviceEvent_State_Error);
  int fd = -1;
  uint64_t offset = 0;
  uint64_t size = 0;
  EXPECT_THAT(pjrt::PjrtErrorToStatus(
                  access.vtable->get_info(&access, &fd, &offset, &size)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  access.vtable->destroy(&access);
}

TEST(RawBufferTest, ExternalDmaAccessCApiEventOutlivesAccess) {
  // A caller may keep its own reference on `event` past `destroy`; that keeps
  // the memory pinned until it is dropped too.
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  PJRT_ExternalDmaAccess access;
  ASSERT_EQ(buffer->vtable->acquire_external_dma_access(buffer.get(), &access),
            nullptr);
  PjRtDeviceEventRef event =
      PjRtDeviceEventPtr(access.vtable->get_event(&access)).CopyRef();

  buffer->SetInfo(PjRtRawBufferInterface::ExternalDmaInfo{});
  access.vtable->destroy(&access);
  EXPECT_EQ(buffer->releases(), 0);
  event.reset();
  EXPECT_EQ(buffer->releases(), 1);
}

TEST(RawBufferTest, ExternalDmaAccessOlderPlugin) {
  // A plugin built before the slot was appended has a shorter table.
  auto buffer = tsl::MakeRef<FakeExternalDmaRawBuffer>();
  PJRT_RawBuffer_FunctionTable older = *buffer->vtable;
  older.struct_size =
      offsetof(PJRT_RawBuffer_FunctionTable, acquire_external_dma_access);
  buffer->vtable = &older;

  auto access = AcquireThroughCApi(buffer.get());
  ASSERT_TRUE(access.IsError());
  EXPECT_THAT(access.GetError(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace xla
