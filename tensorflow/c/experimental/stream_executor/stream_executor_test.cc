/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_test_util.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace stream_executor {
namespace {

/*** Registration tests ***/
TEST(StreamExecutor, SuccessfulRegistration) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
  };
  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  TF_ASSERT_OK(status);
  absl::StatusOr<Platform*> maybe_platform =
      PlatformManager::PlatformWithName("MY_DEVICE");
  TF_ASSERT_OK(maybe_platform.status());
  Platform* platform = std::move(maybe_platform).value();
  ASSERT_EQ(platform->Name(), test_util::kDeviceName);
  ASSERT_EQ(platform->VisibleDeviceCount(), test_util::kDeviceCount);

  absl::StatusOr<StreamExecutor*> maybe_executor =
      platform->ExecutorForDevice(0);
  TF_ASSERT_OK(maybe_executor.status());
}

TEST(StreamExecutor, NameNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = nullptr;
  };

  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.message(), "'name' field in SP_Platform must be set.");
}

TEST(StreamExecutor, InvalidNameWithSemicolon) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = "INVALID:NAME";
  };

  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  EXPECT_THAT(
      status.message(),
      testing::ContainsRegex("Device name/type 'INVALID:NAME' must match"));
}

TEST(StreamExecutor, InvalidNameWithSlash) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->name = "INVALID/";
  };

  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  EXPECT_THAT(status.message(),
              testing::ContainsRegex("Device name/type 'INVALID/' must match"));
}

TEST(StreamExecutor, CreateDeviceNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform_fns->create_device = nullptr;
  };

  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.message(),
            "'create_device' field in SP_PlatformFns must be set.");
}

TEST(StreamExecutor, UnifiedMemoryAllocateNotSet) {
  auto plugin_init = [](SE_PlatformRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    test_util::PopulateDefaultPlatformRegistrationParams(params);
    params->platform->supports_unified_memory = true;
  };

  std::string device_type, platform_name;
  absl::Status status =
      InitStreamExecutorPlugin(plugin_init, &device_type, &platform_name);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(
      status.message(),
      "'unified_memory_allocate' field in SP_StreamExecutor must be set.");
}

/*** StreamExecutor behavior tests ***/
class StreamExecutorTest : public ::testing::Test {
 protected:
  StreamExecutorTest() {}
  void SetUp() override {
    test_util::PopulateDefaultPlatform(&platform_, &platform_fns_);
    test_util::PopulateDefaultDeviceFns(&device_fns_);
    test_util::PopulateDefaultStreamExecutor(&se_);
    test_util::PopulateDefaultTimerFns(&timer_fns_);
  }
  void TearDown() override {}

  StreamExecutor* GetExecutor(int ordinal) {
    if (!cplatform_) {
      cplatform_ = absl::make_unique<CPlatform>(
          platform_, test_util::DestroyPlatform, platform_fns_,
          test_util::DestroyPlatformFns, device_fns_, se_, timer_fns_);
    }
    absl::StatusOr<StreamExecutor*> maybe_executor =
        cplatform_->ExecutorForDevice(ordinal);
    TF_CHECK_OK(maybe_executor.status());
    return std::move(maybe_executor).value();
  }
  SP_Platform platform_;
  SP_PlatformFns platform_fns_;
  SP_DeviceFns device_fns_;
  SP_StreamExecutor se_;
  SP_TimerFns timer_fns_;
  std::unique_ptr<CPlatform> cplatform_;
};

TEST_F(StreamExecutorTest, Allocate) {
  se_.allocate = [](const SP_Device* const device, uint64_t size,
                    int64_t memory_space, SP_DeviceMemoryBase* const mem) {
    mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    mem->opaque = malloc(size);
    mem->size = size;
  };
  se_.deallocate = [](const SP_Device* const device,
                      SP_DeviceMemoryBase* const mem) {
    EXPECT_EQ(mem->size, 2 * sizeof(int));
    free(mem->opaque);
    mem->opaque = nullptr;
    mem->size = 0;
  };
  StreamExecutor* executor = GetExecutor(0);
  DeviceMemory<int> mem = executor->AllocateArray<int>(2);
  ASSERT_NE(mem.opaque(), nullptr);
  ASSERT_EQ(mem.size(), 2 * sizeof(int));
  executor->Deallocate(&mem);
}

TEST_F(StreamExecutorTest, HostMemoryAllocate) {
  static bool allocate_called = false;
  static bool deallocate_called = false;
  se_.host_memory_allocate = [](const SP_Device* const device, uint64_t size) {
    allocate_called = true;
    return malloc(size);
  };
  se_.host_memory_deallocate = [](const SP_Device* const device, void* mem) {
    free(mem);
    deallocate_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(allocate_called);
  TF_ASSERT_OK_AND_ASSIGN(auto mem, executor->HostMemoryAllocate(8));
  ASSERT_NE(mem->opaque(), nullptr);
  ASSERT_TRUE(allocate_called);
  ASSERT_FALSE(deallocate_called);
  mem.reset();
  ASSERT_TRUE(deallocate_called);
}

TEST_F(StreamExecutorTest, HostMemoryAllocator) {
  static bool allocate_called = false;
  static bool deallocate_called = false;
  se_.host_memory_allocate = [](const SP_Device* const device, uint64_t size) {
    allocate_called = true;
    return malloc(size);
  };
  se_.host_memory_deallocate = [](const SP_Device* const device, void* mem) {
    free(mem);
    deallocate_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(allocate_called);
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          executor->CreateMemoryAllocator(MemoryType::kHost));
  TF_ASSERT_OK_AND_ASSIGN(auto mem, allocator->Allocate(8));
  ASSERT_NE(mem->opaque(), nullptr);
  ASSERT_TRUE(allocate_called);
  ASSERT_FALSE(deallocate_called);
  mem.reset();
  ASSERT_TRUE(deallocate_called);
}

TEST_F(StreamExecutorTest, UnifiedMemoryAllocate) {
  static bool allocate_called = false;
  static bool deallocate_called = false;
  se_.unified_memory_allocate = [](const SP_Device* const device,
                                   uint64_t size) {
    allocate_called = true;
    return malloc(size);
  };
  se_.unified_memory_deallocate = [](const SP_Device* const device, void* mem) {
    free(mem);
    deallocate_called = true;
  };
  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(allocate_called);
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator, executor->CreateMemoryAllocator(MemoryType::kUnified));
  TF_ASSERT_OK_AND_ASSIGN(auto mem, allocator->Allocate(8));
  ASSERT_NE(mem->opaque(), nullptr);
  ASSERT_TRUE(allocate_called);
  ASSERT_FALSE(deallocate_called);
  mem.reset();
  ASSERT_TRUE(deallocate_called);
}

TEST_F(StreamExecutorTest, GetAllocatorStats) {
  se_.get_allocator_stats = [](const SP_Device* const device,
                               SP_AllocatorStats* const stat) -> TF_Bool {
    stat->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
    stat->bytes_in_use = 123;
    return true;
  };

  StreamExecutor* executor = GetExecutor(0);
  absl::optional<AllocatorStats> optional_stats = executor->GetAllocatorStats();
  ASSERT_TRUE(optional_stats.has_value());
  AllocatorStats stats = optional_stats.value();
  ASSERT_EQ(stats.bytes_in_use, 123);
}

TEST_F(StreamExecutorTest, DeviceMemoryUsage) {
  se_.device_memory_usage = [](const SP_Device* const device,
                               int64_t* const free,
                               int64_t* const total) -> TF_Bool {
    *free = 45;
    *total = 7;
    return true;
  };

  StreamExecutor* executor = GetExecutor(0);
  int64_t free = 0;
  int64_t total = 0;
  executor->DeviceMemoryUsage(&free, &total);
  ASSERT_EQ(free, 45);
  ASSERT_EQ(total, 7);
}

TEST_F(StreamExecutorTest, CreateStream) {
  static bool stream_created = false;
  static bool stream_deleted = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
    stream_created = true;
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void {
    auto custom_stream = static_cast<SP_Stream_st*>(stream);
    ASSERT_EQ(custom_stream->stream_id, 14);
    delete custom_stream;
    stream_deleted = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(stream_created);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_TRUE(stream_created);
  ASSERT_FALSE(stream_deleted);
  stream.reset();
  ASSERT_TRUE(stream_deleted);
}

TEST_F(StreamExecutorTest, CreateStreamDependency) {
  static bool create_stream_dependency_called = false;
  se_.create_stream_dependency = [](const SP_Device* const device,
                                    SP_Stream dependent, SP_Stream other,
                                    TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    create_stream_dependency_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto dependent, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto other, executor->CreateStream());
  ASSERT_FALSE(create_stream_dependency_called);
  TF_ASSERT_OK(dependent->WaitFor(other.get()));
  ASSERT_TRUE(create_stream_dependency_called);
}

TEST_F(StreamExecutorTest, StreamStatus) {
  static bool status_ok = true;
  se_.get_stream_status = [](const SP_Device* const device, SP_Stream stream,
                             TF_Status* const status) -> void {
    if (status_ok) {
      TF_SetStatus(status, TF_OK, "");
    } else {
      TF_SetStatus(status, TF_INTERNAL, "Test error");
    }
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK(stream->RefreshStatus());
  status_ok = false;
  auto updated_status = stream->RefreshStatus();
  ASSERT_FALSE(stream->ok());
  ASSERT_EQ(updated_status.message(), "Test error");
}

TEST_F(StreamExecutorTest, CreateEvent) {
  static bool event_created = false;
  static bool event_deleted = false;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(123);
    event_created = true;
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void {
    auto custom_event = static_cast<SP_Event_st*>(event);
    ASSERT_EQ(custom_event->event_id, 123);
    delete custom_event;
    event_deleted = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(event_created);
  TF_ASSERT_OK_AND_ASSIGN(auto event, executor->CreateEvent());
  ASSERT_TRUE(event_created);
  ASSERT_FALSE(event_deleted);
  event.reset();
  ASSERT_TRUE(event_deleted);
}

TEST_F(StreamExecutorTest, PollForEventStatus) {
  static SE_EventStatus event_status = SE_EVENT_COMPLETE;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(123);
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void { delete event; };
  se_.get_event_status = [](const SP_Device* const device,
                            SP_Event event) -> SE_EventStatus {
    EXPECT_EQ(event->event_id, 123);
    return event_status;
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto event, executor->CreateEvent());
  ASSERT_EQ(event->PollForStatus(), Event::Status::kComplete);
  event_status = SE_EVENT_ERROR;
  ASSERT_EQ(event->PollForStatus(), Event::Status::kError);
}

TEST_F(StreamExecutorTest, RecordAndWaitForEvent) {
  static bool record_called = false;
  static bool wait_called = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(1);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(2);
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void { delete event; };
  se_.record_event = [](const SP_Device* const device, SP_Stream stream,
                        SP_Event event, TF_Status* const status) {
    EXPECT_EQ(stream->stream_id, 1);
    EXPECT_EQ(event->event_id, 2);
    TF_SetStatus(status, TF_OK, "");
    record_called = true;
  };
  se_.wait_for_event = [](const SP_Device* const device, SP_Stream stream,
                          SP_Event event, TF_Status* const status) {
    EXPECT_EQ(stream->stream_id, 1);
    EXPECT_EQ(event->event_id, 2);
    TF_SetStatus(status, TF_OK, "");
    wait_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto event, executor->CreateEvent());
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_FALSE(record_called);
  TF_ASSERT_OK(stream->RecordEvent(event.get()));
  ASSERT_TRUE(record_called);
  ASSERT_FALSE(wait_called);
  TF_ASSERT_OK(stream->WaitFor(event.get()));
  ASSERT_TRUE(wait_called);
}

TEST_F(StreamExecutorTest, MemcpyToHost) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.memcpy_dtoh = [](const SP_Device* const device, SP_Stream stream,
                       void* host_dst,
                       const SP_DeviceMemoryBase* const device_src,
                       uint64_t size, TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    std::memcpy(host_dst, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  size_t size = sizeof(int);
  int src_data = 34;
  int dst_data = 2;
  DeviceMemoryBase device_src(&src_data, size);
  TF_ASSERT_OK(stream->Memcpy(&dst_data, device_src, size));
  ASSERT_EQ(dst_data, 34);
}

TEST_F(StreamExecutorTest, MemcpyFromHost) {
  se_.memcpy_htod = [](const SP_Device* const device, SP_Stream stream,
                       SP_DeviceMemoryBase* const device_dst,
                       const void* host_src, uint64_t size,
                       TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, host_src, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  TF_ASSERT_OK(stream->Memcpy(&device_dst, &src_data, size));
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, MemcpyDeviceToDevice) {
  se_.memcpy_dtod = [](const SP_Device* const device, SP_Stream stream,
                       SP_DeviceMemoryBase* const device_dst,
                       const SP_DeviceMemoryBase* const device_src,
                       uint64_t size, TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  DeviceMemoryBase device_src(&src_data, size);
  TF_ASSERT_OK(stream->Memcpy(&device_dst, device_src, size));
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, SyncMemcpyToHost) {
  se_.sync_memcpy_dtoh = [](const SP_Device* const device, void* host_dst,
                            const SP_DeviceMemoryBase* const device_src,
                            uint64_t size, TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    std::memcpy(host_dst, device_src->opaque, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  size_t size = sizeof(int);
  int src_data = 34;
  int dst_data = 2;
  DeviceMemoryBase device_src(&src_data, size);
  TF_ASSERT_OK(executor->SynchronousMemcpyD2H(device_src, size, &dst_data));
  ASSERT_EQ(dst_data, 34);
}

TEST_F(StreamExecutorTest, SyncMemcpyFromHost) {
  se_.sync_memcpy_htod =
      [](const SP_Device* const device, SP_DeviceMemoryBase* const device_dst,
         const void* host_src, uint64_t size, TF_Status* const status) {
        TF_SetStatus(status, TF_OK, "");
        std::memcpy(device_dst->opaque, host_src, size);
      };

  StreamExecutor* executor = GetExecutor(0);
  size_t size = sizeof(int);
  int src_data = 18;
  int dst_data = 0;
  DeviceMemoryBase device_dst(&dst_data, size);
  TF_ASSERT_OK(executor->SynchronousMemcpyH2D(&src_data, size, &device_dst));
  ASSERT_EQ(dst_data, 18);
}

TEST_F(StreamExecutorTest, BlockHostForEvent) {
  static bool block_host_for_event_called = false;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) {
    *event = new SP_Event_st(357);
  };
  se_.destroy_event = [](const SP_Device* const device, SP_Event event) {
    delete event;
  };
  se_.block_host_for_event = [](const SP_Device* const device, SP_Event event,
                                TF_Status* const status) -> void {
    ASSERT_EQ(event->event_id, 357);
    TF_SetStatus(status, TF_OK, "");
    block_host_for_event_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_FALSE(block_host_for_event_called);
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_TRUE(block_host_for_event_called);
}

TEST_F(StreamExecutorTest, BlockHostUntilDone) {
  static bool block_host_until_done_called = false;
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) {
    *stream = new SP_Stream_st(58);
  };
  se_.destroy_stream = [](const SP_Device* const device, SP_Stream stream) {
    delete stream;
  };
  se_.block_host_until_done = [](const SP_Device* const device,
                                 SP_Stream stream,
                                 TF_Status* const status) -> void {
    ASSERT_EQ(stream->stream_id, 58);
    TF_SetStatus(status, TF_OK, "");
    block_host_until_done_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_FALSE(block_host_until_done_called);
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_TRUE(block_host_until_done_called);
}

TEST_F(StreamExecutorTest, SynchronizeAllActivity) {
  static bool synchronize_all_called = false;
  se_.synchronize_all_activity = [](const SP_Device* const device,
                                    TF_Status* const status) {
    TF_SetStatus(status, TF_OK, "");
    synchronize_all_called = true;
  };

  StreamExecutor* executor = GetExecutor(0);
  ASSERT_FALSE(synchronize_all_called);
  ASSERT_TRUE(executor->SynchronizeAllActivity());
  ASSERT_TRUE(synchronize_all_called);
}

TEST_F(StreamExecutorTest, HostCallbackOk) {
  se_.host_callback = [](const SP_Device* const device, SP_Stream stream,
                         SE_StatusCallbackFn const callback_fn,
                         void* const callback_arg) -> TF_Bool {
    TF_Status* status = TF_NewStatus();
    callback_fn(callback_arg, status);
    bool ok = TF_GetCode(status) == TF_OK;
    TF_DeleteStatus(status);
    return ok;
  };
  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::function<absl::Status()> callback = []() -> absl::Status {
    return absl::OkStatus();
  };
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(callback));
}

TEST_F(StreamExecutorTest, HostCallbackError) {
  se_.host_callback = [](const SP_Device* const device, SP_Stream stream,
                         SE_StatusCallbackFn const callback_fn,
                         void* const callback_arg) -> TF_Bool {
    TF_Status* status = TF_NewStatus();
    callback_fn(callback_arg, status);
    bool ok = TF_GetCode(status) == TF_OK;
    TF_DeleteStatus(status);
    return ok;
  };
  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::function<absl::Status()> callback = []() -> absl::Status {
    return tsl::errors::Unimplemented("Unimplemented");
  };
  ASSERT_FALSE(stream->DoHostCallbackWithStatus(callback).ok());
}

TEST_F(StreamExecutorTest, DeviceDescription) {
  static const char* hardware_name = "TestName";
  static const char* vendor = "TestVendor";
  static const char* pci_bus_id = "TestPCIBusId";
  platform_fns_.create_device = [](const SP_Platform* platform,
                                   SE_CreateDeviceParams* params,
                                   TF_Status* status) {
    params->device->hardware_name = hardware_name;
    params->device->device_vendor = vendor;
    params->device->pci_bus_id = pci_bus_id;
  };

  device_fns_.get_numa_node = [](const SP_Device* device) { return 123; };
  device_fns_.get_memory_bandwidth = [](const SP_Device* device) -> int64_t {
    return 54;
  };
  device_fns_.get_gflops = [](const SP_Device* device) -> double { return 32; };

  StreamExecutor* executor = GetExecutor(0);
  const DeviceDescription& description = executor->GetDeviceDescription();
  ASSERT_EQ(description.name(), "TestName");
  ASSERT_EQ(description.device_vendor(), "TestVendor");
  ASSERT_EQ(description.pci_bus_id(), "TestPCIBusId");
  ASSERT_EQ(description.numa_node(), 123);
  ASSERT_EQ(description.memory_bandwidth(), 54);
}

TEST_F(StreamExecutorTest, DeviceDescriptionNumaNodeNotSet) {
  static const char* hardware_name = "TestName";
  static const char* vendor = "TestVendor";
  static const char* pci_bus_id = "TestPCIBusId";
  platform_fns_.create_device = [](const SP_Platform* platform,
                                   SE_CreateDeviceParams* params,
                                   TF_Status* status) {
    params->device->hardware_name = hardware_name;
    params->device->device_vendor = vendor;
    params->device->pci_bus_id = pci_bus_id;
  };

  device_fns_.get_memory_bandwidth = [](const SP_Device* device) -> int64_t {
    return 54;
  };
  device_fns_.get_gflops = [](const SP_Device* device) -> double { return 32; };

  StreamExecutor* executor = GetExecutor(0);
  const DeviceDescription& description = executor->GetDeviceDescription();
  ASSERT_EQ(description.name(), "TestName");
  ASSERT_EQ(description.device_vendor(), "TestVendor");
  ASSERT_EQ(description.pci_bus_id(), "TestPCIBusId");
  ASSERT_EQ(description.numa_node(), -1);
  ASSERT_EQ(description.memory_bandwidth(), 54);
}

TEST_F(StreamExecutorTest, MemZero) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.mem_zero = [](const SP_Device* device, SP_Stream stream,
                    SP_DeviceMemoryBase* location, uint64_t size,
                    TF_Status* status) {
    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    std::memset(location->opaque, 0, size);
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  size_t size = sizeof(int);
  int data = 2;
  DeviceMemoryBase device_data(&data, size);
  TF_ASSERT_OK(stream->MemZero(&device_data, size));
  ASSERT_EQ(data, 0);
}

TEST_F(StreamExecutorTest, Memset32) {
  se_.create_stream = [](const SP_Device* const device, SP_Stream* stream,
                         TF_Status* const status) -> void {
    *stream = new SP_Stream_st(14);
  };
  se_.destroy_stream = [](const SP_Device* const device,
                          SP_Stream stream) -> void { delete stream; };

  se_.memset32 = [](const SP_Device* device, SP_Stream stream,
                    SP_DeviceMemoryBase* location, uint32_t pattern,
                    uint64_t size, TF_Status* status) {
    TF_SetStatus(status, TF_OK, "");
    EXPECT_EQ(stream->stream_id, 14);
    EXPECT_EQ(size % 4, 0);
    auto ptr = static_cast<uint32_t*>(location->opaque);
    for (int i = 0; i < size / 4; i++) {
      *(ptr + i) = pattern;
    }
  };

  StreamExecutor* executor = GetExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  size_t size = sizeof(int);
  int data = 2;
  DeviceMemoryBase device_data(&data, size);
  TF_ASSERT_OK(stream->Memset32(&device_data, 18, size));
  ASSERT_EQ(data, 18);
}

}  // namespace
}  // namespace stream_executor
