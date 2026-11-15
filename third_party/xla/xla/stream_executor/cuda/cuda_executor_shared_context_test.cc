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

// Tests for CudaExecutor with
// XLA_CUDA_REGISTER_HOST_MEMORY_IN_SHARED_CONTEXT=1, which the BUILD target
// sets for the whole test process.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ne;

constexpr size_t kBufferSize = 64 * 1024;

absl::StatusOr<StreamExecutor*> GetExecutor(int ordinal) {
  ABSL_ASSIGN_OR_RETURN(Platform * platform,
                   PlatformManager::PlatformWithName("CUDA"));
  return platform->ExecutorForDevice(ordinal);
}

absl::StatusOr<CUcontext> CurrentContext() {
  CUcontext context = nullptr;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(cuCtxGetCurrent(&context)));
  return context;
}

// Returns the current context after activating `executor`'s primary context.
absl::StatusOr<CUcontext> PrimaryContext(StreamExecutor* executor) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return CurrentContext();
}

// Returns the context that `ptr` was registered in.
absl::StatusOr<CUcontext> RegistrationContext(const void* ptr) {
  CUcontext context = nullptr;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(
      cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT,
                            reinterpret_cast<CUdeviceptr>(ptr))));
  return context;
}

// Shared context creation runs once per process, on the thread of the first
// registration. Do that first registration on a fresh thread, so we can check
// that creating the shared context leaves only the primary context on the
// thread's context stack.
class FirstRegistrationOnFreshThread : public ::testing::Environment {
 public:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
    ASSERT_OK_AND_ASSIGN(CUcontext primary, PrimaryContext(executor));
    // The thread's destructor joins it.
    std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "first_registration", [&] {
          ASSERT_OK_AND_ASSIGN(CUcontext current, CurrentContext());
          ASSERT_EQ(current, nullptr);
          std::vector<uint8_t> buffer(kBufferSize);
          ASSERT_TRUE(
              executor->HostMemoryRegister(buffer.data(), buffer.size()));
          ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));

          CUcontext popped = nullptr;
          ASSERT_OK(cuda::ToStatus(cuCtxPopCurrent(&popped)));
          EXPECT_EQ(popped, primary);
          ASSERT_OK_AND_ASSIGN(current, CurrentContext());
          EXPECT_EQ(current, nullptr);
        }));
  }
};

[[maybe_unused]] ::testing::Environment* const kFirstRegistrationEnv =
    ::testing::AddGlobalTestEnvironment(new FirstRegistrationOnFreshThread);

TEST(CudaExecutorSharedContextTest, RegistersInSharedContext) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(CUcontext primary, PrimaryContext(executor));
  std::vector<uint8_t> buffer(kBufferSize);

  ASSERT_TRUE(executor->HostMemoryRegister(buffer.data(), buffer.size()));
  EXPECT_TRUE(executor->IsHostMemoryPinned(buffer.data(), buffer.size()));
  ASSERT_OK_AND_ASSIGN(CUcontext context, RegistrationContext(buffer.data()));
  EXPECT_NE(context, nullptr);
  EXPECT_NE(context, primary);

  ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));
  EXPECT_FALSE(executor->IsHostMemoryPinned(buffer.data(), buffer.size()));
}

TEST(CudaExecutorSharedContextTest, LeavesPrimaryContextCurrent) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(CUcontext primary, PrimaryContext(executor));
  std::vector<uint8_t> buffer(kBufferSize);

  // A fresh thread has no current context. Like registering in the primary
  // context, registering in the shared context leaves the primary context
  // current, and the shared context is no longer on the stack.
  {
    // The thread's destructor joins it.
    std::unique_ptr<tsl::Thread> thread(
        tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "register", [&] {
          EXPECT_THAT(CurrentContext(), IsOkAndHolds(nullptr));
          ASSERT_TRUE(
              executor->HostMemoryRegister(buffer.data(), buffer.size()));
          EXPECT_THAT(CurrentContext(), IsOkAndHolds(primary));
          ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));
          EXPECT_THAT(CurrentContext(), IsOkAndHolds(primary));

          CUcontext popped = nullptr;
          ASSERT_OK(cuda::ToStatus(cuCtxPopCurrent(&popped)));
          EXPECT_EQ(popped, primary);
          EXPECT_THAT(CurrentContext(), IsOkAndHolds(nullptr));
        }));
  }

  // Same inside an active scope of the primary context.
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  ASSERT_TRUE(executor->HostMemoryRegister(buffer.data(), buffer.size()));
  EXPECT_THAT(CurrentContext(), IsOkAndHolds(primary));
  ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));
  EXPECT_THAT(CurrentContext(), IsOkAndHolds(primary));
}

TEST(CudaExecutorSharedContextTest, CopiesThroughRegisteredMemory) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());
  std::vector<uint8_t> src(kBufferSize);
  std::vector<uint8_t> dst(kBufferSize, 0);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i * 7 + 1);
  }
  ASSERT_TRUE(executor->HostMemoryRegister(src.data(), src.size()));
  ASSERT_TRUE(executor->HostMemoryRegister(dst.data(), dst.size()));

  DeviceAddressBase device = executor->Allocate(kBufferSize);
  ASSERT_NE(device.opaque(), nullptr);
  ASSERT_OK(stream->Memcpy(&device, src.data(), kBufferSize));
  ASSERT_OK(stream->Memcpy(dst.data(), device, kBufferSize));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(src, dst);
  executor->Deallocate(&device);

  ASSERT_TRUE(executor->HostMemoryUnregister(src.data()));
  ASSERT_TRUE(executor->HostMemoryUnregister(dst.data()));
}

TEST(CudaExecutorSharedContextTest, CopiesOnAnotherDevice) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Needs 2 GPUs.";
  }
  // Like PJRT, register through device 0 and copy on device 1.
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor0, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor1, GetExecutor(1));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor1->CreateStream());
  std::vector<uint8_t> src(kBufferSize, 0x5a);
  std::vector<uint8_t> dst(kBufferSize, 0);
  ASSERT_TRUE(executor0->HostMemoryRegister(src.data(), src.size()));
  ASSERT_TRUE(executor0->HostMemoryRegister(dst.data(), dst.size()));
  EXPECT_TRUE(executor1->IsHostMemoryPinned(src.data(), src.size()));
  {
    // Device 1's primary context sees the memory as page-locked: it has a
    // device pointer. Without ATS (pageable memory access through the host
    // page tables), unregistered memory doesn't.
    std::unique_ptr<ActivateContext> activation = executor1->Activate();
    CUdeviceptr device_ptr = 0;
    EXPECT_OK(cuda::ToStatus(
        cuMemHostGetDevicePointer(&device_ptr, src.data(), /*Flags=*/0)));
    CUdevice device1 = 0;
    ASSERT_OK(cuda::ToStatus(cuCtxGetDevice(&device1)));
    int uses_host_page_tables = 0;
    ASSERT_OK(cuda::ToStatus(cuDeviceGetAttribute(
        &uses_host_page_tables,
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        device1)));
    if (!uses_host_page_tables) {
      std::vector<uint8_t> unregistered(kBufferSize);
      EXPECT_NE(cuMemHostGetDevicePointer(&device_ptr, unregistered.data(),
                                          /*Flags=*/0),
                CUDA_SUCCESS);
    }
  }

  DeviceAddressBase device = executor1->Allocate(kBufferSize);
  ASSERT_NE(device.opaque(), nullptr);
  ASSERT_OK(stream->Memcpy(&device, src.data(), kBufferSize));
  ASSERT_OK(stream->Memcpy(dst.data(), device, kBufferSize));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(src, dst);
  executor1->Deallocate(&device);

  ASSERT_TRUE(executor0->HostMemoryUnregister(src.data()));
  ASSERT_TRUE(executor0->HostMemoryUnregister(dst.data()));
}

// The point of the shared context: unregistering doesn't wait for work in any
// executor's context, even though the registering executor's primary context
// is on the context stack below the shared context during the unregister.
TEST(CudaExecutorSharedContextTest, UnregisterDoesNotWaitForExecutorWork) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(CUcontext primary, PrimaryContext(executor));
  std::vector<uint8_t> buffer(kBufferSize);
  ASSERT_TRUE(executor->HostMemoryRegister(buffer.data(), buffer.size()));
  ASSERT_THAT(RegistrationContext(buffer.data()), IsOkAndHolds(Ne(primary)));

  // Block a stream of every device's executor until the unregister is done.
  absl::Notification unblock;
  std::vector<std::unique_ptr<Stream>> streams;
  for (int ordinal = 0; ordinal < platform->VisibleDeviceCount(); ++ordinal) {
    ASSERT_OK_AND_ASSIGN(StreamExecutor * device_executor,
                         GetExecutor(ordinal));
    ASSERT_OK_AND_ASSIGN(streams.emplace_back(),
                         device_executor->CreateStream());
    ASSERT_OK(
        streams.back()->DoHostCallback([&] { unblock.WaitForNotification(); }));
  }

  bool unregistered = false;
  {
    absl::Notification done;
    // The thread's destructor joins it.
    std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "unregister", [&] {
          unregistered = executor->HostMemoryUnregister(buffer.data());
          done.Notify();
        }));
    EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(30)));
    unblock.Notify();
  }
  EXPECT_TRUE(unregistered);
  for (std::unique_ptr<Stream>& stream : streams) {
    ASSERT_OK(stream->BlockHostUntilDone());
  }
}

TEST(CudaExecutorSharedContextTest, AllExecutorsShareOneContext) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * cached, GetExecutor(0));
  CudaExecutor uncached(platform, /*device_ordinal=*/0);
  ASSERT_OK(uncached.Init());
  std::vector<StreamExecutor*> executors = {cached, &uncached};
  if (platform->VisibleDeviceCount() >= 2) {
    ASSERT_OK_AND_ASSIGN(StreamExecutor * other_device, GetExecutor(1));
    executors.push_back(other_device);
  }

  std::vector<std::vector<uint8_t>> buffers(executors.size(),
                                            std::vector<uint8_t>(kBufferSize));
  for (int i = 0; i < executors.size(); ++i) {
    ASSERT_TRUE(
        executors[i]->HostMemoryRegister(buffers[i].data(), kBufferSize));
  }
  ASSERT_OK_AND_ASSIGN(CUcontext context,
                       RegistrationContext(buffers[0].data()));
  for (int i = 1; i < executors.size(); ++i) {
    EXPECT_THAT(RegistrationContext(buffers[i].data()), IsOkAndHolds(context));
  }

  for (int i = 0; i < executors.size(); ++i) {
    EXPECT_TRUE(executors[i]->HostMemoryUnregister(buffers[i].data()));
  }
}

TEST(CudaExecutorSharedContextTest, ConcurrentRegistration) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetExecutor(0));
  ASSERT_OK_AND_ASSIGN(CUcontext primary, PrimaryContext(executor));
  constexpr int kNumThreads = 8;
  constexpr int kNumIterations = 50;
  absl::Mutex mu;
  std::vector<CUcontext> contexts;
  absl::Notification start;
  {
    // The threads' destructors join them.
    std::vector<std::unique_ptr<tsl::Thread>> threads;
    for (int t = 0; t < kNumThreads; ++t) {
      threads.emplace_back(tsl::Env::Default()->StartThread(
          tsl::ThreadOptions(), "register", [&] {
            std::vector<uint8_t> buffer(kBufferSize);
            start.WaitForNotification();
            for (int i = 0; i < kNumIterations; ++i) {
              ASSERT_TRUE(
                  executor->HostMemoryRegister(buffer.data(), buffer.size()));
              absl::StatusOr<CUcontext> context =
                  RegistrationContext(buffer.data());
              ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));
              ASSERT_OK(context);
              absl::MutexLock lock(mu);
              contexts.push_back(*context);
            }
          }));
    }
    start.Notify();
  }
  ASSERT_EQ(contexts.size(), kNumThreads * kNumIterations);
  EXPECT_THAT(contexts, Each(Eq(contexts.front())));
  EXPECT_THAT(contexts.front(), Ne(primary));
}

}  // namespace
}  // namespace stream_executor::gpu
