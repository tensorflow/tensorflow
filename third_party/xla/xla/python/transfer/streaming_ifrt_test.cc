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
#include "xla/python/transfer/streaming_ifrt.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/test_pattern.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {
namespace {

xla::ifrt::PjRtDevice* GetOtherDevice(xla::ifrt::ArrayRef arr) {
  auto* ifrt_client =
      llvm::dyn_cast_or_null<xla::ifrt::PjRtClient>(arr->client());
  return llvm::dyn_cast<xla::ifrt::PjRtDevice>(ifrt_client->devices()[1]);
}

xla::ifrt::PjRtClient* GetIfrtClient(xla::ifrt::ArrayRef arr) {
  return llvm::dyn_cast_or_null<xla::ifrt::PjRtClient>(arr->client());
}

xla::Shape ShapeFromIfrt(xla::ifrt::ArrayRef arr) {
  auto* pjrt_arr =
      llvm::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(arr.get());
  auto buffer = pjrt_arr->pjrt_buffers()[0].get();
  return buffer->on_device_shape();
}

struct SingleBufferCopyPlan {
  std::vector<tsl::RCReference<ChunkDestination>> dests;
  std::vector<xla::ifrt::ArrayRef> arrays;
};

// Single buffer copy plan example.
absl::StatusOr<SingleBufferCopyPlan> SetupTransferDestList(
    xla::Shape shape, xla::ifrt::PjRtDevice* device,
    xla::ifrt::PjRtClient* ifrt_client, size_t xfer_size) {
  auto* pjrt_client = ifrt_client->pjrt_client();
  // CHECK_EQ(pjrt_client->platform_id(), xla::TpuId());
  TF_ASSIGN_OR_RETURN(auto* pjrt_memory_space,
                      device->pjrt_device()->default_memory_space());
  TF_ASSIGN_OR_RETURN(auto atm_owned,
                      pjrt_client->CreateBuffersForAsyncHostToDevice(
                          {shape}, pjrt_memory_space));
  auto atm = std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>(
      std::move(atm_owned));
  SingleBufferCopyPlan results;
  size_t copy_size = xla::ShapeUtil::ByteSizeOf(shape);

  results.dests.push_back(MakeDmaDestination(atm, 0, copy_size));
  TF_ASSIGN_OR_RETURN(auto arr,
                   ifrt_client->CreatePjRtArray(atm->RetrieveBuffer(0)));
  results.arrays.push_back(std::move(arr));
  return results;
}

TEST(PremappedCopierState, RoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  size_t xfer_size = 1024 * 1024;
  auto test_pattern = tests::CreateTestPattern(0, 16l * 1024 * 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, tests::CopyTestPatternToDevice(
                    client.get(), client->devices()[0], test_pattern));
  TF_ASSERT_OK_AND_ASSIGN(
      auto scratch, AllocateAndMapPjrtMemory(arr->client(), 1024 * 1024 * 16));
  TF_ASSERT_OK_AND_ASSIGN(
      auto src_work_units,
      DmaCopyChunk::DivideBufferCopiesEvenly(arr, xfer_size, 0));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_copy_plan,
      SetupTransferDestList(ShapeFromIfrt(arr), GetOtherDevice(arr),
                            GetIfrtClient(arr), xfer_size));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, xfer_size);

  absl::Mutex mu;
  struct LocalQueueInfo {
    void* buff;
    size_t offset;
    size_t size;
  };
  std::deque<LocalQueueInfo> local_queue;

  for (size_t i = 0; i < src_work_units.size(); ++i) {
    cstate->ScheduleCopy(
        src_work_units[i],
        [&mu, &local_queue](PremappedCopierState* state, void* buf,
                            const DmaCopyChunk& chunk) {
          absl::MutexLock l(&mu);
          local_queue.push_back(LocalQueueInfo{buf, chunk.offset, chunk.size});
        });
  }
  for (size_t i = 0; i < src_work_units.size(); ++i) {
    auto cond = [&]() -> bool { return !local_queue.empty(); };
    mu.LockWhen(absl::Condition(&cond));
    auto state = local_queue.front();
    local_queue.pop_front();
    mu.Unlock();
    TF_ASSERT_OK(dest_copy_plan.dests[0]->Put(
        state.buff, state.offset, state.size,
        [cstate, buf = state.buff]() { cstate->ReturnBuffer(buf); }));
  }

  auto pull_result_arr = dest_copy_plan.arrays[0];

  std::vector<int32_t> result;
  result.resize(test_pattern.size());
  TF_ASSERT_OK(pull_result_arr
                ->CopyToHostBuffer(result.data(), std::nullopt,
                                   xla::ifrt::ArrayCopySemantics::kReuseInput)
                .Await());
  EXPECT_EQ(result, test_pattern);
}

TEST(Semaphore, Basic) {
  internal::IsLastSemaphore semaphore(15);
  for (size_t i = 0; i < 10; ++i) {
    semaphore.DoWork(1, [&](bool is_last) { EXPECT_FALSE(is_last); });
  }
  semaphore.DoWork(5, [&](bool is_last) { EXPECT_TRUE(is_last); });
}

TEST(Semaphore, Async) {
  internal::IsLastSemaphore o_semaphore(16);

  absl::Notification t1_done;
  absl::Notification t2_done;
  absl::Mutex mu;
  size_t thread_id = 0;
  auto thread_wait_flip = [&thread_id, &mu](size_t my_thread_id) {
    auto cond = [&] { return thread_id == my_thread_id; };
    mu.LockWhen(absl::Condition(&cond));
    mu.Unlock();
  };
  auto thread_flip = [&thread_id, &mu](size_t my_thread_id) {
    absl::MutexLock l(&mu);
    thread_id = 1 - thread_id;
  };

  std::unique_ptr<tsl::Thread> t1(
      tsl::Env::Default()->StartThread({}, "t1", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(0);
          o_semaphore.DoWork(1, [&](bool is_last) {
            thread_flip(0);
            EXPECT_FALSE(is_last);
          });
        }
      }));
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread({}, "t2", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(1);
          o_semaphore.DoWork(1, [&](bool is_last) {
            thread_flip(1);
            if (i == 7) {
              EXPECT_TRUE(is_last);
            } else {
              EXPECT_FALSE(is_last);
            }
          });
        }
      }));
}

}  // namespace
}  // namespace aux
