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
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/raw_buffer.h"
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
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

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
  // `CreateBuffersForAsyncHostToDevice` uses a default layout.
  TF_ASSIGN_OR_RETURN(
      auto arr, ifrt_client->CreatePjRtArray(atm->RetrieveBuffer(0),
                                             /*has_custom_layout=*/false));
  results.arrays.push_back(std::move(arr));
  return results;
}

void CopyIntoDest(tsl::RCReference<ChunkDestination> dest,
                  tsl::RCReference<xla::ifrt::Array> arr, size_t xfer_size,
                  size_t buffer_id) {
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(arr->client())
          ->shared_ptr_pjrt_client();

  auto* array = static_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto scratch, AllocateAndMapPjrtMemory(pjrt_client, 1024 * 1024 * 16));
  TF_ASSERT_OK_AND_ASSIGN(auto src_work_units,
                          DmaCopyChunk::DivideBufferCopiesEvenly(
                              array->pjrt_buffers()[0], xfer_size, 0));
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
        std::move(src_work_units[i]),
        [&mu, &local_queue](PremappedCopierState* state,
                            absl::StatusOr<void*> buf,
                            const DmaCopyChunk& chunk) {
          CHECK_OK(buf.status());
          absl::MutexLock l(mu);
          local_queue.push_back(LocalQueueInfo{*buf, chunk.offset, chunk.size});
        });
  }
  for (size_t i = 0; i < src_work_units.size(); ++i) {
    auto cond = [&]() -> bool { return !local_queue.empty(); };
    mu.LockWhen(absl::Condition(&cond));
    auto state = local_queue.front();
    local_queue.pop_front();
    mu.unlock();
    TF_ASSERT_OK(
        dest->Put(state.buff, state.offset, state.size,
                  [cstate, buf = state.buff]() { cstate->ReturnBuffer(buf); }));
  }
}

absl::StatusOr<std::vector<int32_t>> FetchResult(
    tsl::RCReference<xla::ifrt::Array> arr, size_t result_size) {
  std::vector<int32_t> result;
  result.resize(result_size);
  TF_RETURN_IF_ERROR(
      arr->CopyToHostBuffer(result.data(), std::nullopt,
                            xla::ifrt::ArrayCopySemantics::kReuseInput)
          .Await());
  return result;
}

TEST(PremappedCopierState, FreeCycle) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(client.get())
          ->shared_ptr_pjrt_client();
  TF_ASSERT_OK_AND_ASSIGN(
      auto scratch, AllocateAndMapPjrtMemory(pjrt_client, 1024 * 1024 * 16));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, 4096);
  std::vector<void*> buffers_to_return;
  for (size_t i = 0; i < 2; ++i) {
    cstate->ScheduleCopy(
        {/*copy_fn=*/[](void* dst, int64_t offset,
                        int64_t transfer_size) -> xla::Future<> {
           return xla::Future<>(absl::OkStatus());
         },
         /*buffer_id=*/0,
         /*offset=*/100,
         /*size=*/100},
        [&buffers_to_return](PremappedCopierState* state,
                             absl::StatusOr<void*> buf,
                             const DmaCopyChunk& chunk) {
          CHECK_OK(buf.status());
          buffers_to_return.push_back(buf.value());
        });
  }
  class BufferReturner {
   public:
    explicit BufferReturner(absl::AnyInvocable<void() &&> on_done)
        : on_done_(std::move(on_done)) {}
    ~BufferReturner() { std::move(on_done_)(); }

   private:
    absl::AnyInvocable<void() &&> on_done_;
  };
  cstate->ScheduleCopy(
      {/*copy_fn=*/[buffer = std::make_unique<BufferReturner>(
                        [b = buffers_to_return[0], cstate]() {
                          cstate->ReturnBuffer(b);
                        })](void* dst, int64_t offset,
                            int64_t transfer_size) -> xla::Future<> {
         return xla::Future<>(absl::OkStatus());
       },
       /*buffer_id=*/0,
       /*offset=*/100,
       /*size=*/100},
      [buffer = std::make_unique<BufferReturner>(
           [b = buffers_to_return[1], cstate]() { cstate->ReturnBuffer(b); })](
          PremappedCopierState* state, absl::StatusOr<void*> buf,
          const DmaCopyChunk& chunk) {
        CHECK_OK(buf.status());
        state->ReturnBuffer(buf.value());
      });
}

TEST(PremappedCopierState, RoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  size_t xfer_size = 1024 * 1024;
  auto test_pattern = tests::CreateTestPattern(0, 16l * 1024 * 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, tests::CopyTestPatternToDevice(
                    client.get(), client->devices()[0], test_pattern));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_copy_plan,
      SetupTransferDestList(ShapeFromIfrt(arr), GetOtherDevice(arr),
                            GetIfrtClient(arr), xfer_size));
  CopyIntoDest(dest_copy_plan.dests[0], arr, xfer_size, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, FetchResult(dest_copy_plan.arrays[0], test_pattern.size()));
  EXPECT_EQ(result, test_pattern);
}

TEST(PremappedCopierState, RoundTripSlicedRaw) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  size_t xfer_size = 1024 * 1024;
  auto test_pattern = tests::CreateTestPattern(0, 16l * 1024 * 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_arr, tests::CopyTestPatternToDevice(
                         client.get(), client->devices()[0],
                         std::vector<int32_t>(test_pattern.size() * 3 / 2, 0)));
  TF_ASSERT_OK(dest_arr->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, tests::CopyTestPatternToDevice(
                    client.get(), client->devices()[0], test_pattern));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_raw_buffer,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
          static_cast<xla::ifrt::PjRtCompatibleArray*>(dest_arr.get())
              ->pjrt_buffers()[0]
              .get()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fut_and_dest,
      CreateSlicedRawBufferDest(
          dest_raw_buffer, dest_raw_buffer->GetOnDeviceSizeInBytes() / 3,
          dest_raw_buffer->GetOnDeviceSizeInBytes() * 2 / 3));
  CopyIntoDest(std::move(fut_and_dest.first), arr, xfer_size, 0);
  TF_ASSERT_OK(fut_and_dest.second.Await());

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          FetchResult(dest_arr, test_pattern.size() * 3 / 2));
  std::vector<int32_t> padded_test_pattern(test_pattern.size() / 2, 0);
  padded_test_pattern.insert(padded_test_pattern.end(), test_pattern.begin(),
                             test_pattern.end());
  EXPECT_EQ(result, padded_test_pattern);
}

TEST(PremappedCopierState, PoisonSlicedRaw) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto dest_arr, tests::CopyTestPatternToDevice(
                                             client.get(), client->devices()[0],
                                             std::vector<int32_t>(4096, 0)));
  TF_ASSERT_OK(dest_arr->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_raw_buffer,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
          static_cast<xla::ifrt::PjRtCompatibleArray*>(dest_arr.get())
              ->pjrt_buffers()[0]
              .get()));
  TF_ASSERT_OK_AND_ASSIGN(auto fut_and_dest,
                          CreateSlicedRawBufferDest(dest_raw_buffer, 0, 4096));
  fut_and_dest.first->Poison(absl::InternalError("Poisoning."));
  ASSERT_FALSE(fut_and_dest.second.Await().ok());
}

TEST(Semaphore, Basic) {
  internal::IsLastSemaphore semaphore(15);
  for (size_t i = 0; i < 10; ++i) {
    CHECK_OK(semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
      EXPECT_FALSE(is_last);
      return absl::OkStatus();
    }));
  }
  CHECK_OK(semaphore.DoWork(5, [&](bool is_last) -> absl::Status {
    EXPECT_TRUE(is_last);
    return absl::OkStatus();
  }));
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
    mu.unlock();
  };
  auto thread_flip = [&thread_id, &mu](size_t my_thread_id) {
    absl::MutexLock l(mu);
    thread_id = 1 - thread_id;
  };

  std::unique_ptr<tsl::Thread> t1(
      tsl::Env::Default()->StartThread({}, "t1", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(0);
          CHECK_OK(o_semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
            thread_flip(0);
            EXPECT_FALSE(is_last);
            return absl::OkStatus();
          }));
        }
      }));
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread({}, "t2", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(1);
          CHECK_OK(o_semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
            thread_flip(1);
            if (i == 7) {
              EXPECT_TRUE(is_last);
            } else {
              EXPECT_FALSE(is_last);
            }
            return absl::OkStatus();
          }));
        }
      }));
}

TEST(Semaphore, Poison) {
  internal::IsLastSemaphore o_semaphore(16);

  EXPECT_TRUE(o_semaphore.Poison());
  EXPECT_FALSE(o_semaphore.Poison());
  EXPECT_FALSE(o_semaphore.Poison());
}

}  // namespace
}  // namespace aux
