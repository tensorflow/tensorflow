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

#include "xla/pjrt/thread_pool_async_work_runner.h"

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

void EnqueueWork(tsl::thread::ThreadPool* pool,
                 absl::AnyInvocable<void() &&> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule(
      [ptr = new absl::AnyInvocable<void() &&>(std::move(callee))]() {
        std::move (*ptr)();
        delete ptr;
      });
}

// Enqueue to PjRtClient pool when all `values` are ready.
void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
    absl::AnyInvocable<void() &&> callee) {
  RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    EnqueueWork(pool, std::move(callee));
  });
}

class ThreadPoolAsyncWorkRunner : public AsyncWorkRunner {
 public:
  explicit ThreadPoolAsyncWorkRunner(tsl::thread::ThreadPool* pool)
      : pool_(pool) {}

  void Schedule(absl::AnyInvocable<void() &&> work) override {
    EnqueueWork(pool_, std::move(work));
  }

  void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void() &&> work) override {
    EnqueueWorkWhenReady(pool_, values, std::move(work));
  }

 private:
  tsl::thread::ThreadPool* pool_;
};

}  // namespace

std::unique_ptr<AsyncWorkRunner> MakeThreadPoolAsyncWorkRunner(
    tsl::thread::ThreadPool* pool) {
  return std::make_unique<ThreadPoolAsyncWorkRunner>(pool);
}

}  // namespace xla
