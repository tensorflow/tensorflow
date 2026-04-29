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
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/unbounded_work_queue.h"

namespace xla {

ThreadPoolAsyncWorkRunner::ThreadPoolAsyncWorkRunner(
    tsl::Env* env, absl::string_view name, int num_threads,
    const tsl::ThreadOptions& thread_options)
    : pool_(env, thread_options, std::string(name), num_threads) {}

void ThreadPoolAsyncWorkRunner::Execute(Task task) {
  // TSL ThreadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool_.Schedule([ptr = new absl::AnyInvocable<void() &&>(std::move(task))]() {
    std::move (*ptr)();
    delete ptr;
  });
}

UnboundedAsyncWorkRunner::UnboundedAsyncWorkRunner(
    absl::string_view name, const tsl::ThreadOptions& thread_options)
    : queue_(tsl::Env::Default(), std::string(name), thread_options) {}

void UnboundedAsyncWorkRunner::Execute(Task task) {
  // UnboundedWorkQueue expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  queue_.Schedule([ptr = new Task(std::move(task))] {
    std::move (*ptr)();
    delete ptr;
  });
}

std::unique_ptr<ThreadPoolAsyncWorkRunner> MakeThreadPoolAsyncWorkRunner(
    tsl::Env* env, absl::string_view name, int num_threads,
    const tsl::ThreadOptions& thread_options) {
  return std::make_unique<ThreadPoolAsyncWorkRunner>(env, name, num_threads,
                                                     thread_options);
}

std::unique_ptr<UnboundedAsyncWorkRunner> MakeUnboundedAsyncWorkRunner(
    absl::string_view name, const tsl::ThreadOptions& thread_options) {
  return std::make_unique<UnboundedAsyncWorkRunner>(name, thread_options);
}

}  // namespace xla
