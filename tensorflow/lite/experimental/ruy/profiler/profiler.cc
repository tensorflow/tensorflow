/* Copyright 2020 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/profiler/profiler.h"

#ifdef RUY_PROFILER
#include <atomic>
#include <chrono>  // NOLINT
#include <cstdio>
#include <cstdlib>
#include <thread>  // NOLINT
#include <vector>
#endif

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/profiler/treeview.h"

namespace ruy {
namespace profiler {

#ifdef RUY_PROFILER

ScopeProfile::ScopeProfile() { Start(); }
ScopeProfile::ScopeProfile(bool enable) {
  if (enable) {
    Start();
  }
}
ScopeProfile::~ScopeProfile() {
  if (!thread_) {
    return;
  }
  finishing_.store(true);
  thread_->join();
  Finish();
}

void ScopeProfile::Start() {
  {
    std::lock_guard<std::mutex> lock(*detail::GlobalsMutex());
    if (detail::GlobalIsProfilerRunning()) {
      fprintf(stderr, "FATAL: profiler already running!\n");
      abort();
    }
    detail::GlobalIsProfilerRunning() = true;
  }
  finishing_ = false;
  thread_.reset(new std::thread(&ScopeProfile::ThreadFunc, this));
}

void ScopeProfile::ThreadFunc() {
  while (!finishing_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> lock(*detail::GlobalsMutex());
    auto* thread_stacks = detail::GlobalAllThreadStacks();
    for (detail::ThreadStack* thread_stack : *thread_stacks) {
      Sample(*thread_stack);
    }
  }
}

void ScopeProfile::Sample(const detail::ThreadStack& thread_stack) {
  std::lock_guard<std::mutex> lock(thread_stack.Mutex());
  // Drop empty stacks.
  // This ensures that profiles aren't polluted by uninteresting threads.
  if (thread_stack.stack().size == 0) {
    return;
  }
  int sample_size = detail::GetBufferSize(thread_stack.stack());
  int old_buf_size = samples_buf_.size();
  samples_buf_.resize(old_buf_size + sample_size);
  detail::CopyToBuffer(thread_stack.stack(),
                       samples_buf_.data() + old_buf_size);
}

void ScopeProfile::Finish() {
  {
    std::lock_guard<std::mutex> lock(*detail::GlobalsMutex());
    if (!detail::GlobalIsProfilerRunning()) {
      fprintf(stderr, "FATAL: profiler is not running!\n");
      abort();
    }
    detail::GlobalIsProfilerRunning() = false;
  }
  if (user_treeview_) {
    user_treeview_->Populate(samples_buf_);
  } else {
    TreeView treeview;
    treeview.Populate(samples_buf_);
    Print(treeview);
  }
}

#endif  // RUY_PROFILER

}  // namespace profiler
}  // namespace ruy
