/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/bounded_executor.h"

#include <algorithm>
#include <atomic>

#include "absl/functional/bind_front.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace serving {
StatusOr<std::unique_ptr<BoundedExecutor>> BoundedExecutor::Create(
    const Options& options) {
  if (options.env == nullptr) {
    return errors::InvalidArgument("options.env must not be nullptr");
  }
  if (options.num_threads <= 0) {
    return errors::InvalidArgument("options.num_threads must be positive");
  }
  return absl::WrapUnique(new BoundedExecutor(options));
}

BoundedExecutor::BoundedExecutor(const Options& options) : options_(options) {
  InitWorker();
}

void BoundedExecutor::InitWorker() {
  for (int i = 0; i < options_.num_threads; i++) {
    std::unique_ptr<Thread> thread = absl::WrapUnique(
        options_.env->StartThread(options_.thread_options, options_.thread_name,
                                  [this]() { this->Run(); }));
    threads_.push_back(std::move(thread));
  }
}

BoundedExecutor::~BoundedExecutor() {
  {
    mutex_lock l(work_queue_mu_);
    // Enqueue an empty task (nullptr) to signal exit.
    // This way, each thread blocks on waiting a task, and exit run-loop
    // if task is nullptr.
    for (int i = 0; i < NumThreads(); i++) {
      work_queue_.push_back(nullptr);
      work_queue_cv_.notify_one();
    }
  }
  // Each thread will be joined in its destructor.
  threads_.clear();
}

void BoundedExecutor::Schedule(std::function<void()> func) {
  // use DCHECK so as not to introduce CHECK in prod code.
  DCHECK(func != nullptr) << "func is nullptr";
  mutex_lock l(work_queue_mu_);

  work_queue_.push_back(std::move(func));

  work_queue_cv_.notify_one();
}

int BoundedExecutor::NumThreads() const { return options_.num_threads; }

int BoundedExecutor::CurrentThreadId() const { return -1; }

void BoundedExecutor::Run() {
  while (true) {
    std::function<void()> func = nullptr;
    {
      mutex_lock l(work_queue_mu_);

      while (work_queue_.empty()) {
        work_queue_cv_.wait(l);
      }

      func = std::move(work_queue_.front());
      work_queue_.pop_front();
    }

    // Exit run-loop when func is nullptr.
    if (func != nullptr) {
      func();
    } else {
      break;
    }
  }
}
}  // namespace serving
}  // namespace tensorflow
