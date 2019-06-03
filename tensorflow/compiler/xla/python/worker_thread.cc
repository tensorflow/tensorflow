/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/worker_thread.h"

namespace xla {

WorkerThread::WorkerThread(tensorflow::Env* env, const std::string& name) {
  thread_.reset(env->StartThread(tensorflow::ThreadOptions(), name,
                                 [this]() { WorkLoop(); }));
}

WorkerThread::~WorkerThread() {
  absl::MutexLock lock(&mu_);
  work_queue_.push(nullptr);
}

void WorkerThread::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(std::move(fn));
}

bool WorkerThread::WorkAvailable() { return !work_queue_.empty(); }

void WorkerThread::WorkLoop() {
  while (true) {
    std::function<void()> fn;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &WorkerThread::WorkAvailable));
      fn = std::move(work_queue_.front());
      work_queue_.pop();
    }
    if (!fn) {
      return;
    }
    fn();
  }
}

}  // namespace xla
