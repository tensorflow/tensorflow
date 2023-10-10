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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_THREAD_POOL_H_
#define TENSORFLOW_CORE_TFRT_UTILS_THREAD_POOL_H_

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_interface.h"

namespace tensorflow {
namespace tfrt_stub {

class TfThreadPool : public thread::ThreadPoolInterface {
 public:
  explicit TfThreadPool(const std::string& name, int num_threads)
      : underlying_threadpool_(tensorflow::Env::Default(), name, num_threads) {}

  void Schedule(std::function<void()> fn) override {
    underlying_threadpool_.Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    underlying_threadpool_.ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {
    underlying_threadpool_.AsEigenThreadPool()->Cancel();
  }

  int NumThreads() const override {
    return underlying_threadpool_.NumThreads();
  }

  int CurrentThreadId() const override {
    return underlying_threadpool_.CurrentThreadId();
  }

 private:
  tensorflow::thread::ThreadPool underlying_threadpool_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_THREAD_POOL_H_
