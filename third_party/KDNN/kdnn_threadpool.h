/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

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

#ifndef TENSORFLOW_THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_
#define TENSORFLOW_THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kdnn.hpp"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/threadpool.h"

namespace kdnn {

using tensorflow::thread::ThreadPool;

class KDNNThreadPool : public KDNN::Threading::ThreadpoolIface {
 public:
  KDNNThreadPool() = default;

  KDNNThreadPool(ThreadPool* thread_pool,
                int num_threads = -1)
      : thread_pool_(thread_pool),
      eigen_interface_(thread_pool->AsEigenThreadPool()) {
    set_num_and_max_threads(num_threads);
  }

  int GetNumThreads() const override { return num_threads_; }

  void ParallelFor(int n, int64_t cost_per_unit,
                const std::function<void(int, int)>& fn) override {
    thread_pool_->ParallelFor(n, cost_per_unit, fn);
  }

  bool IsInParallel() const override {
    return eigen_interface_->CurrentThreadId() != -1;
  }

  ~KDNNThreadPool() {}

 private:
  ThreadPool* thread_pool_ = nullptr;
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
  int num_threads_ = 1;
  inline void set_num_and_max_threads(int num_threads) {
    num_threads_ =
        num_threads == -1 ? eigen_interface_->NumThreads() : num_threads;
  }
};

}  // namespace kdnn

#endif  // TENSORFLOW_THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_
