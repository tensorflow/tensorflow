/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_DUCC_GOOGLE_THREADING_H_
#define THIRD_PARTY_DUCC_GOOGLE_THREADING_H_

#include "ducc/src/ducc0/infra/threading.h"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace ducc0 {
namespace google {

using std::size_t;

// Pseudo thread-pool for single-threaded execution.
class NoThreadPool : public ducc0::detail_threading::thread_pool {
 public:
  size_t nthreads() const override { return 1; }
  size_t adjust_nthreads(size_t nthreads_in) const override { return 1; };
  void submit(std::function<void()> work) override { work(); }
};

// Thread-pool wrapper around Eigen's ThreadPool.
class EigenThreadPool : public ducc0::detail_threading::thread_pool {
 public:
  EigenThreadPool(Eigen::ThreadPoolInterface& pool) : pool_{&pool} {}
  size_t nthreads() const override { return pool_->NumThreads(); }
  size_t adjust_nthreads(size_t nthreads_in) const override {
    // If called by a thread in the pool, return 1
    if (pool_->CurrentThreadId() >= 0) {
      return 1;
    } else if (nthreads_in == 0) {
      return pool_->NumThreads();
    }
    return std::min<size_t>(nthreads_in, pool_->NumThreads());
  };
  void submit(std::function<void()> work) override {
    pool_->Schedule(std::move(work));
  }

 private:
  Eigen::ThreadPoolInterface* pool_;
};

}  // namespace google
}  // namespace ducc0

#endif  // THIRD_PARTY_DUCC_GOOGLE_THREADING_H_
