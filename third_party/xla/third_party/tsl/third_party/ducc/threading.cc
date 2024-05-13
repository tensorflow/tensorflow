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
#include "ducc/google/threading.h"

#include <thread>
#include <utility>

#include "ducc/src/ducc0/infra/threading.h"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace ducc0 {

namespace google {

namespace {

// Default shared global pool.  It is created on first use.
EigenThreadPool* GetGlobalThreadPoolSingleton() {
  static Eigen::ThreadPool* eigen_pool =
      new Eigen::ThreadPool(std::thread::hardware_concurrency());
  static EigenThreadPool* pool = new EigenThreadPool(*eigen_pool);
  return pool;
}

// Thread-local active pool for current execution.
ducc0::detail_threading::thread_pool*& GetActiveThreadPoolSingleton() {
  thread_local thread_pool* active_pool = nullptr;
  return active_pool;
}

}  // namespace
}  // namespace google

// Implementations required by ducc0.
namespace detail_threading {

// Missing variable used by DUCC threading.cc.
thread_local bool in_parallel_region = false;

thread_pool* set_active_pool(thread_pool* new_pool) {
  return std::exchange(ducc0::google::GetActiveThreadPoolSingleton(), new_pool);
}

thread_pool* get_active_pool() {
  thread_pool* pool = google::GetActiveThreadPoolSingleton();
  if (pool == nullptr) {
    // Set to use a global pool.  This may trigger threadpool creation.
    // Since the active pool is thread-local, this is thread-safe.
    pool = google::GetGlobalThreadPoolSingleton();
    set_active_pool(pool);
  }
  return pool;
}

}  // namespace detail_threading
}  // namespace ducc0