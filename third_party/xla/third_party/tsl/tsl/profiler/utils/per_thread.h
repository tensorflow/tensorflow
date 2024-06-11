/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_UTILS_PER_THREAD_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_PER_THREAD_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace tsl {
namespace profiler {

// PerThread<T> provides a thread-local instance of T accessible to each
// application application thread, and provides the profiler thread access to
// all thread-local instances of T.
//
// Get() returns a thread-local instance of T that is created on first access.
//
// The thread-local instance is destroyed when the thread exits, unless
// StartRecording has been called. During recording, if a thread exits, its
// thread-local instance of T is kept alive until StopRecording is called.
template <typename T>
class PerThread {
 public:
  // Returns the thread-local instance of T.
  static T& Get() {
    static thread_local ThreadLocalPtr thread;
    return thread.Get();
  }

  // Starts keeping all thread-local instances of T alive.
  // Returns all instances of T from live threads.
  static std::vector<std::shared_ptr<T>> StartRecording() {
    return Registry::Get().StartRecording();
  }

  // Stops keeping thread-local instances of T alive.
  // Returns all instances of T from live and destroyed threads.
  static std::vector<std::shared_ptr<T>> StopRecording() {
    return Registry::Get().StopRecording();
  }

 private:
  // Prevent instantiation.
  PerThread() = delete;
  ~PerThread() = delete;

  // Singleton registry of all thread-local instances of T.
  class Registry {
   public:
    static Registry& Get() {
      static Registry* singleton = new Registry();
      return *singleton;
    }

    std::vector<std::shared_ptr<T>> StartRecording() {
      std::vector<std::shared_ptr<T>> threads;
      absl::MutexLock lock(&mutex_);
      threads.reserve(threads_.size());
      for (auto iter = threads_.begin(); iter != threads_.end(); ++iter) {
        threads.push_back(iter->first);
      }
      recording_ = true;
      return threads;
    }

    std::vector<std::shared_ptr<T>> StopRecording() {
      std::vector<std::shared_ptr<T>> threads;
      absl::MutexLock lock(&mutex_);
      threads.reserve(threads_.size());
      for (auto iter = threads_.begin(); iter != threads_.end();) {
        if (!iter->second) {  // The creator thread is dead.
          threads.push_back(std::move(iter->first));
          threads_.erase(iter++);
        } else {
          threads.push_back(iter->first);
          ++iter;
        }
      }
      recording_ = false;
      return threads;
    }

    void Register(std::shared_ptr<T> thread) {
      absl::MutexLock lock(&mutex_);
      threads_.insert_or_assign(std::move(thread), true);
    }

    void Unregister(const std::shared_ptr<T>& thread) {
      absl::MutexLock lock(&mutex_);
      if (!recording_) {
        threads_.erase(thread);
      } else {
        if (auto it = threads_.find(thread); it != threads_.end()) {
          it->second = false;
        }
      }
    }

   private:
    Registry() = default;

    Registry(const Registry&) = delete;
    void operator=(const Registry&) = delete;

    absl::Mutex mutex_;
    absl::flat_hash_map<std::shared_ptr<T>, bool> threads_
        ABSL_GUARDED_BY(mutex_);
    bool recording_ ABSL_GUARDED_BY(mutex_) = false;
  };

  // Thread-local instance of T.
  class ThreadLocalPtr {
   public:
    ThreadLocalPtr() : ptr_(std::make_shared<T>()) {
      Registry::Get().Register(ptr_);
    }

    ~ThreadLocalPtr() { Registry::Get().Unregister(ptr_); }

    T& Get() { return *ptr_; }

   private:
    std::shared_ptr<T> ptr_;
  };
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_PER_THREAD_H_
