/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_POOL_H_
#define TENSORFLOW_COMPILER_XLA_POOL_H_

#include <functional>
#include <vector>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {

// Pool of values, which are created as needed and destroyed when the `Pool` is
// destroyed
template <typename T>
class Pool {
 public:
  struct Deleter {
    void operator()(T* ptr) { pool->Deallocate(ptr); }

    Pool<T>* pool;
  };

  // A pointer to a taken element of a `Pool` which returns it to the pool on
  // destruction
  using SmartPtr = std::unique_ptr<T, Deleter>;

  // Constructs a `Pool` with given factory function, which need not be
  // thread-safe.
  explicit Pool(std::function<std::unique_ptr<T>()> factory)
      : factory_(factory) {}

  explicit Pool() : Pool([]() { return MakeUnique<T>(); }) {}

  // Returns a pointer to a value in the pool, creating a new value if none is
  // free. The returned smart pointer returns the element to the pool on
  // destruction.
  //
  // This method is thread-safe.
  SmartPtr Allocate() {
    tensorflow::mutex_lock lock(mu_);
    T* ptr;
    if (!xs_.empty()) {
      ptr = std::move(xs_.back()).release();
      xs_.pop_back();
    } else {
      ptr = factory_().release();
    }
    Deleter del = {this};
    return std::unique_ptr<T, Deleter>(ptr, del);
  }

 private:
  // Puts a pointer to a value back into the pool, leaving it free for future
  // use.
  //
  // This method is thread-safe.
  void Deallocate(T* ptr) {
    tensorflow::mutex_lock lock(mu_);
    xs_.push_back(std::unique_ptr<T>(ptr));
  }

  const std::function<std::unique_ptr<T>()> factory_ GUARDED_BY(mu_);
  std::vector<std::unique_ptr<T>> xs_ GUARDED_BY(mu_);
  tensorflow::mutex mu_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_POOL_H_
