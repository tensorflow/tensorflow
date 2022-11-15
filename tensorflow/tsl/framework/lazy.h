/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_FRAMEWORK_LAZY_H_
#define TENSORFLOW_TSL_FRAMEWORK_LAZY_H_

#include <functional>

#include "tensorflow/tsl/platform/mutex.h"

namespace tsl {

// A thread safe, in-place lazy template.
template <typename T>
class Lazy {
 public:
  template <class... Args>
  explicit Lazy(Args&&... args)
      : initialized_(false),
        initialize_([args = std::make_tuple(std::forward<Args>(args)...)](
                        Lazy<T>* self) {
          self->DoInitialize(std::index_sequence_for<Args...>(),
                             std::move(args));
        }) {}

  ~Lazy() {
    if (initialized_) {
      val_.~T();
    }
  }

  bool IsInitialized() const { return initialized_; }

  T& Get() {
    if (!initialized_) {
      initialize_(this);
    }
    return val_;
  }

  T* TryGet() {
    if (!initialized_) {
      return nullptr;
    }
    return &val_;
  }

 private:
  union {
    char place_holder_;
    T val_;
  };

  template <std::size_t... I, class Tuple>
  void DoInitialize(std::index_sequence<I...>, Tuple args) {
    mutex_lock l(initialize_mutex_);
    if (!initialized_) {
      new (&val_) T(std::get<I>(args)...);
      initialized_ = true;
    }
  }

  mutex initialize_mutex_;
  std::atomic<bool> initialized_;
  std::function<void(Lazy<T>*)> initialize_;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_LAZY_H_
