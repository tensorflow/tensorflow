/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_SAFE_STATIC_INIT_H_
#define XLA_PYTHON_SAFE_STATIC_INIT_H_

#include <atomic>
#include <type_traits>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/synchronization/mutex.h"
#include "nanobind/nanobind.h"

namespace xla {

// Serialized static variable initialization from the `init_fn` output.
// It avoids a deadlock between static variable initialization lock
// and a lock in the `init_fn` function.
// Expected signature of `init_fn` function: `std::unique_ptr<T> init_fn()`.
// We have the following assumptions on `init_fn` function:
// a) it can call python code and may release the GIL.
//
// Usage:
// Instead of incorrect code with potential deadlock
// static SomeType* p = [](){
//     // for example we call some python code using nanobind
//     nb::module_ numpy = nb::module_::import_("numpy");
//     auto np_int8 = nb::object(numpy.attr("int8"));
//     SomeType* obj = new SomeType(np_int8);
//     return obj;
// }();
//
// let us use SafeStatic
// static SafeStatic<SomeType> p;
// auto func = [](){
//     // for example we call some python code using nanobind
//     nb::module_ numpy = nb::module_::import_("numpy");
//     auto np_int8 = nb::object(numpy.attr("int8"));
//     return std::make_unique<SomeType>(np_int8);
// };
// SomeType& value = p.Get(func);
template <typename T>
class SafeStatic {
 public:
  template <typename F>
  T& Get(F init_fn) {
    // Opportunistic check outside the lock.
    if (T* result = output_.load()) {
      return *result;
    }
    // Locking must always be ordered, so we must release and reacquire
    // the gil because init_fn() may release the gil which forces us
    // to order mutex before gil.
    // In free-threading mode, the effect is the same but we are ordering
    // mutex before any critical sections because release_gil releases
    // all critical sections.
    nanobind::gil_scoped_release release_gil;
    absl::MutexLock lock(*mutex_);
    // Second check under the lock.
    if (T* result = output_.load()) {
      return *result;
    }
    nanobind::gil_scoped_acquire acquire_gil;
    output_.store(init_fn().release());
    return *output_.load();
  }

 private:
  absl::NoDestructor<absl::Mutex> mutex_{absl::kConstInit};
  std::atomic<T*> output_{nullptr};
};

// Google C style requires static objects be trivially destructible.
static_assert(std::is_trivially_destructible_v<SafeStatic<int>>);

}  // namespace xla

#endif  // XLA_PYTHON_SAFE_STATIC_INIT_H_
