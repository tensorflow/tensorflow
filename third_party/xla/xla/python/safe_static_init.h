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
#include <memory>

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
//     SomeType* obj = new SomeType(np_uint8);
//     return obj;
// }();
//
// let us use SafeStaticInit
// auto func = [](){
//     // for example we call some python code using nanobind
//     nb::module_ numpy = nb::module_::import_("numpy");
//     auto np_int8 = nb::object(numpy.attr("int8"));
//     return std::make_unique<SomeType>(np_uint8);
// }
// SomeType& p = SafeStaticInit<SomeType>(func);
template <typename T, typename F>
T& SafeStaticInit(F init_fn) {
  static absl::Mutex mutex;
  static std::atomic<T*> output{nullptr};
  // Opportunistic check outside the lock.
  if (T* result = output.load()) {
    return *result;
  }
  // Locking must always be ordered, so we must release and reacquire
  // the gil because init_fn() may release the gil which forces us
  // to order mutex before gil.
  // In free-threading mode, the effect is the same but we are ordering
  // mutex before any critical sections because release_gil releases
  // all critical sections.
  nanobind::gil_scoped_release release_gil;
  absl::MutexLock lock(&mutex);
  // Second check under the lock.
  if (T* result = output.load()) {
    return *result;
  }
  nanobind::gil_scoped_acquire acquire_gil;
  output.store(init_fn().release());
  return *output.load();
}

}  // namespace xla

#endif  // XLA_PYTHON_SAFE_STATIC_INIT_H_
