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

#include <memory>

#include "absl/synchronization/mutex.h"

namespace xla {

// Serialized static variable initialization from the `init_fn` output.
// It avoids a deadlock between static variable initialization lock
// and a lock in the `init_fn` function.
// Expected signature of `init_fn` function: `std::unique_ptr<T> init_fn()`.
// We have the following assumptions on `init_fn` function:
// a) it can call python code and may release the GIL.
// When the function is called we do not hold any non-GIL or
// free-threading mutex.
// b) function can be called multiple times if invoked concurrently,
// but the output from all but one will be discarded.
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
  static T* output = nullptr;
  {
    absl::MutexLock lock(&mutex);
    if (output) {
      return *output;
    }
  }
  std::unique_ptr<T> p = init_fn();
  absl::MutexLock lock(&mutex);
  if (!output) {
    output = p.release();
  }
  return *output;
}

}  // namespace xla

#endif  // XLA_PYTHON_SAFE_STATIC_INIT_H_
