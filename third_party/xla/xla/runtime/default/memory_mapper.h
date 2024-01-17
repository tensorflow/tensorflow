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

#ifndef XLA_RUNTIME_DEFAULT_MEMORY_MAPPER_H_
#define XLA_RUNTIME_DEFAULT_MEMORY_MAPPER_H_

#include <errno.h>

namespace xla {
namespace runtime {

// Some syscalls can be interrupted by a signal handler; retry if that happens.
template <typename FunctionType>
static auto RetryOnEINTR(FunctionType func, decltype(func()) failure_value) {
  using ReturnType = decltype(func());
  ReturnType ret;
  do {
    ret = func();
  } while (ret == failure_value && errno == EINTR);
  return ret;
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_DEFAULT_MEMORY_MAPPER_H_
