/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_PREFETCH_H_
#define TENSORFLOW_CORE_PLATFORM_PREFETCH_H_

#include "tsl/platform/prefetch.h"

namespace tensorflow {
namespace port {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::port::prefetch;
using ::tsl::port::PREFETCH_HINT_NTA;
using ::tsl::port::PREFETCH_HINT_T0;
using ::tsl::port::PrefetchHint;
// NOLINTEND(misc-unused-using-decls)
}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PREFETCH_H_
