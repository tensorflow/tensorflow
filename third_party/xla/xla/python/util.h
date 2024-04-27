/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_UTIL_H_
#define XLA_PYTHON_UTIL_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/status.h"

namespace xla {

// Requests if given buffers are ready, awaits for results and returns OK if
// all of the buffers are ready or the last non-ok status.
Status AwaitBuffersReady(absl::Span<ifrt::Array* const> ifrt_arrays);

}  // namespace xla

#endif  // XLA_PYTHON_UTIL_H_
