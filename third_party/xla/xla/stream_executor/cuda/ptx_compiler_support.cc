/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/ptx_compiler_support.h"

namespace stream_executor {
bool IsLibNvPtxCompilerSupported() {
  // Libnvptxcompiler as a precompiled library is not compatible with MSan, so
  // we disable its support so that we at least can run some larger tests under
  // MSAN. This is not ideal because it means these tests will take different
  // code paths but the alternative would be not running them at all.
#ifdef MEMORY_SANITIZER
  return false;
#else
  return LIBNVPTXCOMPILER_SUPPORT;
#endif
}
}  // namespace stream_executor
