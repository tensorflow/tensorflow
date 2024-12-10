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

#include "xla/stream_executor/cuda/nvjitlink_known_issues.h"

#include "xla/stream_executor/cuda/nvjitlink.h"

namespace stream_executor {

bool LoadedNvJitLinkHasKnownIssues() {
  // There is a memory leak in libnvjitlink from version 12.0 to 12.4.
  // The memory leak was fixed in CUDA Toolkit 12.4 Update 1, but we can't
  // distinguish between NvJitLink coming from CUDA Toolkit 12.4 and 12.4
  // Update 1. Therefore we only return true for 12.5 and higher to be on the
  // safe side.
  constexpr NvJitLinkVersion kMinVersionWithoutKnownIssues{12, 5};

  // Note that this needs to be a runtime version test because we load
  // LibNvJitLink as a dynamic library and the version might vary and not be the
  // same that we saw at compile time.
  return GetNvJitLinkVersion().value_or(NvJitLinkVersion{0, 0}) >=
         kMinVersionWithoutKnownIssues;
}

}  // namespace stream_executor
