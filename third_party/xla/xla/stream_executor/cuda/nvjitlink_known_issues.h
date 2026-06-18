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

#ifndef XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_KNOWN_ISSUES_H_
#define XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_KNOWN_ISSUES_H_

namespace stream_executor {

// Returns true if the loaded NvJitLink library is known to have bugs and
// shouldn't be used unconditionally. Returns false otherwise - also returns
// false if NvJitLink is not available.
bool LoadedNvJitLinkHasKnownIssues();

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_KNOWN_ISSUES_H_
