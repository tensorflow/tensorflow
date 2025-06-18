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

#ifndef XLA_PJRT_PROFILING_PROFILING_CONTEXT_H_
#define XLA_PJRT_PROFILING_PROFILING_CONTEXT_H_

#include <memory>

#include "llvm/Support/ExtensibleRTTI.h"

namespace xla {

// Interface for a profiling contex that can be passed between device execute
// threads.
class ProfilingContext
    : public llvm::RTTIExtends<ProfilingContext, llvm::RTTIRoot> {
 public:
  ProfilingContext() = default;
  ~ProfilingContext() override = default;

  // For llvm::RTTIExtends.
  static char ID;  // NOLINT
};

// Factory function for creating a ProfilingContext object with the current
// thread's profiling context.
std::unique_ptr<ProfilingContext> CreateProfilingContext();

// Scoped object that sets the current thread's profiling context to switch_to
// until the object is destroyed.
class WithProfilingContext
    : public llvm::RTTIExtends<WithProfilingContext, llvm::RTTIRoot> {
 public:
  WithProfilingContext() = default;
  ~WithProfilingContext() override = default;

  WithProfilingContext(const WithProfilingContext&) = delete;
  WithProfilingContext& operator=(const WithProfilingContext&) = delete;

  // For llvm::RTTIExtends.
  static char ID;  // NOLINT
};

// Factory function for creating a WithProfilingContext object that installs
// the switch_to context on the current thread.
std::unique_ptr<WithProfilingContext> CreateWithProfilingContext(
    ProfilingContext* switch_to);

}  // namespace xla

#endif  // XLA_PJRT_PROFILING_PROFILING_CONTEXT_H_
