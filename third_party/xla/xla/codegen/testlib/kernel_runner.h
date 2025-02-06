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

#ifndef XLA_CODEGEN_TESTLIB_KERNEL_RUNNER_H_
#define XLA_CODEGEN_TESTLIB_KERNEL_RUNNER_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape_util.h"

namespace xla {

// A base API for running XLA kernels. Intended for use in tests only.
class KernelRunner {
 public:
  virtual ~KernelRunner() = default;

  // Kernel argument is a non-owning view into the byte array on the host.
  // Kernel runners are responsible for transferring data from these buffers to
  // the device (if kernel is a device kernel, i.e. GPU kernel), and
  // transferring updates from the device back to these buffers.
  using Argument = absl::Span<std::byte>;

  // Calls the kernel with the given arguments.
  //
  // It's important to note that kernels (in contrast to HLO operations and XLA
  // programs) do not have parameters and results, they have buffer arguments
  // and they might read and write into the given buffers. Memory access kind is
  // available in the KernelSpec buffer uses.
  virtual absl::Status Call(absl::Span<const Argument> arguments) = 0;

  // Wrapper that takes in a set of Literals and converts them to Arguments.
  // Intentionally takes in raw pointers to allow for zero copy of the Literals
  // held by python.
  absl::Status Call(absl::Span<Literal*> literals);
};

// A collection of utility functions for working with KernelRunners.
class KernelRunnerUtil {
 public:
  // Creates a KernelRunner::Argument from a Literal.
  static KernelRunner::Argument CreateArgument(Literal& literal,
                                               const ShapeIndex& index = {});
};

}  // namespace xla

#endif  // XLA_CODEGEN_TESTLIB_KERNEL_RUNNER_H_
