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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_FACTORY_H_
#define XLA_STREAM_EXECUTOR_KERNEL_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Creates Kernels from kernel specifications.
class KernelFactory {
 public:
  // Creates kernel on a given executor from a given kernel specification.
  static inline absl::StatusOr<std::unique_ptr<Kernel>> Create(
      StreamExecutor *executor, const MultiKernelLoaderSpec &spec) {
    return executor->LoadKernel(spec);
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_FACTORY_H_
