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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_KERNEL_NAME_TRACER_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_KERNEL_NAME_TRACER_H_

#include <memory>
#include <string>
#include <vector>

namespace xla::gpu {

// In some cases we need to know what exact kernel was used. It happens when we
// have no direct way to get this information from the HLO. For example, when we
// have a fusion with a custom call to cuBLAS or another third party library.
// This class allows to get the names of the kernels that were used.
class KernelNameTracer {
 public:
  static std::unique_ptr<KernelNameTracer> Create();

  virtual void start() = 0;

  // It should return the names of the kernels that were executed on GPU:0.
  virtual std::vector<std::string> stop() = 0;

  virtual ~KernelNameTracer() = default;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_KERNEL_NAME_TRACER_H_
