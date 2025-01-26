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

#ifndef XLA_CODEGEN_KERNEL_EMITTER_H_
#define XLA_CODEGEN_KERNEL_EMITTER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/codegen/kernel_definition.h"

namespace xla {

// TODO(ezhulenev): Do we need virtual KernelEmitterContext in API?

// KernelEmitter is an API that emits kernel definition from a given input
// (i.e. it emits kernels compiled from HLO fusions).
class KernelEmitter {
 public:
  virtual ~KernelEmitter() = default;

  virtual absl::StatusOr<KernelDefinition> EmitKernelDefinition() = 0;
};

// A base class for backend-specific kernel emitters.
//
// Example: XLA:GPU backend kernel emitter.
//
//   class xla::gpu::GpuPlatform;
//
//   class xla::gpu::HloFusionEmitter :
//     public KernelEmitter<GpuPlatform, const HloFusionInstruction*>;
//
template <typename Platform, typename Operation>
class KernelEmitterBase {
 public:
  KernelEmitterBase(std::shared_ptr<Platform> platform, Operation operation)
      : platform_(std::move(platform)), operation_(std::move(operation)) {}

  const Operation& operation() const { return operation_; }
  const Platform& platform() const { return *platform_; }

 private:
  std::shared_ptr<Platform> platform_;
  Operation operation_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_EMITTER_H_
