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
#include "xla/tsl/platform/statusor.h"

namespace xla {

class KernelEmitterBase {
 public:
  virtual ~KernelEmitterBase() = default;

  virtual absl::StatusOr<std::unique_ptr<KernelDefinitionBase>>
  EmitBaseKernelDefinition() = 0;
};

// KernelEmitter is an API that emits kernel definition from a given input
// (i.e. it emits kernels compiled from HLO fusions).
template <typename KernelDefinitionType>
class KernelEmitter : public KernelEmitterBase {
 public:
  virtual ~KernelEmitter() = default;

  virtual absl::StatusOr<KernelDefinitionType> EmitKernelDefinition() = 0;

 private:
  absl::StatusOr<std::unique_ptr<KernelDefinitionBase>>
  EmitBaseKernelDefinition() final {
    TF_ASSIGN_OR_RETURN(KernelDefinitionType kernel_definition,
                        EmitKernelDefinition());
    return std::make_unique<KernelDefinitionType>(std::move(kernel_definition));
  }
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_EMITTER_H_
