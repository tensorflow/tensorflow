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
#include "absl/strings/string_view.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_source.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

//===----------------------------------------------------------------------===//
// KernelEmitterBase.
//===----------------------------------------------------------------------===//

// A base class for emitting XLA kernels.
class KernelEmitterBase {
 public:
  KernelEmitterBase() = default;
  virtual ~KernelEmitterBase() = default;

  virtual absl::string_view name() const = 0;

  virtual absl::StatusOr<std::unique_ptr<KernelDefinitionBase>>
  EmitKernelDefinitionBase() = 0;

 protected:
  KernelEmitterBase(KernelEmitterBase&&) = default;
  KernelEmitterBase& operator=(KernelEmitterBase&&) = default;
};

//===----------------------------------------------------------------------===//
// KernelEmitter.
//===----------------------------------------------------------------------===//

// KernelEmitter is an API that emits kernel definition from a given input
// (i.e. it emits kernels compiled from HLO fusions).
template <typename Source>
class KernelEmitter : public KernelEmitterBase {
 public:
  static_assert(std::is_base_of_v<KernelSource, Source>,
                "Source must be a subclass of KernelSource");

  using KernelDefinition = ::xla::KernelDefinition<Source>;
  virtual absl::StatusOr<KernelDefinition> EmitKernelDefinition() = 0;

 private:
  absl::StatusOr<std::unique_ptr<KernelDefinitionBase>>
  EmitKernelDefinitionBase() final {
    TF_ASSIGN_OR_RETURN(auto kernel_definition, EmitKernelDefinition());
    return std::make_unique<KernelDefinition>(std::move(kernel_definition));
  }
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_EMITTER_H_
