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

#ifndef XLA_CODEGEN_KERNEL_DEFINITION_H_
#define XLA_CODEGEN_KERNEL_DEFINITION_H_

#include <memory>
#include <utility>

#include "xla/codegen/kernel_source.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

class KernelDefinition {
 public:
  KernelDefinition(KernelSpec spec, std::unique_ptr<KernelSource> source)
      : spec_(std::move(spec)), source_(std::move(source)) {}

  KernelDefinition(KernelDefinition&& other) = default;
  KernelDefinition& operator=(KernelDefinition&& other) noexcept = default;

  const KernelSpec& spec() const { return spec_; }
  const KernelSource& source() const {
    CHECK_NOTNULL(source_);  // CRASH OK - use after move.
    return *source_;
  }

  // Release the kernel definition.
  // This is useful for backends that need to store the kernel definition
  // separately from the kernel spec.
  std::pair<KernelSpec, std::unique_ptr<KernelSource>> release() && {
    return std::make_pair(std::move(spec_), std::move(source_));
  }

 private:
  KernelSpec spec_;
  std::unique_ptr<KernelSource> source_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_DEFINITION_H_
