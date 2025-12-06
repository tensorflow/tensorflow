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

#include <utility>

#include "xla/codegen/kernel_source.h"
#include "xla/codegen/kernel_spec.h"

namespace xla {

//===----------------------------------------------------------------------===//
// KernelDefinitionBase.
//===----------------------------------------------------------------------===//

// A base class for kernel definitions.
//
// KernelDefinition defines how the kernel must be executed via the `KernelSpec`
// and also contains the `KernelSource` that implements the kernel itself.
class KernelDefinitionBase {
 public:
  explicit KernelDefinitionBase(KernelSpec spec) : spec_(std::move(spec)) {}
  virtual ~KernelDefinitionBase() = default;

  const KernelSpec& spec() const { return spec_; }
  KernelSpec& spec() { return spec_; }

  virtual const KernelSource& source() const = 0;
  virtual KernelSource& source() = 0;

 protected:
  KernelDefinitionBase(KernelDefinitionBase&&) = default;
  KernelDefinitionBase& operator=(KernelDefinitionBase&&) noexcept = default;

 private:
  KernelSpec spec_;
};

//===----------------------------------------------------------------------===//
// KernelDefinition.
//===----------------------------------------------------------------------===//

// A concrete kernel definition implementation for the given kernel source type.
template <typename Source>
class KernelDefinition final : public KernelDefinitionBase {
  static_assert(std::is_base_of_v<KernelSource, Source>,
                "Source must be a subclass of KernelSource");

 public:
  KernelDefinition(KernelSpec spec, Source source)
      : KernelDefinitionBase(std::move(spec)), source_(std::move(source)) {}

  KernelDefinition(KernelDefinition&&) = default;
  KernelDefinition& operator=(KernelDefinition&&) noexcept = default;

  const Source& source() const final { return source_; }
  Source& source() final { return source_; }

  // Moves ownership of the source to the caller.
  Source TakeSource() && { return std::move(source_); }

 private:
  Source source_;
};

// Class template argument deduction guide for KernelDefinition.
template <typename Source>
KernelDefinition(KernelSpec, Source) -> KernelDefinition<Source>;

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_DEFINITION_H_
