#include "xla/codegen/kernel_source.h"
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

#include "xla/codegen/kernel_spec.h"

namespace xla {

class KernelDefinitionBase {
 public:
  virtual ~KernelDefinitionBase() = default;

  virtual const KernelSpec& spec() const = 0;
  virtual const KernelSource& source() const = 0;
};

template <typename KernelSourceType>
class KernelDefinition final : public KernelDefinitionBase {
 public:
  struct Storage {
    KernelSpec spec;
    KernelSourceType source;
  };

  KernelDefinition(KernelSpec spec, KernelSourceType source)
      : storage_{std::move(spec), std::move(source)} {}

  KernelDefinition(KernelDefinition&&) = default;
  KernelDefinition& operator=(KernelDefinition&&) = default;

  const KernelSpec& spec() const override { return storage_.spec; }
  const KernelSourceType& source() const override { return storage_.source; }

  // Release the kernel definition implementation.
  // This is useful for backends that need to store the kernel definition
  // separately from the kernel spec.
  Storage ReleaseStorage() && { return std::move(storage_); }

 private:
  Storage storage_;
};

template <typename KernelSourceType>
KernelDefinition(KernelSpec, KernelSourceType)
    -> KernelDefinition<KernelSourceType>;

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_DEFINITION_H_
