/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/runtime/internable_kernel_loader_spec.h"

#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/internable_kernel_loader_spec.pb.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {

namespace se = ::stream_executor;

absl::StatusOr<InternableKernelLoaderSpecProto>
InternableKernelLoaderSpec::ToProto() const {
  InternableKernelLoaderSpecProto proto;
  if (table_ != nullptr) {
    proto.set_kernel_spec_index(table_->Intern(spec_));
    return proto;
  }
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_kernel_spec(), spec_.ToProto());
  return proto;
}

absl::StatusOr<InternableKernelLoaderSpec>
InternableKernelLoaderSpec::FromProto(
    const InternableKernelLoaderSpecProto& proto,
    std::optional<se::KernelLoaderSpec::SymbolResolver> symbol_resolver,
    const KernelSpecTable* absl_nullable table) {
  switch (proto.spec_case()) {
    case InternableKernelLoaderSpecProto::kKernelSpec: {
      ABSL_ASSIGN_OR_RETURN(se::KernelLoaderSpec spec,
                       se::KernelLoaderSpec::FromProto(proto.kernel_spec(),
                                                       symbol_resolver));
      return InternableKernelLoaderSpec(std::move(spec));
    }
    case InternableKernelLoaderSpecProto::kKernelSpecIndex: {
      if (table == nullptr) {
        return absl::InvalidArgumentError(
            "InternableKernelLoaderSpecProto references kernel_spec_index, "
            "but no KernelSpecTable was provided.");
      }
      ABSL_ASSIGN_OR_RETURN(const se::KernelLoaderSpec* spec,
                       table->Get(proto.kernel_spec_index()));
      return InternableKernelLoaderSpec(*spec);
    }
    case InternableKernelLoaderSpecProto::SPEC_NOT_SET:
      return absl::InvalidArgumentError(
          "InternableKernelLoaderSpecProto has neither kernel_spec nor "
          "kernel_spec_index set.");
  }
}

}  // namespace xla::gpu
