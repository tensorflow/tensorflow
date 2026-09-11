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
#ifndef XLA_BACKENDS_GPU_RUNTIME_INTERNABLE_KERNEL_LOADER_SPEC_H_
#define XLA_BACKENDS_GPU_RUNTIME_INTERNABLE_KERNEL_LOADER_SPEC_H_

#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/internable_kernel_loader_spec.pb.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {

// Wraps a `stream_executor::KernelLoaderSpec` with an optional pointer to a
// `KernelSpecTable`.
//
// When `table` is non-null, `ToProto()` interns the spec into the table and
// emits only the table index (`kernel_spec_index`). When `table` is null,
// `ToProto()` emits the inline `KernelLoaderSpecProto` (`kernel_spec`).
//
// The caller is responsible for ensuring that `*table` outlives this object.
class InternableKernelLoaderSpec {
 public:
  explicit InternableKernelLoaderSpec(
      stream_executor::KernelLoaderSpec spec,
      KernelSpecTable* absl_nullable table = nullptr)
      : spec_(std::move(spec)), table_(table) {}

  const stream_executor::KernelLoaderSpec& kernel_spec() const { return spec_; }
  stream_executor::KernelLoaderSpec& mutable_kernel_spec() { return spec_; }

  KernelSpecTable* absl_nullable kernel_spec_table() const { return table_; }
  void set_kernel_spec_table(KernelSpecTable* absl_nullable table) {
    table_ = table;
  }

  absl::StatusOr<InternableKernelLoaderSpecProto> ToProto() const;

  static absl::StatusOr<InternableKernelLoaderSpec> FromProto(
      const InternableKernelLoaderSpecProto& proto,
      std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>
          symbol_resolver = std::nullopt,
      const KernelSpecTable* absl_nullable table = nullptr);

 private:
  stream_executor::KernelLoaderSpec spec_;
  KernelSpecTable* absl_nullable table_ = nullptr;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_INTERNABLE_KERNEL_LOADER_SPEC_H_
