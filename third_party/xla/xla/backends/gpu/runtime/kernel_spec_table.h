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
#ifndef XLA_BACKENDS_GPU_RUNTIME_KERNEL_SPEC_TABLE_H_
#define XLA_BACKENDS_GPU_RUNTIME_KERNEL_SPEC_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.pb.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {

// A deduplicated table of kernel loader specs for one executable.
//
// A kernel that is invoked several times in a module would otherwise be
// serialized once per invocation, because every `CustomKernelThunkProto`
// carries its own copy of the kernel binary. The table holds each distinct
// spec once; `InternableKernelLoaderSpec`s refer to it by index when
// serialized.
//
// Indices are assigned in order of first insertion. Callers that insert in a
// deterministic order therefore get a deterministic table and deterministic
// indices.
//
// Not thread safe.
class KernelSpecTable {
 public:
  // Returns the index of `spec`, inserting it at the end of the table if it is
  // not present yet. Returns an error if `spec` is not serializable.
  absl::StatusOr<int32_t> Intern(stream_executor::KernelLoaderSpec spec);

  bool empty() const { return specs_.empty(); }
  size_t size() const { return specs_.size(); }

  // Returns the spec at `index`, or an error if `index` is out of range.
  absl::StatusOr<stream_executor::KernelLoaderSpec> Get(int32_t index) const;

  absl::StatusOr<KernelSpecTableProto> ToProto() const;

  static absl::StatusOr<KernelSpecTable> FromProto(
      const KernelSpecTableProto& proto,
      const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
          symbol_resolver = std::nullopt);

 private:
  struct KernelLoaderSpecHash {
    size_t operator()(const stream_executor::KernelLoaderSpec& spec) const;
  };
  struct KernelLoaderSpecEq {
    bool operator()(const stream_executor::KernelLoaderSpec& a,
                    const stream_executor::KernelLoaderSpec& b) const;
  };

  int32_t Append(stream_executor::KernelLoaderSpec spec);

  std::vector<stream_executor::KernelLoaderSpec> specs_;

  // Lookup acceleration only. Never iterated and never serialized, so its
  // nondeterministic iteration order cannot leak into the output.
  absl::flat_hash_map<stream_executor::KernelLoaderSpec, int32_t,
                      KernelLoaderSpecHash, KernelLoaderSpecEq>
      index_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_KERNEL_SPEC_TABLE_H_
