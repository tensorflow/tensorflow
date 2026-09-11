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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/stream_executor/kernel_spec.pb.h"

namespace xla::gpu {

// A deduplicated table of kernel loader specs for one executable.
//
// A kernel that is invoked several times in a module would otherwise be
// serialized once per invocation, because every `CustomKernelThunkProto`
// carries its own copy of the kernel binary. The table holds each distinct
// spec once; thunks refer to it by index.
//
// This is purely a serialization level construct: it holds protos, not live
// `se::KernelLoaderSpec`s, and therefore imposes no lifetime constraints on
// thunks.
//
// Indices are assigned in order of first insertion. Callers that insert in a
// deterministic order therefore get a deterministic table and deterministic
// indices.
//
// Not thread safe.
class KernelSpecTable {
 public:
  // Returns the index of `spec`, inserting it at the end of the table if it is
  // not present yet.
  int32_t Intern(stream_executor::KernelLoaderSpecProto spec);

  // Appends `spec` to the end of the table without deduplicating it, and
  // returns its index. Use this when rebuilding a table from its serialized
  // form, where indices must be preserved exactly as they were stored.
  int32_t Append(stream_executor::KernelLoaderSpecProto spec);

  bool empty() const { return specs_.empty(); }
  size_t size() const { return specs_.size(); }

  // Returns the spec at `index`, or an error if `index` is out of range.
  absl::StatusOr<const stream_executor::KernelLoaderSpecProto*> Get(
      int32_t index) const;

  // The table contents, ordered by index. This is what gets serialized.
  absl::Span<const stream_executor::KernelLoaderSpecProto> specs() const {
    return specs_;
  }

 private:
  std::vector<stream_executor::KernelLoaderSpecProto> specs_;

  // Lookup acceleration only. Never iterated and never serialized, so its
  // nondeterministic iteration order cannot leak into the output. Maps a
  // non-cryptographic hash of a spec to the indices of all specs with that
  // hash; candidates are confirmed with an exact comparison, so collisions
  // cost a comparison but cannot produce a wrong result.
  absl::flat_hash_map<uint64_t, absl::InlinedVector<int32_t, 1>> index_;
};

// Rewrites every `CustomKernelProto` reachable from `thunk`, including the ones
// in nested thunk sequences, to reference `table` by index instead of inlining
// its kernel loader spec.
//
// Custom kernels that already reference the table are left alone, so this is
// idempotent.
absl::Status InternKernelSpecs(ThunkProto& thunk, KernelSpecTable& table);

// Inverse of `InternKernelSpecs`: replaces every table reference reachable from
// `thunk` with a copy of the referenced spec.
//
// Custom kernels that already carry an inline spec are left alone, so this is
// safe to run unconditionally on protos produced by a compiler that did not
// deduplicate.
absl::Status InlineKernelSpecs(ThunkProto& thunk, const KernelSpecTable& table);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_KERNEL_SPEC_TABLE_H_
