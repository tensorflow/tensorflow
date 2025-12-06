/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFERS_CHECKSUM_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFERS_CHECKSUM_THUNK_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/gpu/buffer_debug_xor_checksum_kernel.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

class BuffersDebugChecksumThunk : public Thunk {
 public:
  explicit BuffersDebugChecksumThunk(
      ThunkInfo info, BufferAllocation::Slice log_slice,
      ThunkId checked_thunk_id,
      // buffer_idx => buffer slice
      absl::flat_hash_map<size_t, BufferAllocation::Slice>
          checked_thunk_buffers,
      bool runs_before_checked_thunk,
      std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store)
      : Thunk(Thunk::Kind::kBuffersDebugChecksum, std::move(info)),
        log_slice_(log_slice),
        metadata_store_(std::move(metadata_store)),
        checked_thunk_id_(checked_thunk_id),
        checked_thunk_buffers_(std::move(checked_thunk_buffers)),
        runs_before_checked_thunk_(runs_before_checked_thunk) {}

  absl::Status Initialize(const InitializeParams& params) override
      ABSL_LOCKS_EXCLUDED(kernels_mutex_);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override
      ABSL_LOCKS_EXCLUDED(kernels_mutex_);

  std::string ToString(int indent) const override;

  BufferUses buffer_uses() const override {
    // Intentionally left empty to not checksum the checksumming thunk.
    return {};
  }

  const absl::flat_hash_map<size_t, BufferAllocation::Slice>& buffer_slices()
      const {
    return checked_thunk_buffers_;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const BuffersDebugChecksumThunk& thunk) {
    absl::Format(&sink, "BuffersDebugChecksumThunk{buffers=%s}",
                 absl::StrJoin(thunk.checked_thunk_buffers_, ", ",
                               [](std::string* out, const auto& buffer) {
                                 const auto& [id, slice] = buffer;
                                 absl::StrAppend(out, id, "=",
                                                 slice.ToString());
                               }));
  }

 private:
  absl::Mutex kernels_mutex_;
  // Each loaded kernel is associated with a specific device (represented by its
  // StreamExecutor).
  //
  // ExecuteOnStream implementation requires pointer stability of values, hence
  // unique_ptr.
  absl::flat_hash_map<
      stream_executor::StreamExecutor*,
      std::unique_ptr<
          stream_executor::gpu::BufferDebugXorChecksumKernel::KernelType>>
      kernels_ ABSL_GUARDED_BY(kernels_mutex_);

  BufferAllocation::Slice log_slice_;
  std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store_;
  ThunkId checked_thunk_id_;
  absl::flat_hash_map<size_t, BufferAllocation::Slice> checked_thunk_buffers_;
  bool runs_before_checked_thunk_;
  std::atomic<size_t> execution_count_ = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFERS_CHECKSUM_THUNK_H_
