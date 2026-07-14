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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CUSTOM_KERNEL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CUSTOM_KERNEL_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// CustomKernelThunk loads and executes kernels defined by a custom kernel
// (which in practice means hand written CUDA C++ kernel), instead of a kernel
// compiled by XLA and loaded from an executable source.
//
// Also implements Command so it can be recorded directly into command buffers.
class CustomKernelThunk : public Command {
 public:
  CustomKernelThunk(Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
                    const emitters::KernelArguments& kernel_arguments,
                    bool use_pdl = false,
                    std::vector<int64_t> zeroed_output_buffer_indices = {},
                    stream_executor::gpu::TmaMetadata tma_metadata = {});

  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  const CustomKernel& custom_kernel() const { return custom_kernel_; }

  const std::vector<ShapedSlice>& arguments() const { return args_; }

  absl::string_view custom_kernel_name() const { return custom_kernel_.name(); }

  const std::vector<bool>& written() const { return written_; }

  LaunchDimensions launch_dimensions() const {
    return LaunchDimensions(custom_kernel_.block_dims(),
                            custom_kernel_.thread_dims());
  }

  int64_t shmem_bytes() const { return custom_kernel_.shared_memory_bytes(); }

  bool use_pdl() const { return use_pdl_; }
  BufferUses buffer_uses() const override;

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<CustomKernelThunk>> FromProto(
      ThunkInfo thunk_info, const CustomKernelThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      const std::optional<se::KernelLoaderSpec::SymbolResolver>&
          symbol_resolver = std::nullopt);

 private:
  // Private constructor for deserialization.
  CustomKernelThunk(Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
                    std::vector<ShapedSlice> args, std::vector<bool> written,
                    std::vector<int64_t> zeroed_output_buffer_indices,
                    stream_executor::gpu::TmaMetadata tma_metadata,
                    bool use_pdl);

  // Holds the loaded kernel and the device addresses of its arguments.
  struct KernelWithArgs {
    se::Kernel* kernel;
    absl::InlinedVector<se::KernelArg, 4> buffer_args;
  };

  // Looks up the loaded kernel for the given executor and resolves the device
  // addresses of `args_`. Returns InternalError if Initialize() was not called
  // for this executor.
  absl::StatusOr<KernelWithArgs> GetKernelAndArgs(
      const BufferAllocations& buffer_allocations,
      se::StreamExecutor* executor) const;

  // Buffer slices passed to the kernel as arguments.
  std::vector<ShapedSlice> args_;

  // args_[i] is written iff (written_[i] == true).
  std::vector<bool> written_;

  CustomKernel custom_kernel_;

  // Loaded kernels for each `StreamExecutor`.
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);

  // Buffer indices that should be zeroed before the kernel is launched.
  std::vector<int64_t> zeroed_output_buffer_indices_;

  // Map of argument index to TmaDescriptor used to create arguments to the
  // kernel.
  stream_executor::gpu::TmaMetadata tma_metadata_;

  // Programmatic Dependent Launch.
  bool use_pdl_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_CUSTOM_KERNEL_THUNK_H_
