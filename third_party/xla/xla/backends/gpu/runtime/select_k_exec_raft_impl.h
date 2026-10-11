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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_RAFT_IMPL_H_
#define XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_RAFT_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "raft/core/device_mdspan.hpp"
#include "raft/core/mdspan_types.hpp"
#include "raft/core/resource/cuda_stream.hpp"
#include "raft/core/resource/device_memory_resource.hpp"
#include "raft/core/resources.hpp"
#include "raft/matrix/select_k.cuh"
#include "raft/matrix/select_k_types.hpp"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
namespace se = ::stream_executor;
using raft::matrix::SelectAlgo;

namespace raft_internal {

// Custom RMM memory resource backed by a fixed pre-allocated device buffer.
// Employs a monotonic bump allocator without dynamic allocations or
// deallocations.
class FixedBufferDeviceMemoryResource : public rmm::mr::device_memory_resource {
 public:
  FixedBufferDeviceMemoryResource() = default;

  void Reset(void* base_ptr, size_t capacity) {
    base_ptr_ = static_cast<char*>(base_ptr);
    capacity_ = capacity;
    offset_ = 0;
    VLOG(3) << "FixedBufferDeviceMemoryResource: reset with capacity "
            << capacity_;
  }

  size_t capacity() const { return capacity_; }
  size_t allocated_bytes() const { return offset_; }

 protected:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    // Ensure 256-byte alignment for CUDA / CUB operations.
    constexpr size_t kAlignment = 256;
    size_t current_addr = reinterpret_cast<size_t>(base_ptr_ + offset_);
    size_t padding = (kAlignment - (current_addr % kAlignment)) % kAlignment;
    size_t aligned_bytes = (bytes + kAlignment - 1) & ~(kAlignment - 1);
    if (offset_ + padding + aligned_bytes > capacity_) {
      throw rmm::bad_alloc(absl::StrCat(
          "FixedBufferDeviceMemoryResource: scratch capacity exceeded! "
          "Requested: ",
          bytes, " (aligned: ", aligned_bytes,
          "), remaining: ", capacity_ >= offset_ ? capacity_ - offset_ : 0,
          ", total capacity: ", capacity_));
    }
    offset_ += padding;
    void* ptr = base_ptr_ + offset_;
    offset_ += aligned_bytes;
    VLOG(3) << "FixedBufferDeviceMemoryResource: allocated " << aligned_bytes
            << " bytes at offset " << offset_ - aligned_bytes;
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes,
                     rmm::cuda_stream_view stream) noexcept override {
    // Intentional NO-OP: Memory lifetime is managed externally by XLA
    // BufferAssignment.
    VLOG(3) << "FixedBufferDeviceMemoryResource: deallocate " << bytes
            << " bytes at offset "
            << (reinterpret_cast<size_t>(ptr) -
                reinterpret_cast<size_t>(base_ptr_));
  }

 private:
  char* base_ptr_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
};

// RAII wrapper for RAFT resources bound to a CUDA stream
struct RaftStreamResource : public se::Stream::Resource {
  raft::resources res;
  std::shared_ptr<FixedBufferDeviceMemoryResource> fixed_buffer_mr;
  ~RaftStreamResource() override = default;

  // Factory to create a RaftStreamResource tied to a CUDA stream.
  // Sets up `raft::resources` with FixedBufferDeviceMemoryResource
  // and binds it to the provided stream.
  //
  // Args:
  //   cuda_stream: CUDA stream to bind.
  // Returns:
  //   Unique pointer to an initialized RaftStreamResource.
  static std::unique_ptr<RaftStreamResource> Create(cudaStream_t cuda_stream) {
    auto handle = std::make_unique<RaftStreamResource>();
    handle->fixed_buffer_mr =
        std::make_shared<FixedBufferDeviceMemoryResource>();
    raft::resource::set_workspace_resource(handle->res,
                                           handle->fixed_buffer_mr);
    raft::resource::set_cuda_stream(handle->res,
                                    rmm::cuda_stream_view{cuda_stream});
    VLOG(3) << "RaftStreamResource: created for stream " << cuda_stream;
    return handle;
  }
};

template <typename T>
SelectAlgo choose_select_k_algorithm(uint32_t rows, uint32_t cols, uint32_t k);

}  // namespace raft_internal

// Host-side entry point for raft select_k
template <typename T>
absl::Status select_k_exec(int device_ordinal,
                           se::DeviceAddressAllocator* allocator,
                           se::Stream* stream, se::DeviceAddressBase data_in,
                           se::DeviceAddressBase data_out,
                           se::DeviceAddressBase indices_out,
                           std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k,
                           se::DeviceAddressBase scratch_buffer) {
  // Pick the most suitable algorithm
  SelectAlgo algo = raft_internal::choose_select_k_algorithm<T>(batch, n, k);
  VLOG(3) << "select_k_exec_raft: "
          << "device_ordinal: " << device_ordinal << ", "
          << "allocator: " << allocator << ", "
          << "stream: " << stream << ", "
          << "data_in: " << data_in.opaque() << " (" << data_in.size() << "B)"
          << ", data_out: " << data_out.opaque() << " (" << data_out.size()
          << "B)"
          << ", indices_out: " << indices_out.opaque() << " ("
          << indices_out.size() << "B)"
          << ", scratch: " << scratch_buffer.opaque() << " ("
          << scratch_buffer.size() << "B)"
          << ", batch: " << batch << ", n: " << n << ", k: " << k
          << ", algo: " << algo;

  // Retrieve or create RAFT resource for this stream
  cudaStream_t cuda_stream =
      reinterpret_cast<cudaStream_t>(stream->platform_specific_handle().stream);
  TF_RET_CHECK(cuda_stream != nullptr)
      << "Failed to cast se::Stream to cudaStream_t.";
  raft_internal::RaftStreamResource* resContainer =
      stream->GetOrCreateResource<raft_internal::RaftStreamResource>(
          [cuda_stream] {
            return raft_internal::RaftStreamResource::Create(cuda_stream);
          });
  TF_RET_CHECK(resContainer != nullptr)
      << "Failed to create or retrieve RaftStreamResource";

  // Check the pre-allocated scratch buffer is valid.
  if (scratch_buffer.opaque() == nullptr || scratch_buffer.size() == 0) {
    return absl::InvalidArgumentError(
        "select_k_exec requires a valid non-empty scratch_buffer");
  }

  resContainer->fixed_buffer_mr->Reset(scratch_buffer.opaque(),
                                       scratch_buffer.size());
  auto reset_cleanup = absl::MakeCleanup(
      [resContainer] { resContainer->fixed_buffer_mr->Reset(nullptr, 0); });

  try {
    // Wrap raw device pointers in RAFT matrix views
    auto input_view =
        raft::make_device_matrix_view<const T, uint32_t, raft::row_major>(
            reinterpret_cast<const T*>(data_in.opaque()), batch, n);

    auto output_values_view =
        raft::make_device_matrix_view<T, uint32_t, raft::row_major>(
            reinterpret_cast<T*>(data_out.opaque()), batch, k);

    auto output_indices_view =
        raft::make_device_matrix_view<uint32_t, uint32_t, raft::row_major>(
            reinterpret_cast<uint32_t*>(indices_out.opaque()), batch, k);

    // Call RAFT select_k kernel
    raft::matrix::select_k<T, uint32_t>(
        resContainer->res, input_view,
        std::nullopt,  // d_input_indices can be omitted
        output_values_view, output_indices_view,
        /*select_min=*/false,
        /*sorted=*/true,
        /*algo=*/algo);

    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("select_k failed: ", e.what()));
  } catch (...) {
    return absl::InternalError("select_k failed with unknown exception");
  }
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_RAFT_IMPL_H_
