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

// Simple RAII wrapper to manage temporary device memory allocations
class OwningScratchAllocator {
 public:
  OwningScratchAllocator(int device_ordinal,
                         se::DeviceAddressAllocator* allocator)
      : device_ordinal_(device_ordinal), allocator_(allocator) {}

  OwningScratchAllocator(OwningScratchAllocator&&) = default;
  OwningScratchAllocator& operator=(OwningScratchAllocator&&) = default;

  // Allocate memory and track ownership
  absl::StatusOr<se::DeviceAddress<uint8_t>> AllocateBytes(int64_t byte_size) {
    ABSL_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> buffer,
                     allocator_->Allocate(device_ordinal_, byte_size,
                                          /*retry_on_failure=*/false));

    se::DeviceAddress<uint8_t> res = *buffer;
    void* raw_ptr = res.opaque();
    buffers_.emplace(raw_ptr, std::move(buffer));
    return res;
  }

  // Deallocate tracked memory; safe no-op if pointer not found
  absl::Status DeallocateBytes(void* ptr) noexcept {
    auto it = buffers_.find(ptr);
    if (it != buffers_.end()) {
      buffers_.erase(it);  // RAII frees memory
      return absl::OkStatus();
    }
    return absl::NotFoundError("Pointer not found");
  }

  se::DeviceAddressAllocator* get_allocator() const { return allocator_; }

  void set_allocator(se::DeviceAddressAllocator* allocator) {
    allocator_ = allocator;
  }

 private:
  int device_ordinal_;
  se::DeviceAddressAllocator* allocator_;
  // key = raw device pointer, value = owning memory object
  absl::flat_hash_map<void*, se::ScopedDeviceAddress<uint8_t>> buffers_;
};

// Custom RMM memory resource backed by StreamExecutor allocator
class XlaDeviceMemoryResource : public rmm::mr::device_memory_resource {
 public:
  XlaDeviceMemoryResource(int device_ordinal,
                          se::DeviceAddressAllocator* allocator)
      : scratch_allocator_(device_ordinal, allocator) {}

  se::DeviceAddressAllocator* get_allocator() const {
    return scratch_allocator_.get_allocator();
  }

  void set_allocator(se::DeviceAddressAllocator* allocator) {
    scratch_allocator_.set_allocator(allocator);
  }

 protected:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    auto mem = scratch_allocator_.AllocateBytes(bytes);
    if (!mem.ok()) {
      // RMM expects exceptions
      throw rmm::bad_alloc(std::string(mem.status().ToString()));
    }
    return mem->opaque();
  }

  void do_deallocate(void* ptr, std::size_t bytes,
                     rmm::cuda_stream_view stream) noexcept override {
    auto status = scratch_allocator_.DeallocateBytes(ptr);
    if (!status.ok()) {
      // do_deallocate should be noexcept. Don’t throw; just log.
      LOG(ERROR) << "Scratch Deallocation failed: " << status;
    }
  }

 private:
  OwningScratchAllocator scratch_allocator_;
};

// RAII wrapper for RAFT resources bound to a CUDA stream
struct RaftStreamResource : public se::Stream::Resource {
  raft::resources res;
  std::shared_ptr<XlaDeviceMemoryResource> xla_dev_mem_res;
  ~RaftStreamResource() override = default;

  // Factory to create a RaftStreamResource tied to a CUDA stream.
  // Sets up `raft::resources` with a custom XlaDeviceMemoryResource
  // using the given allocator and binds it to the provided stream.
  //
  // Args:
  //   device_ordinal: Device index.
  //   allocator: StreamExecutor memory allocator.
  //   cuda_stream: CUDA stream to bind.
  // Returns:
  //   Unique pointer to an initialized RaftStreamResource.
  static std::unique_ptr<RaftStreamResource> Create(
      int device_ordinal, se::DeviceAddressAllocator* allocator,
      cudaStream_t cuda_stream) {
    // Assign our custom AllocatorForRaft for this device
    auto handle = std::make_unique<RaftStreamResource>();
    handle->xla_dev_mem_res =
        std::make_shared<XlaDeviceMemoryResource>(device_ordinal, allocator);
    raft::resource::set_workspace_resource(handle->res,
                                           handle->xla_dev_mem_res);
    // Set Cuda Stream
    raft::resource::set_cuda_stream(handle->res,
                                    rmm::cuda_stream_view{cuda_stream});
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
                           std::uint32_t k) {
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
          << ", batch: " << batch << ", n: " << n << ", k: " << k
          << ", algo: " << algo;

  // Retrieve or create RAFT resource for this stream
  cudaStream_t cuda_stream =
      reinterpret_cast<cudaStream_t>(stream->platform_specific_handle().stream);
  TF_RET_CHECK(cuda_stream != nullptr)
      << "Failed to cast se::Stream to cudaStream_t.";
  raft_internal::RaftStreamResource* resContainer =
      stream->GetOrCreateResource<raft_internal::RaftStreamResource>(
          [device_ordinal, allocator, cuda_stream] {
            return raft_internal::RaftStreamResource::Create(
                device_ordinal, allocator, cuda_stream);
          });
  TF_RET_CHECK(resContainer != nullptr)
      << "Failed to create or retrieve RaftStreamResource";

  // resContainer is scoped to a single stream.
  // Because a stream does not execute select_k_exec concurrently from multiple
  // threads, it is safe to update the allocator without additional locking.
  if (allocator != resContainer->xla_dev_mem_res->get_allocator()) {
    resContainer->xla_dev_mem_res->set_allocator(allocator);
  }

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
