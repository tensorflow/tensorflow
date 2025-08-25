/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/raft_select_k_thunk.h"

#include <cstdint>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/backends/gpu/runtime/raft_select_k_exec.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// RaftSelectKThunk
//===----------------------------------------------------------------------===//

RaftSelectKThunk::RaftSelectKThunk(
    const HloInstruction* inst, std::uint32_t batch_size,
    std::uint32_t num_elements, std::uint32_t k, xla::PrimitiveType dtype,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kRaftSelectK, Thunk::ThunkInfo::WithProfileAnnotation(inst)),
      batch_size_(batch_size),
      num_elements_(num_elements),
      k_(k),
      dtype_(dtype),
      args_(kernel_arguments.GetArgumentBufferSlices()) {}

std::string RaftSelectKThunk::ToString(int indent) const {
  return absl::StrCat("RaftSelectKThunk(batch_size=", batch_size_,
                      ", num_elements=", num_elements_, ", k=", k_,
                      ", dtype=", dtype_, ")");
}

// Initialize: no special initialization required.
absl::Status RaftSelectKThunk::Initialize(const InitializeParams& /*params*/) {
  return absl::OkStatus();
}

// Execute the TopK operation on the GPU stream.
// Maps kernel arguments to device memory and dispatches the appropriate
// raft_select_k_exec based on the data type.
absl::Status RaftSelectKThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Launching " << ToString(0);

  // Map buffer slices to device memory.
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  int device_ordinal = params.buffer_allocations->device_ordinal();
  auto* allocator = params.buffer_allocations->memory_allocator();
  se::Stream* stream = params.stream;

  // Dispatch to the correct typed implementation based on dtype.
  switch (dtype_) {
    case PrimitiveType::F32:
      return raft_select_k_exec<float>(
          device_ordinal, allocator, stream, buffer_args[0], buffer_args[1],
          buffer_args[2], batch_size_, num_elements_, k_);
    case PrimitiveType::BF16:
      return raft_select_k_exec<nv_bfloat16>(
          device_ordinal, allocator, stream, buffer_args[0], buffer_args[1],
          buffer_args[2], batch_size_, num_elements_, k_);
    default:
      return absl::UnimplementedError(
          absl::StrCat("RaftSelectKThunk: Unsupported dtype: ", dtype_));
  }
}
}  // namespace xla::gpu
