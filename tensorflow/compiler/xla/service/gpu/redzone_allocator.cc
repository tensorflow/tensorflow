/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/redzone_allocator.h"

#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace gpu {

// The size of the redzone at the end of the user buffer is rounded up to a
// multiple of kRhsRedzoneAlign.  This simplifies the implementation a bit.
constexpr int64 kRhsRedzoneAlign = 4;

StatusOr<se::DeviceMemory<uint8>> RedzoneAllocator::AllocateBytes(
    se::Stream* stream, int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes(stream)) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes(stream)));
  }

  int64 rhs_slop = RoundUpToNearest(byte_size, kRhsRedzoneAlign) - byte_size;
  TF_ASSIGN_OR_RETURN(
      OwningDeviceMemory allocated_buffer,
      memory_allocator_->Allocate(device_ordinal_,
                                  byte_size + 2 * redzone_size_ + rhs_slop,
                                  /*retry_on_failure=*/false));
  allocated_bytes_excluding_redzones_ += byte_size;

  static_assert(sizeof(uint8) == 1, "Unexpected size");
  se::DeviceMemory<uint8> allocated_buffer_memory(
      allocated_buffer.AsDeviceMemoryBase());

  se::DeviceMemory<uint8> lhs_redzone = stream->parent()->GetSubBuffer(
      &allocated_buffer_memory, 0, redzone_size_);

  se::DeviceMemory<uint8> data_chunk = stream->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_, byte_size);

  // Split up the RHS redzone into two pieces:
  //  - 0 to kRhsRedzoneAlign bytes adjacent to the user buffer, followed by
  //  - redzone_size_ bytes.
  // We do this because Stream::ThenMemset32 requires the buffer address and
  // size to be aligned to 4 bytes.
  se::DeviceMemory<uint8> rhs_redzone_slop = stream->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_ + byte_size, rhs_slop);

  se::DeviceMemory<uint8> rhs_redzone_nonslop = stream->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_ + byte_size + rhs_slop,
      redzone_size_);

  uint8 pattern_arr[] = {redzone_pattern_, redzone_pattern_, redzone_pattern_,
                         redzone_pattern_};
  uint32 pattern32;
  std::memcpy(&pattern32, pattern_arr, sizeof(pattern32));
  stream->ThenMemset32(&lhs_redzone, pattern32, redzone_size_);
  if (rhs_slop != 0) {
    stream->ThenMemcpy(&rhs_redzone_slop, &pattern32, rhs_slop);
  }
  stream->ThenMemset32(&rhs_redzone_nonslop, pattern32, redzone_size_);

  allocated_buffers_.emplace_back(std::move(allocated_buffer), byte_size);
  return data_chunk;
}

// PTX blob for the function which checks that every byte in
// input_buffer (length is buffer_length) is equal to redzone_pattern.
//
// On mismatch, increment the counter pointed to by out_mismatch_cnt_ptr.
//
// Generated from:
// __global__ void redzone_checker(unsigned char* input_buffer,
//                                 unsigned char redzone_pattern,
//                                 unsigned long long buffer_length,
//                                 int* out_mismatched_ptr) {
//   unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   if (input_buffer[idx] != redzone_pattern) atomicAdd(out_mismatched_ptr, 1);
// }
//
// Code must compile for the oldest GPU XLA may be compiled for.
static const char* redzone_checker_ptx = R"(
.version 4.2
.target sm_20
.address_size 64

.visible .entry redzone_checker(
  .param .u64 input_buffer,
  .param .u8 redzone_pattern,
  .param .u64 buffer_length,
  .param .u64 out_mismatch_cnt_ptr
)
{
  .reg .pred   %p<3>;
  .reg .b16   %rs<3>;
  .reg .b32   %r<6>;
  .reg .b64   %rd<8>;

  ld.param.u64   %rd6, [buffer_length];
  mov.u32   %r1, %tid.x;
  mov.u32   %r2, %ctaid.x;
  mov.u32   %r3, %ntid.x;
  mad.lo.s32   %r4, %r3, %r2, %r1;
  cvt.u64.u32   %rd3, %r4;
  setp.ge.u64   %p1, %rd3, %rd6;
  @%p1 bra   LBB6_3;
  ld.param.u8   %rs1, [redzone_pattern];
  ld.param.u64   %rd4, [input_buffer];
  cvta.to.global.u64   %rd2, %rd4;
  add.s64   %rd7, %rd2, %rd3;
  ld.global.u8   %rs2, [%rd7];
  setp.eq.s16   %p2, %rs2, %rs1;
  @%p2 bra   LBB6_3;
  ld.param.u64   %rd5, [out_mismatch_cnt_ptr];
  cvta.to.global.u64   %rd1, %rd5;
  atom.global.add.u32   %r5, [%rd1], 1;
LBB6_3:
  ret;
}
)";

// The PTX in redzone_checker_ptx has to be launched with specified types
// in the specified order.
using ComparisonKernelT = se::TypedKernel<se::DeviceMemory<uint8>, uint8,
                                          uint64, se::DeviceMemory<uint64>>;

// Compile PTX in redzone_checker_ptx, or get a cached compiled version (for a
// given stream executor and a given CUDA directory specified by an XLA flag).
static StatusOr<absl::Span<const uint8>> CompileRedzoneCheckPtxOrGetCached(
    se::StreamExecutor* executor, const HloModuleConfig& hlo_module_config) {
  // Cache for storing the compiled PTX for redzone checking.
  // The cache key is a stream executor, as it determines the supported
  // CUDA compute capability, and PtxCompilationOptions.
  using PtxCacheKey =
      std::pair<se::StreamExecutor*, PtxCompilationOptions::PtxOptionsTuple>;
  static tensorflow::mutex ptx_cache_mutex(tensorflow::LINKER_INITIALIZED);
  static auto& redzone_check_ptx_cache GUARDED_BY(ptx_cache_mutex) =
      *new absl::flat_hash_map<PtxCacheKey, std::vector<uint8>>();

  tensorflow::mutex_lock lock(ptx_cache_mutex);
  PtxCompilationOptions compilation_options(hlo_module_config);
  PtxCacheKey cache_key{executor, compilation_options.ToTuple()};
  auto it = redzone_check_ptx_cache.find(cache_key);
  if (it != redzone_check_ptx_cache.end()) {
    return {it->second};
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<uint8> compiled,
      CompilePtx(executor, redzone_checker_ptx, compilation_options));

  auto insert_result =
      redzone_check_ptx_cache.emplace(cache_key, std::move(compiled));
  CHECK(insert_result.second);
  return {insert_result.first->second};
}

// Check that redzones weren't overwritten on a host.
//
// Slower, but gives a more useful error message.
static Status CheckRedzoneHost(se::DeviceMemoryBase redzone,
                               se::DeviceMemoryBase user_allocation,
                               absl::string_view name, se::Stream* stream,
                               uint8 redzone_pattern, int64 redzone_size) {
  uint64 size = redzone.size();
  auto redzone_data = absl::make_unique<uint8[]>(size);
  TF_RETURN_IF_ERROR(stream->ThenMemcpy(redzone_data.get(), redzone, size)
                         .BlockHostUntilDone());
  XLA_SCOPED_LOGGING_TIMER("RedzoneAllocator::CheckBufferRedzones CPU loop.");

  std::array<uint8, sizeof(uint64)> pattern_arr;
  pattern_arr.fill(redzone_pattern);
  uint64 pattern64;
  std::memcpy(&pattern64, pattern_arr.data(), sizeof(uint64));

  int64 i;
  for (i = 0; i + 7 < size; i += sizeof(uint64)) {
    uint64 rz_value = *reinterpret_cast<uint64*>(&redzone_data[i]);
    if (rz_value != pattern64) {
      return InternalError(
          "Redzone mismatch in %s redzone of buffer %p at offset %d; "
          "expected %08x but was %08x.",
          name, user_allocation.opaque(), i, pattern64, rz_value);
    }
  }
  for (; i < size; ++i) {
    uint8 rz_value = redzone_data[i];
    if (rz_value != redzone_pattern) {
      return InternalError(
          "Redzone mismatch in %s redzone of buffer %p at offset %d; "
          "expected %08x but was %08x.",
          name, user_allocation.opaque(), i, redzone_pattern, rz_value);
    }
  }
  return Status::OK();
}

// Run the redzone checker on the provided buffer redzone.
//
// Increment out_param if mismatch occurs.
static Status RunRedzoneChecker(se::Stream* stream,
                                const se::DeviceMemory<uint8>& redzone,
                                uint8 redzone_pattern,
                                const se::DeviceMemory<uint64>& out_param,
                                const ComparisonKernelT& comparison_kernel) {
  se::StreamExecutor* executor = stream->parent();
  Shape redzone_shape = ShapeUtil::MakeShape(
      PrimitiveType::U8, {static_cast<int64>(redzone.size())});
  LaunchDimensions dim = CalculateLaunchDimensions(
      redzone_shape, executor->GetDeviceDescription());

  stream->ThenLaunch(se::ThreadDim(dim.threads_per_block()),
                     se::BlockDim(dim.block_count()), comparison_kernel,
                     redzone, redzone_pattern, redzone.size(), out_param);

  return Status::OK();
}

// Check redzones around the user allocation.
//
// Increment out_param if mismatch occurs.
static Status CheckRedzonesForBuffer(se::Stream* stream,
                                     se::DeviceMemoryBase memory,
                                     const se::DeviceMemory<uint64>& out_param,
                                     const ComparisonKernelT& comparison_kernel,
                                     int64 user_allocation_size,
                                     uint64 redzone_size,
                                     uint8 redzone_pattern) {
  se::StreamExecutor* executor = stream->parent();
  int64 rhs_slop =
      RoundUpToNearest<int64>(user_allocation_size, kRhsRedzoneAlign) -
      user_allocation_size;
  CHECK_EQ(memory.size(), user_allocation_size + rhs_slop + 2 * redzone_size);

  se::DeviceMemory<uint8> buffer_uint8(memory);
  se::DeviceMemory<uint8> lhs_redzone =
      executor->GetSubBuffer(&buffer_uint8, 0, redzone_size);
  se::DeviceMemory<uint8> user_allocation =
      executor->GetSubBuffer(&buffer_uint8, redzone_size, user_allocation_size);
  se::DeviceMemory<uint8> rhs_redzone =
      executor->GetSubBuffer(&buffer_uint8, redzone_size + user_allocation_size,
                             redzone_size + rhs_slop);

  TF_RETURN_IF_ERROR(RunRedzoneChecker(stream, lhs_redzone, redzone_pattern,
                                       out_param, comparison_kernel));
  TF_RETURN_IF_ERROR(RunRedzoneChecker(stream, rhs_redzone, redzone_pattern,
                                       out_param, comparison_kernel));
  int64 result;
  CHECK_EQ(out_param.size(), sizeof(result));
  stream->ThenMemcpy(&result, out_param, sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  if (result != 0) {
    TF_RETURN_IF_ERROR(CheckRedzoneHost(lhs_redzone, user_allocation, "LHS",
                                        stream, redzone_pattern, redzone_size));
    TF_RETURN_IF_ERROR(CheckRedzoneHost(rhs_redzone, user_allocation, "RHS",
                                        stream, redzone_pattern, redzone_size));
    LOG(FATAL) << "Mismatched results with host and device comparison";
  }

  return Status::OK();
}

Status RedzoneAllocator::CheckRedzones(se::Stream* stream) const {
  XLA_SCOPED_LOGGING_TIMER("Redzone checking");

  se::StreamExecutor* executor = stream->parent();

  TF_ASSIGN_OR_RETURN(
      absl::Span<const uint8> compiled_ptx,
      CompileRedzoneCheckPtxOrGetCached(executor, hlo_module_config_));

  se::ScopedDeviceMemory<uint64> out_param =
      executor->AllocateOwnedScalar<uint64>();
  stream->ThenMemZero(out_param.ptr(), sizeof(uint64));

  auto typed_or = CreateTypedKernel<se::DeviceMemory<uint8>, uint8, uint64,
                                    se::DeviceMemory<uint64>>(
      "redzone_checker",
      /*num_args=*/4, redzone_checker_ptx, compiled_ptx, executor);

  // TF_ASSIGN_OR_RETURN does not work due to complex template.
  if (!typed_or.ok()) {
    return typed_or.status();
  }
  std::unique_ptr<ComparisonKernelT> comparison_kernel =
      std::move(typed_or.ValueOrDie());

  for (const auto& buf_and_size : allocated_buffers_) {
    TF_RETURN_IF_ERROR(CheckRedzonesForBuffer(
        stream, buf_and_size.first.AsDeviceMemoryBase(), out_param.cref(),
        *comparison_kernel, buf_and_size.second, redzone_size_,
        redzone_pattern_));
  }

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
