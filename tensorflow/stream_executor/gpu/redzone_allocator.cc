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

#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

#include "absl/base/call_once.h"
#include "absl/container/fixed_array.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpToNearest(13, 8) => 16
template <typename T>
static T RoundUpToNearest(T value, T divisor) {
  return tensorflow::MathUtil::CeilOfRatio(value, divisor) * divisor;
}

// The size of the redzone at the end of the user buffer is rounded up to a
// multiple of kRhsRedzoneAlign.  This simplifies the implementation a bit.
constexpr int64_t kRhsRedzoneAlign = 4;

using RedzoneCheckStatus = RedzoneAllocator::RedzoneCheckStatus;

RedzoneAllocator::RedzoneAllocator(Stream* stream,
                                   DeviceMemoryAllocator* memory_allocator,
                                   GpuAsmOpts ptx_compilation_opts,
                                   int64_t memory_limit, int64_t redzone_size,
                                   uint8 redzone_pattern)
    : device_ordinal_(stream->parent()->device_ordinal()),
      stream_(stream),
      memory_limit_(memory_limit),
      redzone_size_(RoundUpToNearest(
          redzone_size,
          static_cast<int64_t>(tensorflow::Allocator::kAllocatorAlignment))),
      redzone_pattern_(redzone_pattern),
      memory_allocator_(memory_allocator),
      gpu_compilation_opts_(ptx_compilation_opts) {}

port::StatusOr<DeviceMemory<uint8>> RedzoneAllocator::AllocateBytes(
    int64_t byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return port::Status(
        port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  int64_t rhs_slop = RoundUpToNearest(byte_size, kRhsRedzoneAlign) - byte_size;
  TF_ASSIGN_OR_RETURN(
      OwningDeviceMemory allocated_buffer,
      memory_allocator_->Allocate(device_ordinal_,
                                  byte_size + 2 * redzone_size_ + rhs_slop,
                                  /*retry_on_failure=*/false));
  allocated_bytes_excluding_redzones_ += byte_size;

  static_assert(sizeof(uint8) == 1, "Unexpected size");
  DeviceMemory<uint8> allocated_buffer_memory(*allocated_buffer);

  DeviceMemory<uint8> lhs_redzone = stream_->parent()->GetSubBuffer(
      &allocated_buffer_memory, 0, redzone_size_);

  DeviceMemory<uint8> data_chunk = stream_->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_, byte_size);

  // Split up the RHS redzone into two pieces:
  //  - 0 to kRhsRedzoneAlign bytes adjacent to the user buffer, followed by
  //  - redzone_size_ bytes.
  // We do this because Stream::ThenMemset32 requires the buffer address and
  // size to be aligned to 4 bytes.
  DeviceMemory<uint8> rhs_redzone_slop = stream_->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_ + byte_size, rhs_slop);

  DeviceMemory<uint8> rhs_redzone_nonslop = stream_->parent()->GetSubBuffer(
      &allocated_buffer_memory, redzone_size_ + byte_size + rhs_slop,
      redzone_size_);

  uint8 pattern_arr[] = {redzone_pattern_, redzone_pattern_, redzone_pattern_,
                         redzone_pattern_};
  uint32 pattern32;
  std::memcpy(&pattern32, pattern_arr, sizeof(pattern32));
  stream_->ThenMemset32(&lhs_redzone, pattern32, redzone_size_);
  if (rhs_slop != 0) {
    stream_->ThenMemcpy(&rhs_redzone_slop, &pattern32, rhs_slop);
  }
  stream_->ThenMemset32(&rhs_redzone_nonslop, pattern32, redzone_size_);

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
.target sm_30
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
using ComparisonKernelT =
    TypedKernel<DeviceMemory<uint8>, uint8, uint64_t, DeviceMemory<uint64_t>>;

// Check that redzones weren't overwritten on a host.
//
// Slower, but gives a more useful error message.
static port::StatusOr<RedzoneCheckStatus> CheckRedzoneHost(
    DeviceMemoryBase redzone, DeviceMemoryBase user_allocation,
    absl::string_view name, Stream* stream, uint8 redzone_pattern) {
  uint64_t size = redzone.size();
  auto redzone_data = absl::make_unique<uint8[]>(size);
  TF_RETURN_IF_ERROR(stream->ThenMemcpy(redzone_data.get(), redzone, size)
                         .BlockHostUntilDone());

  std::array<uint8, sizeof(uint64_t)> pattern_arr;
  pattern_arr.fill(redzone_pattern);
  uint64_t pattern64;
  std::memcpy(&pattern64, pattern_arr.data(), sizeof(uint64_t));

  int64_t i;
  for (i = 0; i + 7 < size; i += sizeof(uint64_t)) {
    uint64_t rz_value = *reinterpret_cast<uint64_t*>(&redzone_data[i]);
    if (rz_value != pattern64) {
      return RedzoneCheckStatus(name, user_allocation.opaque(), i, pattern64,
                                rz_value);
    }
  }
  for (; i < size; ++i) {
    uint8 rz_value = redzone_data[i];
    if (rz_value != redzone_pattern) {
      return RedzoneCheckStatus(name, user_allocation.opaque(), i,
                                redzone_pattern, rz_value);
    }
  }
  return RedzoneCheckStatus::OK();
}

// Run the redzone checker on the provided buffer redzone.
//
// Increment out_param if mismatch occurs.
static port::Status RunRedzoneChecker(
    Stream* stream, const DeviceMemory<uint8>& redzone, uint8 redzone_pattern,
    const DeviceMemory<uint64_t>& out_param,
    const ComparisonKernelT& comparison_kernel) {
  StreamExecutor* executor = stream->parent();

  int64_t num_elements = redzone.size();
  int64_t threads_per_block = std::min(
      executor->GetDeviceDescription().threads_per_block_limit(), num_elements);
  int64_t block_count =
      tensorflow::MathUtil::CeilOfRatio(num_elements, threads_per_block);

  TF_RETURN_IF_ERROR(stream->ThenLaunch(
      ThreadDim(threads_per_block), BlockDim(block_count), comparison_kernel,
      redzone, redzone_pattern, redzone.size(), out_param));
  return ::tensorflow::OkStatus();
}

// Since we reuse the same buffer for multiple checks, we re-initialize redzone
// with a NaN pattern after a failed check.
//
// This function is blocking, since redzone failing is a rare event.
static port::Status ReinitializeRedzone(Stream* stream,
                                        DeviceMemoryBase redzone,
                                        uint8 redzone_pattern) {
  absl::FixedArray<uint8> redzone_array(redzone.size());
  redzone_array.fill(redzone_pattern);
  stream->ThenMemcpy(&redzone, redzone_array.data(), redzone.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return ::tensorflow::OkStatus();
}

// Check redzones around the user allocation.
//
// Precondition: the memory pointed out by out_param is zeroed.
static port::StatusOr<RedzoneCheckStatus> CheckRedzonesForBuffer(
    Stream* stream, DeviceMemoryBase memory,
    const DeviceMemory<uint64_t>& out_param,
    const ComparisonKernelT& comparison_kernel, int64_t user_allocation_size,
    uint64_t redzone_size, uint8 redzone_pattern) {
  StreamExecutor* executor = stream->parent();
  int64_t rhs_slop =
      RoundUpToNearest<int64_t>(user_allocation_size, kRhsRedzoneAlign) -
      user_allocation_size;
  CHECK_EQ(memory.size(), user_allocation_size + rhs_slop + 2 * redzone_size);

  DeviceMemory<uint8> buffer_uint8(memory);
  DeviceMemory<uint8> lhs_redzone =
      executor->GetSubBuffer(&buffer_uint8, 0,
                             /*element_count=*/redzone_size);
  DeviceMemory<uint8> user_allocation =
      executor->GetSubBuffer(&buffer_uint8, redzone_size,
                             /*element_count=*/user_allocation_size);
  DeviceMemory<uint8> rhs_redzone =
      executor->GetSubBuffer(&buffer_uint8, redzone_size + user_allocation_size,
                             /*element_count=*/redzone_size + rhs_slop);

  TF_RETURN_IF_ERROR(RunRedzoneChecker(stream, lhs_redzone, redzone_pattern,
                                       out_param, comparison_kernel));
  TF_RETURN_IF_ERROR(RunRedzoneChecker(stream, rhs_redzone, redzone_pattern,
                                       out_param, comparison_kernel));
  int64_t result;
  CHECK_EQ(out_param.size(), sizeof(result));
  stream->ThenMemcpy(&result, out_param, sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  if (result != 0) {
    TF_ASSIGN_OR_RETURN(RedzoneCheckStatus lhs_check,
                        CheckRedzoneHost(lhs_redzone, user_allocation, "LHS",
                                         stream, redzone_pattern));
    TF_ASSIGN_OR_RETURN(RedzoneCheckStatus rhs_check,
                        CheckRedzoneHost(rhs_redzone, user_allocation, "RHS",
                                         stream, redzone_pattern));

    CHECK(!lhs_check.ok() || !rhs_check.ok())
        << "Mismatched results with host and device comparison";

    TF_RETURN_IF_ERROR(
        ReinitializeRedzone(stream, lhs_redzone, redzone_pattern));
    TF_RETURN_IF_ERROR(
        ReinitializeRedzone(stream, rhs_redzone, redzone_pattern));
    return !lhs_check.ok() ? lhs_check : rhs_check;
  }

  return RedzoneCheckStatus::OK();
}

port::StatusOr<RedzoneCheckStatus> RedzoneAllocator::CheckRedzones() const {
  StreamExecutor* executor = stream_->parent();

  absl::Span<const uint8> compiled_ptx = {};
  port::StatusOr<absl::Span<const uint8>> compiled_ptx_or =
      CompileGpuAsmOrGetCached(executor->device_ordinal(), redzone_checker_ptx,
                               gpu_compilation_opts_);
  if (compiled_ptx_or.ok()) {
    compiled_ptx = compiled_ptx_or.ValueOrDie();
  } else {
    static absl::once_flag ptxas_not_found_logged;
    absl::call_once(ptxas_not_found_logged, [&]() {
      LOG(WARNING) << compiled_ptx_or.status().ToString()
                   << "\nRelying on driver to perform ptx compilation. "
                   << "\nModify $PATH to customize ptxas location."
                   << "\nThis message will be only logged once.";
    });
  }

  ScopedDeviceMemory<uint64_t> out_param =
      executor->AllocateOwnedScalar<uint64_t>();
  stream_->ThenMemZero(out_param.ptr(), sizeof(uint64_t));

#if GOOGLE_CUDA
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<ComparisonKernelT> loaded_kernel,
      (LoadKernelOrGetPtr<DeviceMemory<uint8>, uint8, uint64_t,
                          DeviceMemory<uint64_t>>(
          executor, "redzone_checker", redzone_checker_ptx, compiled_ptx)));
#else
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ComparisonKernelT> loaded_kernel,
      (executor->CreateTypedKernel<DeviceMemory<uint8>, uint8, uint64_t,
                                   DeviceMemory<uint64_t>>(
          "redzone_checker", redzone_checker_ptx, compiled_ptx)));
#endif  // GOOGLE_CUDA

  for (const auto& buf_and_size : allocated_buffers_) {
    TF_ASSIGN_OR_RETURN(
        RedzoneCheckStatus redzone_status,
        CheckRedzonesForBuffer(stream_, *buf_and_size.first, out_param.cref(),
                               *loaded_kernel, buf_and_size.second,
                               redzone_size_, redzone_pattern_));
    if (!redzone_status.ok()) {
      return redzone_status;
    }
  }

  return RedzoneCheckStatus::OK();
}

std::string RedzoneCheckStatus::RedzoneFailureMsg() const {
  return absl::StrFormat(
      "Redzone mismatch in %s redzone of buffer %p at offset %d; "
      "expected %08x but was %08x.",
      buffer_name, user_buffer_address, offset, expected_value, actual_value);
}

}  // namespace stream_executor
