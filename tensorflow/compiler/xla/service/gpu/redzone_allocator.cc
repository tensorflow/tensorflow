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
#include "tensorflow/compiler/xla/status_macros.h"

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

  char* addr =
      reinterpret_cast<char*>(allocated_buffer.AsDeviceMemoryBase().opaque());
  se::DeviceMemoryBase lhs_redzone(addr, redzone_size_,
                                   /*is_sub_buffer=*/true);

  // Split up the RHS redzone into two pieces:
  //  - 0 to kRhsRedzoneAlign bytes adjacent to the user buffer, followed by
  //  - redzone_size_ bytes.
  // We do this because Stream::ThenMemset32 requires the buffer address and
  // size to be aligned to 4 bytes.
  se::DeviceMemoryBase rhs_redzone_slop(addr + redzone_size_ + byte_size,
                                        rhs_slop, /*is_sub_buffer=*/true);
  se::DeviceMemoryBase rhs_redzone_nonslop(
      addr + redzone_size_ + byte_size + rhs_slop, redzone_size_,
      /*is_sub_buffer=*/true);

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
  return se::DeviceMemory<uint8>(se::DeviceMemoryBase(
      addr + redzone_size_, byte_size, /*is_sub_buffer=*/true));
}

Status RedzoneAllocator::CheckRedzones(se::Stream* stream) const {
  for (const auto& buf_and_size : allocated_buffers_) {
    const auto& allocated_buf = buf_and_size.first;
    int64 user_alloc_size = buf_and_size.second;
    char* addr =
        reinterpret_cast<char*>(allocated_buf.AsDeviceMemoryBase().opaque());
    // user_alloc_size isn't necessarily the same as
    // allocated_buf.size() - 2 * redzone_size_ because if user_alloc_size was
    // not a multiple of kRhsRedzoneAlign, we rounded it up.
    se::DeviceMemoryBase buf(addr + redzone_size_, user_alloc_size,
                             /*is_sub_buffer=*/true);
    TF_RETURN_IF_ERROR(CheckBufferRedzones(buf, stream));
  }
  return Status::OK();
}

Status RedzoneAllocator::CheckBufferRedzones(se::DeviceMemoryBase buf,
                                             se::Stream* stream) const {
  XLA_SCOPED_LOGGING_TIMER("RedzoneAllocator::CheckBufferRedzones.");
  char* buf_start = reinterpret_cast<char*>(buf.opaque());
  auto check_redzone = [&](int64 offset, int64 size, absl::string_view name) {
    se::DeviceMemoryBase redzone(buf_start + offset, size,
                                 /*is_sub_buffer=*/true);
    auto redzone_data = absl::make_unique<uint8[]>(size);
    TF_RETURN_IF_ERROR(stream->ThenMemcpy(redzone_data.get(), redzone, size)
                           .BlockHostUntilDone());
    XLA_SCOPED_LOGGING_TIMER("RedzoneAllocator::CheckBufferRedzones CPU loop.");

    std::array<uint8, sizeof(uint64)> pattern_arr;
    pattern_arr.fill(redzone_pattern_);
    uint64 pattern64;
    std::memcpy(&pattern64, pattern_arr.data(), sizeof(uint64));

    int64 i;
    for (i = 0; i + 7 < size; i += sizeof(uint64)) {
      uint64 rz_value = *reinterpret_cast<uint64*>(&redzone_data[i]);
      if (rz_value != pattern64) {
        return InternalError(
            "Redzone mismatch in %s redzone of buffer %p at offset %d; "
            "expected %08x but was %08x.",
            name, buf.opaque(), i, pattern64, rz_value);
      }
    }
    for (; i < size; ++i) {
      uint8 rz_value = redzone_data[i];
      if (rz_value != redzone_pattern_) {
        return InternalError(
            "Redzone mismatch in %s redzone of buffer %p at offset %d; "
            "expected %08x but was %08x.",
            name, buf.opaque(), i, redzone_pattern_, rz_value);
      }
    }
    return Status::OK();
  };

  // `buf` points to the buffer returned to the user, so the LHS redzone starts
  // before `buf`.
  TF_RETURN_IF_ERROR(check_redzone(-redzone_size_, redzone_size_, "LHS"));

  int64 rhs_slop =
      RoundUpToNearest<int64>(buf.size(), kRhsRedzoneAlign) - buf.size();
  TF_RETURN_IF_ERROR(
      check_redzone(buf.size(), redzone_size_ + rhs_slop, "RHS"));

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
