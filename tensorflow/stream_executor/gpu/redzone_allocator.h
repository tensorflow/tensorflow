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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_

#include <vector>

#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"

namespace stream_executor {

// An allocator that allocates a bit of extra memory around the beginning/end of
// every allocation and can check that this memory is unmodified.
//
// This can be used to check for out-of-bounds writes, and, if the redzone is
// filled with a sufficiently "ugly" pattern, may also be able to check for
// out-of-bounds reads.  The default fill pattern of -1 is an unusual NaN
// pattern when interpreted as a floating-point number, so hopefully works for
// out-of-bounds reads and writes in those cases.
//
// This class implements ScratchAllocator, so can be used to allocate temp
// memory for cudnn convolutions.
class RedzoneAllocator : public ScratchAllocator {
 public:
  static constexpr int64_t kDefaultRedzoneSize =
      1LL << 23;  // 8MiB per side, 16MiB total.
  static constexpr uint8 kDefaultRedzonePattern = -1;
  RedzoneAllocator(Stream* stream, DeviceMemoryAllocator* memory_allocator,
                   GpuAsmOpts gpu_compilation_opts_,
                   int64_t memory_limit = (1LL << 32),  // 4GB
                   int64_t redzone_size = kDefaultRedzoneSize,
                   uint8 redzone_pattern = kDefaultRedzonePattern);

  // Redzones don't count towards the memory limit.
  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }

  int64_t TotalAllocatedBytesExcludingRedzones() const {
    return allocated_bytes_excluding_redzones_;
  }

  port::StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override;

  // Non-empty redzone check status implies that there was a write into a
  // redzone, with a string communicating the location of the write.
  struct RedzoneCheckStatus {
    RedzoneCheckStatus() = default;

    RedzoneCheckStatus(absl::string_view buffer_name, void* user_buffer_address,
                       int64_t offset, uint64_t expected_value,
                       uint64_t actual_value)
        : buffer_name(buffer_name),
          user_buffer_address(user_buffer_address),
          offset(offset),
          expected_value(expected_value),
          actual_value(actual_value) {}

    static RedzoneCheckStatus OK() { return {}; }

    bool ok() { return user_buffer_address == nullptr; }

    std::string RedzoneFailureMsg() const;

    std::string buffer_name = {};
    void* user_buffer_address = nullptr;
    int64_t offset = 0;
    uint64_t expected_value = 0;
    uint64_t actual_value = 0;
  };

  // Determines whether redzones around all allocated buffers are unmodified.
  //
  // Reinitializes redzones to the expected value, so that the same buffer
  // could be reused for multiple checks.
  //
  // Returns:
  //
  //  - RedzoneCheckStatus::OK() if everything went well.
  //  - RedzoneCheckStatus with a non-empty error message iff a write into a
  //    redzone has been detected.
  //  - A stream error, if loading or launching the kernel has failed.
  port::StatusOr<RedzoneCheckStatus> CheckRedzones() const;

  Stream* stream() const { return stream_; }

 private:
  const int device_ordinal_;
  Stream* stream_;

  // Memory limit of the allocator in bytes.
  const int64_t memory_limit_;

  // Redzone size on *one side* of allocation in bytes.
  //
  // Must be a multiple of kXlaAllocatedBufferAlignBytes, otherwise the buffers
  // returned to users will be misaligned.
  const int64_t redzone_size_;

  const uint8 redzone_pattern_;
  DeviceMemoryAllocator* memory_allocator_;
  GpuAsmOpts gpu_compilation_opts_;

  // The second element of the pair is the size of the user allocation.  This
  // isn't necessarily just first.size() - 2 * redzone_size_ because when the
  // user allocation size is not a multiple of 4 bytes, we round up the size of
  // the RHS redzone.
  //
  // ScratchAllocators need to free all allocated memory on destruction so we
  // use `OwningDeviceMemory` here.
  std::vector<std::pair<OwningDeviceMemory, int64_t>> allocated_buffers_;

  int64_t allocated_bytes_excluding_redzones_ = 0;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_
