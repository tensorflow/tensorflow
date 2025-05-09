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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// A thunk that copies data from a device buffer to another device buffer.
class DeviceToDeviceCopyThunk : public Thunk {
 public:
  // Constructs a CopyThunk that copies host data from `source_buffer` to the
  // device buffer `destination_buffer`. `mem_size` is the size of the data in
  // bytes.
  DeviceToDeviceCopyThunk(ThunkInfo thunk_info,
                          const BufferAllocation::Slice& source_buffer,
                          const BufferAllocation::Slice& destination_buffer,
                          uint64_t mem_size);

  DeviceToDeviceCopyThunk(const DeviceToDeviceCopyThunk&) = delete;
  DeviceToDeviceCopyThunk& operator=(const DeviceToDeviceCopyThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const BufferAllocation::Slice& source() const { return source_buffer_; }
  const BufferAllocation::Slice& destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
};

//===----------------------------------------------------------------------===//
// CopyThunk
//===----------------------------------------------------------------------===//
class CopyThunk : public Thunk {
 public:
  class AsyncEvents {
   public:
    // Add a new copy-start completion event.
    absl::Status Emplace(se::StreamExecutor* executor,
                         const HloInstruction* instr,
                         std::unique_ptr<se::Event> event);

    // Retrieve a completion event started by copy-start instruction
    // `instr`, and remove the event from the collection.
    absl::StatusOr<std::unique_ptr<se::Event>> Extract(
        se::StreamExecutor* executor, const HloInstruction* instr);

   private:
    using Key = std::pair<se::StreamExecutor*, const HloInstruction*>;
    absl::Mutex mutex_;
    absl::flat_hash_map<Key, std::unique_ptr<se::Event>> events_
        ABSL_GUARDED_BY(mutex_);
  };
  CopyThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
            const BufferAllocation::Slice& destination_buffer,
            uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  const BufferAllocation::Slice& source() const { return source_buffer_; }
  const BufferAllocation::Slice& destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
};

//===----------------------------------------------------------------------===//
// DeviceToHostCopyThunk
//===----------------------------------------------------------------------===//
// The memcpy between a host and a device

// A thunk that copies data from a device buffer to a host buffer.
class DeviceToHostCopyThunk : public CopyThunk {
 public:
  // Constructs a DeviceToHostCopyThunk that copies data from `source_buffer` to
  // the device buffer `destination_buffer`. `mem_size` is the size of the data
  // in bytes. `events` are the cuda record/wait events.
  // `instr` is the copy-start instruction.
  DeviceToHostCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice& source_buffer,
                        const BufferAllocation::Slice& destination_buffer,
                        uint64_t mem_size,
                        std::shared_ptr<CopyThunk::AsyncEvents> events,
                        const HloInstruction* instr);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  std::shared_ptr<CopyThunk::AsyncEvents> async_events_;
  const HloInstruction* instr_;
};

//===----------------------------------------------------------------------===//
// HostToDeviceCopyThunk
//===----------------------------------------------------------------------===//
// The memcpy between a host and a device

// A thunk that copies data from a host buffer to a device buffer.
class HostToDeviceCopyThunk : public CopyThunk {
 public:
  // Constructs a HostToDeviceCopyThunk that copies data from `source_buffer` to
  // the host buffer `destination_buffer`. `mem_size` is the size of the data
  // in bytes. `events` are the cuda record/wait events.
  // `instr` is the copy-start instruction.
  HostToDeviceCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice& source_buffer,
                        const BufferAllocation::Slice& destination_buffer,
                        uint64_t mem_size,
                        std::shared_ptr<CopyThunk::AsyncEvents> events,
                        const HloInstruction* instr);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  std::shared_ptr<CopyThunk::AsyncEvents> async_events_;
  const HloInstruction* instr_;
};

//===----------------------------------------------------------------------===//
// CopyDoneThunk
//===----------------------------------------------------------------------===//

class CopyDoneThunk : public Thunk {
 public:
  CopyDoneThunk(Thunk::Kind kind, ThunkInfo thunk_info,
                std::shared_ptr<CopyThunk::AsyncEvents> events,
                const HloInstruction* copy_start_instr);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::shared_ptr<CopyThunk::AsyncEvents> async_events_;
  const HloInstruction* copy_start_instr_;
};

//===----------------------------------------------------------------------===//
// DynamicMemcpyThunk
//===----------------------------------------------------------------------===//

class DynamicMemcpyThunk : public Thunk {
 public:
  struct MemcpyDescriptor {
    struct DynamicOffset {
      // The while loop whose induction variable defines the offset.
      const HloInstruction* while_loop;
      const HloInstruction* induction_variable;

      // See documentation for ResolveFunctionalDependencyOnInductionVariable.
      absl::flat_hash_map<const HloComputation*, absl::InlinedVector<bool, 1>>
          required_parameters;

      // All dependencies of `offset` must end in `induction_variable` or
      // constants only.
      const HloInstruction* offset;

      // The size of the dimension that this offset corresponds to. As per HLO
      // semantics, values of `offset` will be clamped to one less than this.
      int64_t dimension_size;

      // The stride with which to multiply the induction variable's value.
      int64_t byte_stride;
    };

    std::vector<DynamicOffset> src_dynamic_offsets;
    int64_t src_byte_static_offset = 0;

    std::vector<DynamicOffset> dst_dynamic_offsets;
    int64_t dst_byte_static_offset = 0;
  };

  DynamicMemcpyThunk(ThunkInfo thunk_info,
                     const BufferAllocation::Slice& source_buffer,
                     const BufferAllocation::Slice& destination_buffer,
                     uint64_t mem_size, MemcpyDescriptor descriptor);
  DynamicMemcpyThunk(const DynamicMemcpyThunk&) = delete;
  DynamicMemcpyThunk& operator=(const DynamicMemcpyThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
  MemcpyDescriptor descriptor_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
