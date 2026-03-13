/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CUB_SCAN_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CUB_SCAN_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::gpu {

class CubScanRunnerInterface {
 public:
  virtual ~CubScanRunnerInterface() = default;
  virtual absl::Status Run(const se::DeviceMemoryBase& input_buffer,
                           const se::DeviceMemoryBase& output_buffer,
                           const se::DeviceMemoryBase& scratch_buffer,
                           int64_t num_elements, se::Stream* stream) = 0;
  virtual absl::StatusOr<int64_t> GetScratchSize(int64_t num_elements) = 0;

  static absl::StatusOr<std::unique_ptr<CubScanRunnerInterface>> Create(
      PrimitiveType type, const std::string& platform_name);
};

class CubScanThunk : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<CubScanThunk>> Create(
      ThunkInfo thunk_info, PrimitiveType type,
      const BufferAllocation::Slice& input_slice,
      const BufferAllocation::Slice& output_slice,
      const BufferAllocation::Slice& scratch_slice, int64_t num_elements);

  CubScanThunk(ThunkInfo thunk_info,
               std::unique_ptr<CubScanRunnerInterface> runner,
               PrimitiveType type, std::string platform_name,
               const BufferAllocation::Slice& input_slice,
               const BufferAllocation::Slice& output_slice,
               const BufferAllocation::Slice& scratch_slice,
               int64_t num_elements);

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<CubScanThunk>> FromProto(
      ThunkInfo thunk_info, const CubScanThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

 private:
  std::unique_ptr<CubScanRunnerInterface> runner_;
  PrimitiveType type_;
  std::string platform_name_;
  BufferAllocation::Slice input_slice_;
  BufferAllocation::Slice output_slice_;
  BufferAllocation::Slice scratch_slice_;
  int64_t num_elements_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_CUB_SCAN_THUNK_H_
