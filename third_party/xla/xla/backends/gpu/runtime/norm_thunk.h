/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/protobuf/dnn.pb.h"

namespace xla::gpu {

// Intermediate structure used as input to construct NormThunk.
struct GpuNormDescriptor {
  CudnnNormBackendConfig backend_config;
  Shape x_shape;
  Shape scale_shape;
  std::optional<Shape> bias_shape;
  Shape y_or_dx_shape;
  std::optional<Shape> expectation_shape;
  std::optional<Shape> norm_factor_shape;
  std::optional<Shape> dy_shape;
  std::optional<Shape> dscale_shape;
  std::optional<Shape> dbias_shape;
  Shape scratch_shape;

  static absl::StatusOr<GpuNormDescriptor> FromProto(
      const GpuNormDescriptorProto& proto);

  GpuNormDescriptorProto ToProto() const;
};

class NormThunk : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<NormThunk>> Create(
      ThunkInfo thunk_info, GpuNormDescriptor descriptor,
      BufferAllocation::Slice x, BufferAllocation::Slice scale,
      BufferAllocation::Slice y_or_dx,
      std::optional<BufferAllocation::Slice> bias,
      std::optional<BufferAllocation::Slice> expectation,
      std::optional<BufferAllocation::Slice> norm_factor,
      std::optional<BufferAllocation::Slice> dy,
      std::optional<BufferAllocation::Slice> dscale,
      std::optional<BufferAllocation::Slice> dbias,
      BufferAllocation::Slice scratch, int devices_per_host);

  NormThunk(const NormThunk&) = delete;
  NormThunk& operator=(const NormThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;

  BufferUses buffer_uses() const override;

  static absl::StatusOr<std::unique_ptr<NormThunk>> FromProto(
      ThunkInfo thunk_info, const NormThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      int devices_per_host);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  struct Config {
    static absl::StatusOr<Config> For(const GpuNormDescriptor& desc);

    double epsilon;
    se::dnn::NormKind kind;
    se::dnn::AlgorithmDesc algorithm;
    se::dnn::TensorDescriptor x_descriptor;
    se::dnn::TensorDescriptor scale_descriptor;
    std::optional<se::dnn::TensorDescriptor> bias_descriptor;
    se::dnn::TensorDescriptor y_or_dx_descriptor;
    std::optional<se::dnn::TensorDescriptor> expectation_descriptor;
    std::optional<se::dnn::TensorDescriptor> norm_factor_descriptor;
    std::optional<se::dnn::TensorDescriptor> dy_descriptor;
    std::optional<se::dnn::TensorDescriptor> dscale_descriptor;
    std::optional<se::dnn::TensorDescriptor> dbias_descriptor;
  };

  NormThunk(ThunkInfo thunk_info, Config config, GpuNormDescriptor descriptor,
            BufferAllocation::Slice x, BufferAllocation::Slice scale,
            BufferAllocation::Slice y_or_dx,
            std::optional<BufferAllocation::Slice> bias,
            std::optional<BufferAllocation::Slice> expectation,
            std::optional<BufferAllocation::Slice> norm_factor,
            std::optional<BufferAllocation::Slice> dy,
            std::optional<BufferAllocation::Slice> dscale,
            std::optional<BufferAllocation::Slice> dbias,
            BufferAllocation::Slice scratch, int devices_per_host);

  absl::StatusOr<const se::dnn::NormRunner*> GetOrCreateRunner(
      se::Stream* absl_nonnull stream);

  BufferAllocation::Slice x_buffer_;
  BufferAllocation::Slice scale_buffer_;
  BufferAllocation::Slice y_or_dx_buffer_;
  std::optional<BufferAllocation::Slice> bias_buffer_;
  std::optional<BufferAllocation::Slice> expectation_buffer_;
  std::optional<BufferAllocation::Slice> norm_factor_buffer_;
  std::optional<BufferAllocation::Slice> dy_buffer_;
  std::optional<BufferAllocation::Slice> dscale_buffer_;
  std::optional<BufferAllocation::Slice> dbias_buffer_;
  BufferAllocation::Slice scratch_buffer_;

  // Technically this is only needed during initialization to create the
  // Config, but the actual Config is hard to serialize. So we keep the
  // descriptor around for serialization purposes.
  GpuNormDescriptor descriptor_;
  Config config_;
  PerDeviceState<std::unique_ptr<const se::dnn::NormRunner>> runner_states_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_
