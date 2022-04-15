/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_

#include <functional>
#include <memory>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/tpu/tpu_executable_interface.h"

namespace tensorflow {

// An executable capable of being fed to a TPU device via TpuExecutor.
class TpuOpExecutable : public xla::TpuExecutableInterface {
 public:
  using HostCommandHandler = std::function<void(uint32, int64_t)>;

  // Constructs an executable that holds a non-owning reference to an
  // XLA_TpuProgram.
  explicit TpuOpExecutable(const XLA_TpuProgram* core_program,
                           std::unique_ptr<xla::HloModule> hlo_module,
                           HostCommandHandler host_command_handler = nullptr);
  ~TpuOpExecutable() override = default;

  const XLA_TpuProgram* core_program() const { return core_program_; }

  absl::string_view fingerprint() const override;

 private:
  Status LoadProgramAndEnqueueToStream(
      const xla::ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      absl::optional<stream_executor::DeviceMemoryBase>
          cross_program_prefetch_addr) override;

  xla::Shape HostShapeToDeviceShape(const xla::Shape& host_shape) override;

  int64_t ShapeSize(const xla::Shape& shape) override;

  const XLA_TpuProgram* const core_program_;

  const HostCommandHandler host_command_handler_;

  TF_DISALLOW_COPY_AND_ASSIGN(TpuOpExecutable);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_
