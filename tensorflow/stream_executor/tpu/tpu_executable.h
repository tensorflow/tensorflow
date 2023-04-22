/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_

#include "tensorflow/stream_executor/tpu/tpu_executable_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

namespace xla {

class TpuExecutable : public xla::TpuExecutableInterface {
 public:
  TpuExecutable(SE_Executable* se_executable,
                std::shared_ptr<HloModule> hlo_module)
      : TpuExecutableInterface(std::move(hlo_module)),
        se_executable_(se_executable) {}

  ~TpuExecutable() override;

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  absl::string_view fingerprint() const override;

  // The serialization is not guaranteed to be stable over time and has no
  // compatibility guarantees (i.e. this is not a suitable long-term storage
  // format).
  StatusOr<std::string> Serialize() const;
  static StatusOr<std::unique_ptr<TpuExecutable>> Deserialize(
      absl::string_view serialized, std::unique_ptr<HloModule> hlo_module);

 private:
  Status LoadProgramAndEnqueueToStream(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      absl::optional<stream_executor::DeviceMemoryBase>
          cross_program_prefetch_addr) override {
    LOG(FATAL) << "LoadProgramAndEnqueueToStream unimplemented";
  }

  Shape HostShapeToDeviceShape(const Shape& host_shape) override {
    LOG(FATAL) << "HostShapeToDeviceShape unimplemented";
  }

  int64 ShapeSize(const Shape& shape) override {
    LOG(FATAL) << "ShapeSize unimplemented";
  }

  SE_Executable* se_executable_;
};

}  // namespace xla

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_
