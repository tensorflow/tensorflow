/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executable_interface.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"

namespace xla {

class TpuExecutable : public xla::TpuExecutableInterface {
 public:
  TpuExecutable(SE_Executable* se_executable,
                std::shared_ptr<HloModule> hlo_module)
      : TpuExecutableInterface(std::move(hlo_module)),
        se_executable_(se_executable) {}

  ~TpuExecutable() override;

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  absl::string_view fingerprint() const override;

  // The serialization is not guaranteed to be stable over time and has no
  // compatibility guarantees (i.e. this is not a suitable long-term storage
  // format).
  absl::StatusOr<std::string> Serialize() const;
  static absl::StatusOr<std::unique_ptr<TpuExecutable>> Deserialize(
      absl::string_view serialized);

 private:
  Status LoadProgramAndEnqueueToStream(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      const std::vector<stream_executor::DeviceMemoryBase>&
          cross_program_prefetch_addrs,
      const std::vector<uint32_t>& cross_program_prefetch_offsets) override {
    LOG(FATAL) << "LoadProgramAndEnqueueToStream unimplemented";
  }

  SE_Executable* se_executable_;
};

}  // namespace xla

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_H_
