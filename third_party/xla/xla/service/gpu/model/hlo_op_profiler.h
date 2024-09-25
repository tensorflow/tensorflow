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

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo_runner.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class HloOpProfiler {
  static std::unique_ptr<HloModule> MakeModuleForMeasurements(
      HloOpcode op, PrimitiveType data_type, int chain_length);

  absl::StatusOr<absl::Duration> MeasureOpChainDuration(HloOpcode op,
                                                        PrimitiveType data_type,
                                                        int chain_length);

 public:
  explicit HloOpProfiler(HloRunner& runner);
  absl::StatusOr<HloInstructionProfile> MeasureClockCyclesPerOp(
      HloOpcode op, PrimitiveType data_type);

 private:
  HloRunner& runner_;
  const se::DeviceDescription& dev_info_;
  absl::Duration min_duration_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILER_H_
