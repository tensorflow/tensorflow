/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_COMPILER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_COMPILER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Converts HLO fusions with cuDNN backend config to cuDNN graphs,
// compiles them using a cuDNN handle and serializes them.
class CuDnnFusionCompiler : public HloModulePass {
 public:
  explicit CuDnnFusionCompiler(se::dnn::DnnSupport* dnn_support,
                               const se::DeviceDescription& gpu_device_info,
                               BinaryMap& compilation_results)
      : dnn_support_(dnn_support),
        gpu_device_info_(gpu_device_info),
        compilation_results_(compilation_results) {}

  absl::string_view name() const override { return "cudnn-fusion-compiler"; }

  static absl::StatusOr<int> GetAvailablePlanCount(
      se::StreamExecutor* stream_exec,
      const se::DeviceDescription& gpu_device_info,
      const HloFusionInstruction& hlo);

  enum class DevicelessFusionSupport {
    // cuDNN's deviceless heuristics advertise at least one execution plan.
    // Not a guarantee: building the plan on the real device (possibly a
    // different cuDNN) can still fail.
    kSupported,
    // Authoritative: cuDNN reports no engine supports this graph on the
    // target.
    kUnsupported,
    // No verdict (graph construction or deviceless preparation failed; cause
    // is VLOG(1)-logged). Not evidence either way.
    kUnknown,
  };

  // Deviceless (no GPU / cuDNN handle) probe of cuDNN support for the fusion
  // `hlo` on the target `gpu_device_info`. The verdict applies to the fusion
  // pipeline only; the legacy conv custom-call pipeline enumerates engines
  // differently.
  static DevicelessFusionSupport SupportsFusionDeviceless(
      const se::DeviceDescription& gpu_device_info,
      const HloFusionInstruction& hlo);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::dnn::DnnSupport* dnn_support_;
  const se::DeviceDescription& gpu_device_info_;
  BinaryMap& compilation_results_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_COMPILER_H_
