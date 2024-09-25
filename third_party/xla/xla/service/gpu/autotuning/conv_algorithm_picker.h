/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_CONV_ALGORITHM_PICKER_H_
#define XLA_SERVICE_GPU_AUTOTUNING_CONV_ALGORITHM_PICKER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Choose the fastest algorithm for each conv.
// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
//
// We pick the algorithm before fusion so that we can generate better HLO. After
// GpuConvRewriter, our convolutions are CustomCalls which return a
// tuple (conv_result, scratch_memory), and the each conv uses 0 bytes of
// scratch:
//
//   customcall = (f32[...], f32[0])
//   return gte(customcall, 0)
//
// The algorithm picker then chooses the best algorithm, and potentially
// increases the scratch space.  It replaces customcall with new_tuple,
// giving us the following:
//
//   new_customcall = (f32[...], f32[N])
//   new_tuple = tuple(gte(new_customcall, 0), constant f32[0])
//   return gte(new_tuple, 0)
//
// The new tuple and gte instructions can be simplified away, because
// nobody is expected to use the scratch value.
//
// However, if we were to run GpuConvAlgorithmPicker after fusion
// the gte(customcall, 0) would probably already be into a fusion node.  We
// can't simplify across HloComputation boundaries, so in this case we
// wouldn't be able to simplify away the new_tuple bits.
//
// It supports two modes: device and deviceless.
// In device mode, we run autotuning on the device and store autotune results.
//
// In deviceless mode, we pass in some information related to the device and
// use stored autotune results to rewrite convolutions. If the required autotune
// result is not stored, then the performance of convolution will be suboptimal.
class GpuConvAlgorithmPicker : public HloModulePass {
 public:
  explicit GpuConvAlgorithmPicker(AutotuneConfig config) : config_(config) {}

  absl::string_view name() const override {
    return "gpu-conv-algorithm-picker";
  }

  static bool IsEnabled(const HloModule* module) {
    return module->config().debug_options().xla_gpu_autotune_level() != 0;
  }

  static bool IsCandidate(const HloInstruction* instr) {
    return IsCustomCallToDnnConvolution(*instr);
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);
  absl::StatusOr<bool> RunOnInstruction(HloInstruction* instr);

  absl::StatusOr<AutotuneResult> PickBestAlgorithm(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<AutotuneResult> PickBestAlgorithmNoCache(
      const HloCustomCallInstruction* instr);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
  // Simple bundle of an algorithm and its output, for comparing results across
  // autotuned algorithms.
  struct ReferenceResult {
    stream_executor::dnn::AlgorithmDesc algorithm;
    std::vector<stream_executor::DeviceMemoryBase> buffers;
  };

  // Execution environment for autotuning. Runtime autotuning requires runtime
  // information such as input/output buffers in order to run. It can be
  // constructed from the autotuned instruction by FromInstruction.
  struct AutotuneRuntimeArguments {
    const HloModuleConfig hlo_module_config;
    RedzoneBuffers rz_buffers;
    const GpuConvConfig gpu_conv_config;
    std::optional<std::string> canonical_hlo;

    static absl::StatusOr<AutotuneRuntimeArguments> FromInstruction(
        const HloCustomCallInstruction* instr, const AutotuneConfig& config,
        const DebugOptions& debug_options);
  };

  absl::StatusOr<AutotuneResult> AutotuneOneConvRunner(
      GenericConvRunner* runner,
      std::optional<ReferenceResult>* reference_result,
      absl::Span<const stream_executor::dnn::AlgorithmDesc> disabled_algos,
      std::optional<AutotuneCacheKey> instruction_info,
      const AutotuneRuntimeArguments& runtime_arguments);

  // Pick the best algorithm for CUDA platform.
  absl::StatusOr<AutotuneResult> PickBestAlgorithmNoCacheCuda(
      const HloCustomCallInstruction* instr);
#endif

  absl::StatusOr<AutotuneResult> PickBestAlgorithmNoCacheRocm(
      const HloCustomCallInstruction* instr);

 private:
  AutotuneConfig config_;
};

}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_AUTOTUNING_CONV_ALGORITHM_PICKER_H_
