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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_IPU_SCHEDULER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_IPU_SCHEDULER_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

using IpuSchedulerAlgorithm = std::function<StatusOr<HloInstructionSequence>(
    HloComputation*, const TuplePointsToAnalysis&,
    const LogicalBuffer::SizeFunction&,
    const absl::flat_hash_map<const HloComputation*, int64>&)>;

/**
 * Convert a tensorflow MemorySchedulerAlgorithm to a IpuSchedulerAlgorithm
 *
 * @param algorithm The tensorflow MemorySchedulerAlgorithm to convert
 *
 * @returns a valid IpuSchedulerAlgorithm, or a failure status
 */
IpuSchedulerAlgorithm MemorySchedulerAlgorithmToIPU(
    MemorySchedulerAlgorithm algorithm);

/**
 * Convert an IpuSchedulerAlgorithm into a tensorflow MemorySchedulerAlgorithm
 *
 * @param algorithm The IpuSchedulerAlgorithm to convert
 *
 * @returns a valid MemorySchedulerAlgorithm, or a failure status
 */
MemorySchedulerAlgorithm IpuToMemorySchedulerAlgorithm(
    IpuSchedulerAlgorithm algorithm);

/**
 * Given two scheduling algorithms, create a new schedule algorithm which will
 * return the best of the given scheduling algorithms.
 *
 * @param algorithm_a The first algorithm
 * @param algorithm_b The second algorithm
 *
 * @returns a valid IpuSchedulerAlgorithm, or a failure status
 */
StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    IpuSchedulerAlgorithm algorithm_a, IpuSchedulerAlgorithm algorithm_b);

/**
 * Given a set of scheduling algorithms, create a new schedule algorithm which
 * will return the best of the given scheduling algorithms.
 *
 * @param algorithms The set of algorithms
 *
 * @returns a valid IpuSchedulerAlgorithm
 */
StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    const std::vector<IpuSchedulerAlgorithm>& algorithms);

/**
 * An HLO module pass which applies the given scheduling algorithm to each
 * computation in the module.
 */
class IpuScheduler : public HloModulePass {
 public:
  // size_function is the function returning the number of bytes required for a
  // LogicalBuffer. algorithm is the memory scheduling algorithm to use.
  IpuScheduler(const LogicalBuffer::SizeFunction& size_function,
               const IpuSchedulerAlgorithm& algorithm);

  absl::string_view name() const override { return "ipu-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  LogicalBuffer::SizeFunction size_function_;

  IpuSchedulerAlgorithm algorithm_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
