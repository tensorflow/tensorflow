/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"
#include "triton/ir/builder.h"
#include "triton/ir/function.h"

namespace xla {
namespace gpu {

using tensorflow::AutotuneResult;

// Generate matrix multiplication in Triton IR inside 'fn'
// for 'dot_instr' which is described by an HLO custom call computation.
// Use tiling and execution parameters from 'config'.
// Values in 'config' can be adjusted by this function if the original ones
// are not executable or inefficient.
std::optional<LaunchDimensions> MatMul(triton::ir::builder& b,
                                       const HloDotInstruction* dot_instr,
                                       triton::ir::function* fn,
                                       AutotuneResult::TritonGemmKey& config,
                                       int shmem_budget);

// Generate Triton IR by running the provided generator, compile it into LLVM IR
// and return either launch dimensions or std::nullopt if generation failed.
// The MatMul() above is one of such possible IR generators.
std::optional<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    const se::CudaComputeCapability& cc, const GpuDeviceInfo& device_info,
    AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    std::function<std::optional<LaunchDimensions>(
        triton::ir::builder&, const HloDotInstruction*, triton::ir::function*,
        AutotuneResult::TritonGemmKey&, int)>
        generator);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
