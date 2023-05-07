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

#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

using tensorflow::AutotuneResult;

// Generate matrix multiplication in Triton IR inside 'fn'
// for 'dot_instr' which is described by an HLO custom call computation.
// Use tiling and execution parameters from 'config'.
StatusOr<LaunchDimensions> MatMul(mlir::OpBuilder b,
                                  const HloDotInstruction* dot_instr,
                                  mlir::triton::FuncOp fn,
                                  const AutotuneResult::TritonGemmKey& config,
                                  int shmem_budget);

using LaunchDimensionsGenerator = std::function<StatusOr<LaunchDimensions>(
    mlir::OpBuilder, const HloDotInstruction*, mlir::triton::FuncOp,
    const AutotuneResult::TritonGemmKey&, int)>;

// Generate Triton IR by running the provided generator, compile it into LLVM IR
// and return launch dimensions.
// The MatMul() above is one of such possible IR generators.
StatusOr<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    const se::CudaComputeCapability& cc, const GpuDeviceInfo& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    LaunchDimensionsGenerator generator, mlir::MLIRContext& mlir_context);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
