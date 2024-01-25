/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_PARALLEL_LOOP_EMITTER_H_
#define XLA_SERVICE_GPU_PARALLEL_LOOP_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/loop_emitter.h"

namespace xla {
namespace gpu {

// Emits a parallel loop for every element in the given array shape. This loop
// emitted will be executed by multiple threads in parallel. Therefore, each
// thread instance of the loop iterates over part of the array, and they
// collectively iterates over the entire array.
class ParallelLoopEmitter {
 public:
  // `launch_dimensions` specify the number of threads and blocks to
  // parallelize the loop on.  `launch_config` specify some detail on
  // how to parallelize.
  ParallelLoopEmitter(llvm_ir::BodyEmitter body_emitter, const Shape& shape,
                      const LaunchDimensions& launch_dimensions,
                      llvm::IRBuilder<>* b,
                      LaunchDimensionsConfig launch_config = {});

  // Constructs a loop emitter for a loop that generates on element of each of N
  // arrays on each iteration.
  //
  // This is used in multi-output fusion.  target_element_generator should
  // produce a struct with N elements, one for each of target_arrays.
  ParallelLoopEmitter(const llvm_ir::ElementGenerator& target_element_generator,
                      absl::Span<const llvm_ir::IrArray> target_arrays,
                      const LaunchDimensions& launch_dimensions,
                      llvm::IRBuilder<>* b,
                      LaunchDimensionsConfig launch_config = {});

  ParallelLoopEmitter(const ParallelLoopEmitter&) = delete;
  ParallelLoopEmitter& operator=(const ParallelLoopEmitter&) = delete;

  std::vector<llvm_ir::IrArray::Index> EmitIndexAndSetExitBasicBlock(
      absl::string_view loop_name, llvm::Type* index_type,
      llvm::Value* base_index);

  absl::Status EmitLoop(absl::string_view loop_name = "",
                        llvm::Type* index_type = nullptr);

 private:
  struct LinearBaseAndThreadIdx {
    llvm::Value* linear_base;
    llvm::Value* thread_idx;
  };

  LinearBaseAndThreadIdx EmitLinearBaseAndThreadIdx(llvm::Type* index_type,
                                                    llvm::Value* base_index);
  absl::Status EmitSerialLoop(absl::string_view loop_name,
                              llvm::Type* index_type,
                              llvm::Value* base_indvar = nullptr);

  // The thread and block dimension to parallelize the loop on.
  const LaunchDimensions launch_dimensions_;
  const LaunchDimensionsConfig launch_config_;

  // An IR emitter that generates the loop body.
  llvm_ir::BodyEmitter body_emitter_;

  // The shape that the emitted loop iterates through.
  Shape shape_;

  // Points to the exit block of the emitted loop.
  llvm::BasicBlock* exit_bb_;

  llvm::IRBuilder<>* b_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_PARALLEL_LOOP_EMITTER_H_
