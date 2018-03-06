/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_LOOP_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_LOOP_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

namespace xla {
namespace cpu {

// ParallelLoopEmitter emits a loop nest for the target array shape.
// The outer loop bounds of the loop nest are passed as ir values at runtime
// (specified in 'dynamic_loop_bounds'), and the inner loop bounds are static.
// Dynamic loop bounds are specified as an array of dimension index
// [start, limit) pairs of ir values (one for each partitioned outer dimension).
//
// EX: Let 'shape' = [8, 16, 32], with the loop bounds of the two-most major
//     dimensions dynamic. Then 'dynamic_loop_bounds' will contain the
//     following ir values for the two most-major dimensions:
//       [dim0_index_start_ir_value, dim0_index_limit_ir_value]
//       [dim1_index_start_ir_value, dim1_index_limit_ir_value]
//
// Code emitted by ParallelLoopEmitter will be called in a multi-threaded
// context where each thread will be assigned a different set of outer dimension
// partitions, and where all threads will collectively iterate over the
// entire target array shape.
//
// Outer dimension partitions can be generated using the ShapePartitionAssigner
// and ShapePartitionIterator utility classes from shape_partition.cc.
//
class ParallelLoopEmitter : public llvm_ir::LoopEmitter {
 public:
  // Constructs a ParallelLoopEmitter which uses 'target_element_generator' to
  // generate elements, 'dynamic_loop_bounds' to set the loop bounds of the
  // most-major dimensions, and 'target_array.' shape to set the static loop
  // bounds for the most-minor dimensions.
  ParallelLoopEmitter(const llvm_ir::ElementGenerator& target_element_generator,
                      const llvm_ir::IrArray& target_array,
                      const DynamicLoopBounds* dynamic_loop_bounds,
                      llvm::IRBuilder<>* ir_builder);

  ParallelLoopEmitter(const ParallelLoopEmitter&) = delete;
  ParallelLoopEmitter& operator=(const ParallelLoopEmitter&) = delete;
  ~ParallelLoopEmitter() override = default;

  llvm_ir::IrArray::Index EmitIndexAndSetExitBasicBlock(
      tensorflow::StringPiece loop_name) override;

 private:
  const DynamicLoopBounds* dynamic_loop_bounds_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_LOOP_EMITTER_H_
