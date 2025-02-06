/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_IR_EMITTER2_H_
#define XLA_SERVICE_CPU_IR_EMITTER2_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::cpu {

// IrEmitter emits host kernels form HLO instructions into the LLVM module(s).
//
// Host kernel is simply a function that implements StreamExecutor HostKernel
// interface (defined as C API for ABI stability), and XLA:CPU runtime is
// responsible for launching host kernels on the host as a part of the Thunk
// sequence execution.
//
// In addition to a host kernel function itself, host kernel defines how much
// concurrency it can support by picking the right thread and block sizes.
// Runtime might launch host kernel blocks and threads on a thread pool, with an
// assumption that threads and blocks that are close to each other in three
// dimensional space are likely to touch the same memory, and thus should be
// executed on the same thread (or same NUMA node).
//
// At run time thunks resolve kernel functions by name in the compiled LLVM
// module.
//
// WARNING: This is under construction and will eventually replace IrEmitter.
class IrEmitter2 {
 public:
  friend class IrEmitter2Test;

 private:
  using KernelParameter = KernelApiIrBuilder::KernelParameter;
  using KernelPrototype = KernelApiIrBuilder::KernelPrototype;

 public:
  IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
             IrEmitter* nested_ir_emitter);

  // Emitted kernel information that defines how to launch it at run time.
  struct KernelInfo {
    explicit KernelInfo(KernelPrototype prototype,
                        const se::BlockDim& block_dims,
                        const se::ThreadDim& thread_dims);

    std::string name;
    se::BlockDim block_dims;
    se::ThreadDim thread_dims;
    absl::flat_hash_set<int64_t> invariant_arguments;
  };

  // Emitted comparator function information (for sort operation).
  struct ComparatorInfo {
    std::string name;
  };

  // Returns all the kernels emitted so far via this emitter.
  absl::Span<const KernelInfo> kernels() const { return kernels_; }

  absl::Span<const ComparatorInfo> comparators() const { return comparators_; }

  // Emits a host kernel for the pad instruction.
  absl::StatusOr<KernelInfo> EmitPadHostKernel(const HloInstruction* pad);

  // Emits a host kernel for the given fusion instruction.
  absl::StatusOr<KernelInfo> EmitFusionHostKernel(
      const HloFusionInstruction* fusion);

  // Emits a host kernel for the given dot fusion instruction (output fusion).
  absl::StatusOr<KernelInfo> EmitDotFusionHostKernel(
      const HloFusionInstruction* fusion);

  // Emits a host kernel for the given slice-to-dynamic instruction.
  absl::StatusOr<KernelInfo> EmitSliceToDynamicHostKernel(
      const HloInstruction* instr);

  // Emits a host kernel for the given dynamic-update-slice instruction.
  absl::StatusOr<KernelInfo> EmitDynamicUpdateSliceHostKernel(
      const HloInstruction* instr);

  // Emits a comparator function for the given sort instruction.
  absl::StatusOr<ComparatorInfo> EmitSortComparator(HloComputation* comparator);

  bool CanUpdateDynamicSliceInPlace(const HloInstruction* update) const;

 private:
  class ElementalIrEmitter;

  // Emits a host kernel prototype for the given HLO instruction.
  absl::StatusOr<KernelPrototype> EmitKernelPrototype(
      const HloInstruction* instr);

  // Parallel partition bounds for parallelized outer dimensions:
  //   vector<[i64 lower_bound, i64 upper_bound]>
  using ParallelPartitionBounds =
      std::vector<std::pair<llvm::Value*, llvm::Value*>>;

  // A config for running kernel in parallel. We rely on partitioning iteration
  // space along the outer dimension(s) and run each partition as a separate
  // task inside a runtime-managed thread pool.
  struct ParallelConfig {
    std::vector<int64_t> outer_dimension_partitions;
  };

  // Returns parallel config for the given instruction or std::nullopt if
  // the instruction has to be compiled to a single threaded loop.
  std::optional<ParallelConfig> GetParallelConfig(const HloInstruction* instr);

  // Emits LLVM IR that computes parallel partition bounds from the call frame's
  // block and thread dimensions and parallel execution config.
  ParallelPartitionBounds EmitParallelPartitionBounds(
      llvm::IRBuilderBase& b, const KernelPrototype& kernel_prototype,
      const ParallelConfig& parallel_config, const Shape& shape,
      absl::string_view name);

  // Emits LLVM IR using elemental loop emitter and the given element generator.
  // If the instruction is parallelized, it will emit a parallel loop partition
  // and return the requested number of execution threads.
  absl::StatusOr<se::ThreadDim> EmitElementalLoops(
      llvm::IRBuilderBase& b, const HloInstruction* instr,
      const KernelPrototype& kernel_prototype,
      const llvm_ir::ElementGenerator& element_generator);

  bool fast_min_max() const;

  // Returns the number of bytes within the shape.
  int64_t ByteSizeOf(const Shape& shape) const;

  // Given a load instruction, annotate the load's result with the invariant
  // load metadata.
  void AttachInvariantLoadMetadataForLoad(llvm::LoadInst* instr) const;

  CpuElementalIrEmitter ElementalIrEmmiterFactory(llvm::IRBuilderBase* b) const;

  const HloModule& hlo_module_;
  llvm::Module* module_;

  // Nested IrEmitter to emit embedded computations (e.g. computations attached
  // to reductions inside fusions).
  IrEmitter* nested_ir_emitter_;

  KernelApiIrBuilder kernel_api_ir_builder_;

  // Keeps track of all the functions emitted so far.
  std::vector<KernelInfo> kernels_;
  std::vector<ComparatorInfo> comparators_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_IR_EMITTER2_H_
