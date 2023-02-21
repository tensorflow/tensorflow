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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
inline constexpr int64_t kMinDimensionToTransposeTiled = 16;

// Matrix multiplication before the rewrite.
//
// This function should never return "true" on instructions after
// GemmRewriter pass has finished.
bool IsMatrixMultiplication(const HloInstruction& dot);

inline constexpr int64_t WarpSize() { return 32; }

// Need at least 1024 threads/block for reasonable tree reduction
// performance (assuming all data fits).
inline constexpr int64_t MinThreadsXRowReduction() { return 1024; }

// When doing batched row reduction, how big the batch dimension could be.
inline constexpr int64_t BatchedReductionRaceFreeBound() { return 8; }

// Identifies Triton GEMM fusions.
bool IsTritonCustomCall(const HloInstruction& hlo);

inline constexpr absl::string_view kTritonCallTarget = "__triton";

// Returns true if `hlo` will be implemented as a call to a cuSolver routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCusolver... constants, but returns *false* for HLOs with
// say, a kCholesky opcode.
bool IsCustomCallToCusolver(const HloInstruction& hlo);

// Cholesky decomposition. Takes a (batched) matrix as input, and returns a
// tuple of (result, workspace, info), where result is the result of the
// Cholesky decomposition, workspace is scratch space for cuSolver, and info
// is a success/failure code per batch element.
extern const char* const kCusolverCholeskyCallTarget;

// Returns true if either the dimensions being reduced or the dimensions being
// kept are contiguous in the input of the reduce instruction.
bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce);

// Returns whether unnested_hlo is an input fusion whose root is either a slice
// or a tuple of slices. If verify_no_strides is true, returns false unless all
// ROOT slices have no strides.
bool IsInputFusibleSlices(mlir::Operation* unnested_hlo,
                          bool verify_no_strides);

struct ReductionDimensions {
  // Indicates whether the reduction is a row reduction or a column reduction.
  bool is_row_reduction;

  // Contains the size of the three contiguous components for
  // the reduction [depth, height, width] (major-to-minor ordering).
  //
  // For row reduction, we do: [D, H, W] -> [D, H].
  // For column reduction, we do: [D, H, W] -> [D, W].
  Vector3 dimensions;
};

// Given the input shape and dimensions to reduce for a reduction, returns
// ReductionDimensions.
//
// Prerequisite: the reduction instruction passes the check
// IsReductionFromOrToContiguousDimensions, which guarantees either the
// dimensions to reduce or the dimensions to keep are consecutive.
ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce);

// Get tiling per thread for the given reduction in dimensions [D, H, W].
Vector3 GetReductionTiling(const ReductionDimensions& reduction_dimensions);

// Emits call to "vprintf" with given format and arguments.
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder);

// Emits code to shuffle data between threads of a warp. This has the same
// semantics as the PTX "shfl.sync.down" instruction but works for values that
// aren't 32 bits in size. The last operand of the emitted "shfl" is
// `WarpSize() - 1`.
//
// This function emits a "full-warp" shuffle, which all threads of a warp
// participate in.  *Do not use this function from a divergent context:* You
// can't correctly do so on both Volta and earlier GPUs.
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync
llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder);

// Emits code that determines whether the current thread is thread 0 within
// block 0 of the kernel.
llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b);

inline std::string MlirToString(mlir::Operation* op) {
  std::string s;
  {
    llvm::raw_string_ostream os(s);
    op->print(os);
  }
  return s;
}

inline std::string MlirToString(const mlir::Location& loc) {
  std::string s;
  {
    llvm::raw_string_ostream os(s);
    loc.print(os);
  }
  return s;
}

int PartitionLmhloOperandsAndOutputs(mlir::Operation* op);
llvm::SmallVector<mlir::Value> GetHloOperands(mlir::Operation* op);
llvm::SmallVector<mlir::Value> GetHloOutputs(mlir::Operation* op);

bool WritesMlirBuffer(mlir::Operation* op, mlir::Value operand);

template <typename T>
std::vector<T> ToStdVector(const llvm::SmallVectorImpl<T>& v) {
  return std::vector<T>(v.begin(), v.end());
}

StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    mlir::Value v, absl::Span<const BufferAllocation> allocations,
    std::string* constant_name = nullptr);

bool CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const BufferAllocation> allocations);

Shape GetShape(mlir::Value value);

// Returns whether the given reduction can be safely generated without atomics:
// that is, at most one block will write to every output element.
bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions);

// Description of how to emit a given transposition.
//
// On a group of input parameters that are 0-2-1 transpose of the outputs
// of a fusion kernel, stores the input parameters that are safe for the
// shared memory transpose implementation and the dimension permutation.
//
// When a tile based shared memory transpose is used to implement an input
// with 0-2-1 transpose, we preload a tile of the input elements [z, y..y+31,
// x..x+31] to compute the output tile elements of the same indices.
// Preloading the input tile this way is only safe when the computation of the
// output tile elements do not need any input element outside the preloaded
// tile. We inspect all the transitive users of the input parameter up to the
// fusion root instruction to see if we can find any instruction that can make
// preloading the input tile unsafe.
struct TransposeDimsAndParams {
  // Permutation of the dimensions relative to output.
  Vector3 dims;

  // Indices of parameters which are permuted.
  std::vector<int64_t> params;

  std::string ToString() const {
    return absl::StrFormat("{dims={%s}, params={%s}}",
                           absl::StrJoin(dims, ", "),
                           absl::StrJoin(params, ", "));
  }
};

// Returns instructions which are roots of the fusion, following the operands of
// GTE instructions in the root tuple. Groups multiple subsequent instructions
// with the same root. CHECKs that the fusion never outputs the same instruction
// twice, as well as that there are no explicitly created tuples or nested gtes
// in fusion output.
//
// For input: (tuple (gte R1) (gte R1) O2)
// Expected output: [R1, O2]
//
// For input: (tuple R1 R2 O2)
// Expected output: [R1, R2, O2]
//
// For input: (tuple (gte R1) (gte R1) R2 O3)
// Expected output: [R1, R2, O3]
//
// For input: R1
// Expected output: [R1]
std::vector<HloInstruction*> GetFusionRoots(HloComputation* computation);

// Returns whether the computation has at least one root triggering unnested
// reduction emitter.
bool HasAnyUnnestedReductionRoot(HloComputation* computation);

const HloInstruction& FindNonTrivialHero(const HloInstruction& instr);

// Whether there is a fusion root triggering transposition emitter.
bool HasAnyTiledTransposeRoot(HloComputation* computation);

std::optional<Vector3> FindTiledLogicalTranspose(const HloInstruction& instr,
                                                 Vector3& permutation);

std::optional<std::pair<Vector3, Vector3>> FindAnyTiledTranspose(
    const HloInstruction& instr);

// Log and verify an LLVM module.
void LogAndVerify(const llvm::Module* m);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
