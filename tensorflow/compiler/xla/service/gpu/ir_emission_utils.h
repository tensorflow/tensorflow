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

#include <utility>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

// TODO(jlebar): Move functions related to cublas/cudnn to a separate file; they
// don't belong in "ir_emission_utils".

namespace xla {
namespace gpu {

// Different types of convolutions supported by cudnn.
//
// A way to think about these is that a convolution is defined by three arrays
// -- the "input", the "filter", and the "output" -- and given any two of these,
// we can compute the third.  For example, a backward-input convolution takes as
// input a filter and an "output" and produces an "input" such that if one were
// to do a forward convolution of "input" using filter, the result would be
// something with the same shape as "output".
//
// This way of thinking is not correct if you look at the values produced. For
// example, a backward-input convolution is not actually the mathematical
// inverse of a forward convolution.  But it's right as far as the shapes and
// "connectivity" (i.e. which elements of the input affect which elements of
// the output) are concerned.
enum class CudnnConvKind {
  kForward,            // input  + filter => output
  kBackwardInput,      // filter + output => input
  kBackwardFilter,     // input  + output => filter
  kForwardActivation,  // activation(conv(input, filter) + broadcast(bias) +
                       // (optionally) side_input) => output
};

StatusOr<CudnnConvKind> GetCudnnConvKind(const HloCustomCallInstruction* instr);

// Converts a CudnnConvKind value to a string.
string CudnnConvKindToString(CudnnConvKind kind);

// Matrix multiplication before the rewrite.
//
// This function should never return "true" on instructions after
// GemmRewriter pass has finished.
bool IsMatrixMultiplication(const HloInstruction& dot);

// Matrix multiplication rewritten into a GEMM custom call.
// All matrix multiplications should be rewritten as such custom calls
// after a GemmRewriter lowering pass.
bool IsCublasGemm(const HloInstruction& hlo);

constexpr int64 kWarpSize = 32;

// A call to cuBLAS general matrix multiplication API.
extern const char* const kGemmCallTarget;

// A call to cuDNN for batch normalization is represented as CustomCall HLO with
// a call target equal to one of these strings.
//
// The operands to and outputs of these calls are the same as those of the
// corresponding HLOs, except:
//
//  - epsilon and feature_index are proper operands, at the end of the operands
//    list.  They must be HLO constants.
//  - The cuDNN forward training call returns inv_stddev =
//    1/sqrt(variance + epsilon) in place of plain variance.
//  - Similarly, BatchNormGrad accepts inv_stddev in place of the variance
//    operand.
extern const char* const kCudnnBatchNormForwardInferenceCallTarget;
extern const char* const kCudnnBatchNormForwardTrainingCallTarget;
extern const char* const kCudnnBatchNormBackwardCallTarget;

// Returns true if `hlo` will be implemented as a call to a cuDNN batch
// normalization routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCudnnBatchNormFoo constants above, but returns *false* for HLOs
// with one of the kBatchNorm opcodes, because these are lowered either to a
// sequence of generic HLOs or to a cuDNN CustomCall.
bool IsCustomCallToDnnBatchNorm(const HloInstruction& hlo);

// A call to cuDNN for convolution (forward, backward filter, or backward input)
// is represented as a CustomCall HLO with a call target equal to one of these
// strings.
//
// These CustomCalls have window() and convolution_dimension_numbers() set like
// regular convolution ops.  They have the same LHS and RHS operands, plus two
// additional constant operands: an int64 operand for the cudnn algorithm and
// a bool operand for whether tensor_ops is enabled. A value of -1 for the cudnn
// algorithm means that the implementation is free to choose the best algorithm
// it can.
//
// These calls output a tuple (conv_result, scratch_memory), where conv_result
// is the actual result of the convolution, and scratch_memory is temporary
// memory used by cudnn.  Callers shouldn't inspect scratch_memory, as its value
// is not well-defined.
//
// GpuConvRewriter lowers kConvolution HLOs to these custom calls.
// When it does so, it chooses algorithm -1 and 0 bytes of scratch space.  Later
// on in the pipeline, CudnnConvAlgorithmChooser chooses an explicit
// algorithm for each conv and sets the amount of scratch space needed.
//
// (Representing the scratch memory as an output may seem strange at first, but
// it's quite sensible, from a certain point of view.  The scratch buffer is a
// location in memory that the conv can write into, but which it can't legally
// read from, at least until it's written something first.  But that's exactly
// the definition of an output buffer.)
extern const char* const kCudnnConvForwardCallTarget;
extern const char* const kCudnnConvBackwardInputCallTarget;
extern const char* const kCudnnConvBackwardFilterCallTarget;
extern const char* const kCudnnConvBiasActivationForwardCallTarget;

// Returns true if `hlo` will be implemented as a call to a cuDNN convolution
// routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCudnnConvFoo constants above, but returns *false* for HLOs with a
// kConvolution opcode.
bool IsCustomCallToDnnConvolution(const HloInstruction& hlo);

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

// Returns true if `hlo` will be implemented as a library call, e.g. cuBLAS gemm
// or cuDNN convolution.
bool ImplementedAsLibraryCall(const HloInstruction& hlo);

// Returns true if either the dimensions being reduced or the dimensions being
// kept are contiguous in the input of the reduce instruction.
bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce);

struct ReductionDimensions {
  // Indicates whether the reduction is a row reduction or a column reduction.
  bool is_row_reduction;

  // Contains the size of the three contiguous components for
  // the reduction [depth, height, width] (major-to-minor ordering).
  //
  // For row reduction, we do: [D, H, W] -> [D, H].
  // For column reduction, we do: [D, H, W] -> [D, W].
  std::array<int64, 3> dimensions;
};

// Given the input shape and dimensions to reduce for a reduction, returns
// ReductionDimensions.
//
// Prerequisite: the reduction instruction passes the check
// IsReductionFromOrToContiguousDimensions, which guarantees either the
// dimensions to reduce or the dimensions to keep are consecutive.
ReductionDimensions GetReductionKindAndContiguousComponents(
    const Shape& input_shape, absl::Span<const int64> dims_to_reduce);

// Emits call to "vprintf" with given format and arguments.
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder);

// Emits code to shuffle data between threads of a warp. This has the same
// semantics as the PTX "shfl.sync.down" instruction but works for values that
// aren't 32 bits in size. The last operand of the emitted "shfl" is
// `kWarpSize - 1`.
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

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
