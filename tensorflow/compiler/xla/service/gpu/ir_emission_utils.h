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

// TODO(jlebar): Move functions related to cublas/cudnn to a separate file; they
// don't belong in "ir_emission_utils".

namespace xla {
namespace gpu {

constexpr int64 kWarpSize = 32;

// Returns true if `hlo` will be implemented as a call to BLAS gemm.
//
// Precondition: `hlo` is in an "unnested context", meaning, it lives within the
// entry computation, within the either of a while loop's subcomputations,
// within any of a conditional's subcomputations, etc., but *does not* live
// within a reduce subcomputation, a map subcomputation, a fusion
// subcomputation, etc.  It's OK if `hlo` *is* a fusion.
bool ImplementedAsGemm(const HloInstruction& hlo);

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
// CudnnConvolutionRewriter lowers kConvolution HLOs to these custom calls.
// When it does so, it chooses algorithm -1 and 0 bytes of scratch space.  Later
// on in the pipeline, CudnnConvolutionAlgorithmChooser chooses an explicit
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

// Returns true if `hlo` will be implemented as a call to a cuDNN convolution
// routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCudnnConvFoo constants above, but returns *false* for HLOs with a
// kConvolution opcode.
bool IsCustomCallToDnnConvolution(const HloInstruction& hlo);

// Creates a CustomCall for a cudnn forward/backward-input/backward-filter conv.
// Note that these CustomCalls return a tuple (conv_result, scratch_memory).  If
// you want just the conv result, you'll need to get-tuple-element the value
// returned by this function.
//
// The created cudnn call will use the default cudnn algorithm and no scratch
// space.
HloInstruction* CreateCudnnConvForward(
    const Shape& shape, HloInstruction* input, HloInstruction* kernel,
    const Window& window, const ConvolutionDimensionNumbers& dnums);
HloInstruction* CreateCudnnConvBackwardInput(
    const Shape& shape, HloInstruction* output, HloInstruction* reverse_filter,
    const Window& window, const ConvolutionDimensionNumbers& dnums);
HloInstruction* CreateCudnnConvBackwardFilter(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    const Window& window, const ConvolutionDimensionNumbers& dnums);

// Returns true if `hlo` will be implemented as a library call, e.g. cuBLAS gemm
// or cuDNN convolution.
bool ImplementedAsLibraryCall(const HloInstruction& hlo);

bool IsReductionToVector(const HloInstruction& reduce);

// Emits call to "vprintf" with given format and arguments.
llvm::Value* EmitPrintf(tensorflow::StringPiece fmt,
                        tensorflow::gtl::ArraySlice<llvm::Value*> arguments,
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

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
