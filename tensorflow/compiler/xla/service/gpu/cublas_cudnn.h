/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_CUDNN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_CUDNN_H_

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/core/platform/statusor.h"

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
std::string CudnnConvKindToString(CudnnConvKind kind);

// Matrix multiplication rewritten into a GEMM custom call.
// All matrix multiplications should be rewritten as such custom calls
// after a GemmRewriter lowering pass.
bool IsCublasGemm(const HloInstruction& hlo);

// Matrix multiplication that calls into legacy cublas.
bool IsLegacyCublasMatmul(const HloInstruction& hlo);

// Matrix multiplication that calls into cublasLt.
bool IsCublasLtMatmul(const HloInstruction& hlo);

// A call to cuBLAS general matrix multiplication API.
extern const char* const kGemmCallTarget;

// A call to cuBLAS Lt API matrix multiplication.
extern const char* const kCublasLtMatmulCallTarget;

// A call to cuBLAS for a triangular solve.
//
// Like cudnn convolutions, this op returns a tuple (result, scratch_memory).
extern const char* const kTriangularSolveCallTarget;

// A call to cuDNN for convolution (forward, backward filter, or backward input)
// is represented as a CustomCall HLO with a call target equal to one of these
// strings.
//
// These CustomCalls have window() and convolution_dimension_numbers() set like
// regular convolution ops.  They have the same LHS and RHS operands, plus two
// additional constant operands: an int64_t operand for the cudnn algorithm and
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

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_CUDNN_H_
