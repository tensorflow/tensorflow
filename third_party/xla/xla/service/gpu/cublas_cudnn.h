/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_CUBLAS_CUDNN_H_
#define XLA_SERVICE_GPU_CUBLAS_CUDNN_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"

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
  kForwardGraph,       // pointwise(...pointwise(conv(input, filter))...)
                       // => output
};

enum class CudnnNormKind {
  kLayerForwardInfer,
  kLayerForwardTrain,
  kLayerBackward,
};

enum class CudnnfMHAKind {
  kSoftmaxDropout,
  kSoftmax,
  kScaleBiasSoftmax,
  kScaleBiasSoftmaxDropout,
  kBackwardSoftmaxDropout,
  kBackwardSoftmax,
  kBackwardScaleBiasSoftmax,
  kBackwardScaleBiasSoftmaxDropout,
  kSoftmaxF8,
  kBackwardSoftmaxF8,
};

enum class CudnnfMHAMaskKind {
  kNoMask,
  kPadding,
  kCausal,
  kPaddingCausal,
  kAlibi,
};

absl::StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr);

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

// Scaled matrix multiplication in FP8. Calls into cublasLt.
bool IsCublasLtMatmulF8(const HloInstruction& hlo);

// Triangular solve that calls into legacy cublas.
bool IsTriangularSolve(const HloInstruction& hlo);

// A call to cuBLAS general matrix multiplication API.
extern const absl::string_view kGemmCallTarget;

// A call to cuBLAS Lt API matrix multiplication.
extern const absl::string_view kCublasLtMatmulCallTarget;

// A call to cuBLASLt for scaled matrix multiplication in FP8.
extern const absl::string_view kCublasLtMatmulF8CallTarget;

// A call to cuBLAS for a triangular solve.
//
// Like cudnn convolutions, this op returns a tuple (result, scratch_memory).
extern const absl::string_view kTriangularSolveCallTarget;

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
extern const absl::string_view kCudnnConvForwardCallTarget;
extern const absl::string_view kCudnnConvBackwardInputCallTarget;
extern const absl::string_view kCudnnConvBackwardFilterCallTarget;
extern const absl::string_view kCudnnConvBiasActivationForwardCallTarget;
extern const absl::string_view kCudnnConvForwardGraphCallTarget;

// cuDNN specific convolution helper (emitted together with a int8x32
// convolution, if reordering is required).
extern const absl::string_view kCudnnConvReorderFilterCallTarget;
extern const absl::string_view kCudnnConvReorderFilterAndBiasCallTarget;

// Returns true if `hlo` will be implemented as a call to a cuDNN convolution
// routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCudnnConvFoo constants above, but returns *false* for HLOs with a
// kConvolution opcode.
bool IsCustomCallToDnnConvolution(const HloInstruction& hlo);

// Returns true if `hlo` will be implemented as a call to cuDNN convolution
// reordering helper (required for int8x32 convolutions).
bool IsCudnnConvolutionReorder(const HloInstruction& hlo);

// A call to cuDNN for a fused norm.
extern const absl::string_view kCudnnNormCallTarget;

// Returns true if `hlo` will be implemented as a call to a cuDNN norm kernel.
bool IsCustomCallToDnnNorm(const HloInstruction& hlo);

// The fused_mha_rewriter phase where each of the MHA signatures are pattern
// matched and rewritten into a custom-call with specific custom-call target.
// The custom-call target specifies the MHA signature. For example,  BMM1 -Scale
// - Bias - Softmax - BMM2 pattern can have the target as
// cudnn$fmhaScaleBiasSoftmax. The fMHA signatures currently supported by cudnn
// are:
// 1. BMM1 - Softmax - BMM2
// 2. BMM1 - Softmax - Dropout - BMM2
// 3. BMM1 - scale - Bias - Softmax - BMM2
// 4. BMM1 - scale - Bias - Softmax - Dropout - BMM2
// Forward calls
extern const absl::string_view kCudnnfMHASoftmaxF8CallTarget;
extern const absl::string_view kCudnnfMHASoftmaxCallTarget;
extern const absl::string_view kCudnnfMHASoftmaxDropoutCallTarget;
extern const absl::string_view kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget;
extern const absl::string_view kCudnnfMHAScaleBiasSoftmaxCallTarget;
// Backward calls
extern const absl::string_view kCudnnfMHASoftmaxBackwardF8CallTarget;
extern const absl::string_view kCudnnfMHASoftmaxBackwardCallTarget;
extern const absl::string_view kCudnnfMHASoftmaxDropoutBackwardCallTarget;
extern const absl::string_view
    kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
extern const absl::string_view kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget;

bool IsFwdCustomCallTofMHAF8(const HloInstruction& hlo);
bool IsBwdCustomCallTofMHAF8(const HloInstruction& hlo);
bool IsCustomCallTofMHAF8(const HloInstruction& hlo);
bool IsFwdCustomCallTofMHA(const HloInstruction& hlo);
bool IsBwdCustomCallTofMHA(const HloInstruction& hlo);
bool IsCustomCallTofMHA(const HloInstruction& hlo);

absl::StatusOr<CudnnfMHAKind> GetCudnnfMHAKind(
    const HloCustomCallInstruction* instr);

std::string CudnnfMHAKindToString(CudnnfMHAKind kind);
absl::Status SetFMHAInstructionName(HloModule* module, HloInstruction* fmha);

bool MHACallHasDropout(absl::string_view fmha_call_name);

// A call to cuDNN for a block scaled dot.
extern const absl::string_view kCudnnBlockScaledDotCallTarget;

bool IsCustomCallToBlockScaledDot(const HloInstruction& hlo);

// CUB library calls.
// Reference: https://nvlabs.github.io/cub/
extern const absl::string_view kCubDeviceRadixSortTarget;

bool IsCubDeviceRadixSort(const HloInstruction& hlo);
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUBLAS_CUDNN_H_
