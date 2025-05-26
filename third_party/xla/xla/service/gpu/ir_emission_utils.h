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

#ifndef XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
#define XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emission_utils.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {

// <HLO computation fingerprint, serialized compiled object>.
using BinaryMap = absl::flat_hash_map<std::string, std::string>;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
inline constexpr int64_t kMinDimensionToTransposeTiled = 4;
// If the product of the dimensions to be swapped is larger than
// 'kMinTotalDimensionsToTransposeTiled', tiled transposition may be more
// efficient. See go/xla-transpose-emitter-performance-analysis.
inline constexpr int64_t kMinTotalDimensionsToTransposeTiled = 16 * 16;
// As the amount of shared memory is limited, we need to make sure that we don't
// detect 102 transposes that would require too much bytes for the most minor
// dimension.
inline constexpr int64_t kMaxBitsInMostMinorDimension = 8 * 8;

// Returns true if the given dot is supported by cuBLAS.
absl::StatusOr<bool> IsCublasSupportedMatMul(
    const HloInstruction& dot, bool allow_matrix_vector_multiplication);

inline constexpr int64_t WarpSize(
    const se::DeviceDescription& gpu_device_info) {
  return gpu_device_info.threads_per_warp();
}

// Fusions that implemented with pre-compiled device kernels have
// FusionBackendConfig.kind requel to this string.
inline constexpr absl::string_view kCustomFusionKind = "__custom_fusion";

// Generic fusions that use Triton have FusionBackendConfig.kind equal to this
// string. This fusion kind will eventually subsume all usages of
// kTritonGemmFusionKind.
inline constexpr absl::string_view kTritonFusionKind = "__triton";

// Fusions that use Triton have FusionBackendConfig.kind equal to this string.
inline constexpr absl::string_view kTritonGemmFusionKind = "__triton_gemm";

// Generic fusions that use Triton have FusionBackendConfig.kind equal to this
// string. Used for fusions that implement a dot expressed as nested fusions.
inline constexpr absl::string_view kTritonNestedGemmFusionKind =
    "__triton_nested_gemm_fusion";

inline constexpr absl::string_view kCuDnnFusionKind = "__cudnn$fusion";

// Fusions that can be emitted using a dynamic memcpy. A dynamic memcpy depends
// on some loop induction variable.
inline constexpr absl::string_view kDynamicMemcpyFusionKind =
    "__dynamic_memcpy";

inline constexpr absl::string_view kUncompilableFusion =
    "__uncompilable_fusion";

inline constexpr absl::string_view kTopKCustomCallTarget = "__gpu$TopK";

// The number of shared memory banks.
inline constexpr int64_t kNumShmemBanks = 32;

// The bitwidth of a shared memory bank.
inline constexpr int64_t kBankBitwidth = 32;

// The name of the custom fusion config for dynamic slice fusion with static
// slices, such that the offset can be computed at compile time.
inline constexpr absl::string_view
    kDynamicSliceFusionWithStaticAddressComputationConfigName =
        "address_computation";
// The name of the custom fusion config for dynamic slice fusion with dynamic
// slices, such that the offset is computed at runtime.
inline constexpr absl::string_view
    kDynamicSliceFusionWithDynamicAddressComputationConfigName =
        "dynamic_address_computation";

// Returns the name of the custom fusion config if the given instruction is a
// custom fusion and has a custom fusion name, otherwise returns std::nullopt.
// The custom fusion name is basically the value of
// instr.backend_config().fusion_backend_config().custom_fusion_config().name().
// If any of this does not exist in the chain, then we return std::nullopt.
std::optional<std::string> GetCustomFusionConfigName(
    const HloInstruction* instr);

// Returns true if the given instruction is a custom fusion for dynamic slice
// fusion. This is determined by checking the name of custom fusion config.
bool IsDynamicSliceFusion(const HloInstruction* instr);

// Returns the bitwidth of the given primitive type. Unfortunately,
// primitive_util::BitWidth(PRED) return 1 instead of 8.
int GetBitwidth(PrimitiveType type);

// Returns true if the given instruction is a dynamic memcpy fusion. This
// function only checks the fusion kind, which is populated by the
// FusionDispatch pipeline.
bool IsDynamicMemcpyFusion(const HloInstruction* instr);

// Returns true if `hlo` will be implemented as a call to a cuSolver routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCusolver... constants, but returns *false* for HLOs with
// say, a kCholesky opcode.
bool IsCustomCallToCusolver(const HloInstruction& hlo);

// Returns true if `hlo` will be implemented as a call to a TopK routine.
bool IsCustomCallToTopK(const HloInstruction& hlo);

// Cholesky decomposition. Takes a (batched) matrix as input, and returns a
// tuple of (result, workspace, info), where result is the result of the
// Cholesky decomposition, workspace is scratch space for cuSolver, and info
// is a success/failure code per batch element.
extern const char* const kCusolverCholeskyCallTarget;

// Returns true if `instr` is a non-strided slice.
bool IsSliceWithUnitStrides(const HloInstruction* instr);

// Returns true if `instr` is a slice (or dynamic slice) instruction and
// operates on a contiguous slice of the input buffer.
bool IsContiguousSlice(const HloInstruction& instr);

// Emits code that determines whether the current thread is thread 0 within
// block 0 of the kernel.
llvm::Value* IsBlock0Thread0(llvm::IRBuilderBase* b);

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index);

// Returns whether the fusion represented by 'fusion_adaptor' can be emitted
// with the dynamic update slice in-place emitter. If 'fusion_adaptor'
// represents a single fusion computation, 'fusion' should provide the fusion
// instruction corresponding to that fusion computation. 'get_allocation_slice'
// is a callback for getting the allocated buffer slice, given an instruction
// and a shape index. This is ignored in case 'fusion' is a nullptr.
absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    const HloFusionAdaptor& fusion_adaptor,
    std::function<absl::StatusOr<BufferAllocation::Slice>(
        const HloInstruction* instr, const ShapeIndex& index)>
        get_allocation_slice,
    const HloInstruction* fusion = nullptr);

// Returns the dynamic-update-slice instructions defining the results of a
// fusion node. A dynamic slice update is said to be "defining" of a result if
// that result is the output of a dynamic slice update, or if that result is the
// output of a bitcast of a dynamic slice update---since such bitcast may be
// handled as a no-op.
std::vector<HloInstructionAdaptor> GetOutputDefiningDynamicUpdateSlices(
    absl::Span<HloInstructionAdaptor const> roots);

// Returns the first hero instruction reachable from `instr` as root. Hero
// instruction can be in a different computation if the parent HloFusionAdaptor
// is a producer-consumer fusion.
HloInstructionAdaptor FindNonTrivialHero(const HloInstructionAdaptor& instr);

// Same as above, but fusion is the parent computation of the hlo instruction.
const HloInstruction& FindNonTrivialHero(const HloInstruction& instr);

/// Description of how to emit a given transposition.
struct TransposeDescription {
  // Transpose instruction.
  const HloInstruction* instr;

  // Normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> dimensions;

  // Permutations of normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> permutation;

  // Required amount of shared memory in bytes.
  int64_t shmem_usage = 0;

  TransposeDescription(const HloInstruction* instr,
                       absl::InlinedVector<int64_t, 3> dimensions,
                       absl::InlinedVector<int64_t, 3> permutation,
                       int64_t shmem_usage)
      : instr(instr),
        dimensions(dimensions),
        permutation(permutation),
        shmem_usage(shmem_usage) {}

  // Transpose instruction input shape.
  const Shape& input_shape() const { return instr->operand(0)->shape(); }

  // Returns true, if both descriptions have the same dimensions and
  // permutation, even if they're produced by different instructions.
  bool IsEquivalent(const TransposeDescription& other) const {
    return dimensions == other.dimensions && permutation == other.permutation &&
           GetBitwidth(instr->shape().element_type()) ==
               GetBitwidth(other.instr->shape().element_type());
  }
};

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero);

// Canonical transpose permutes the input shape
// <D_0 x ... x D_n x T2 x D_{n+1} x ... x D_m x A x T1 x B> into
// <D'_0 x ... x D'_n' x T1 x D'_{n'+1} x ... x D'_m x A x T2 x B>.
// Note that the `D` dimensions are batch dimensions. They can also be
// permuted, but they are tiled by 1.
//
// Examples:
// 1. <8x32> -> <32x8> will be canonicalized to <8x1x32x1> -> <32x1x8x1>.
// 2. <8x2x32> -> <32x2x8> will be canonicalized to <8x2x32x1> -> <32x2x8x1>.
// 3. <8x2x32x7x6> -> <6x32x2x7x8> becomes <8x2x32x7x6x1> -> <6x32x2x7x8x1>.

// TODO(b/370690811): Unify this with TransposeDescription.
struct TransposeSpec {
  PrimitiveType elem_type() const { return input_shape().element_type(); }

  const Shape& input_shape() const { return transpose->operand(0)->shape(); }
  const Shape& output_shape() const { return transpose->shape(); }

  int64_t rank() const { return input_shape().dimensions().size(); }
  int64_t canonical_rank() const { return canonical_input_shape.size(); }

  int64_t dim_A() const { return canonical_input_shape[dim_A_id()]; }
  int64_t dim_A_id() const { return canonical_rank() - 3; }

  int64_t dim_B() const { return canonical_input_shape.back(); }
  int64_t dim_B_id() const { return canonical_rank() - 1; }

  int64_t dim_T1() const { return canonical_input_shape[dim_T1_input_id()]; }
  int64_t dim_T1_input_id() const { return canonical_rank() - 2; }
  int64_t dim_T1_output_id() const {
    return canonical_inv_permutation[canonical_rank() - 2];
  }

  int64_t dim_T2() const { return canonical_input_shape[dim_T2_input_id()]; }
  int64_t dim_T2_input_id() const {
    return canonical_permutation[canonical_rank() - 2];
  }
  int64_t dim_T2_output_id() const { return canonical_rank() - 2; }

  std::string ToString() const;

  const HloTransposeInstruction* transpose;

  llvm::SmallVector<int64_t, 3> permutation;
  llvm::SmallVector<int64_t, 3> inv_permutation;

  llvm::SmallVector<int64_t, 3> canonical_output_shape;
  llvm::SmallVector<int64_t, 3> canonical_permutation;
  llvm::SmallVector<int64_t, 3> canonical_inv_permutation;
  llvm::SmallVector<int64_t, 3> canonical_input_shape;
};

TransposeSpec GetTransposeSpec(const HloTransposeInstruction* transpose);

// Returns the default tile sizes for the packed transpose emitter.
absl::StatusOr<absl::InlinedVector<int64_t, 3>> GetPackedTransposeTileSizes(
    const TransposeSpec& spec);

// Checks if the instruction is elementwise.
bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count = 1);

// Log the given module if the VLOG level is >= level.
void VLogModule(int level, const llvm::Module& module);

// Verify the given module, and crash if it failed.
void VerifyModule(const llvm::Module& module);

// Returns the llvm type for the indices used in the kernel that contains the
// hlo instruction. Such indices include the index for the parallel loop and
// the indices for the tensors accessed by the kernel. The return type is i32
// iff the following conditions are met:
//  . The launch_size of the kernel is within the range of i32.
//  . The sizes of all the tensors accessed within the kernel are within the
//    range of i32.
// Otherwise, the return type is i64.
llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo,
                                  int64_t launch_size, llvm::IRBuilderBase* b);

// This class stores either a non-owning reference or owns data that represents
// a dense array in XLA format. It is used for intermediate storage during IR
// constant emission.
class DenseDataIntermediate {
 public:
  // Creates an instance of DenseDataIntermediate that owns the provided vector.
  static DenseDataIntermediate Own(std::vector<uint8_t> owned) {
    DenseDataIntermediate di;
    di.data_ = std::move(owned);
    return di;
  }

  // Creates an instance of DenseDataIntermediate that aliases the input.
  static DenseDataIntermediate Alias(absl::Span<const uint8_t> aliased) {
    DenseDataIntermediate di;
    di.data_ = aliased;
    return di;
  }

  // Returns a reference to the data this object represents.
  absl::Span<const uint8_t> span() const {
    return data_.index() == 0 ? absl::Span<const uint8_t>(std::get<0>(data_))
                              : std::get<1>(data_);
  }

  // Converts `this` into its protobuf representation.
  // Note that the protobuf message will always contain a copy of the data -
  // also for non-owning instances of DenseDataIntermediate.
  DenseDataIntermediateProto ToProto() const;

  // Constructs a data-owning instance of DenseDataIntermediate from its
  // protobuf representation.
  static DenseDataIntermediate FromProto(
      const DenseDataIntermediateProto& proto);

 private:
  std::variant<std::vector<uint8_t>, absl::Span<const uint8_t>> data_;
};

absl::StatusOr<DenseDataIntermediate> LiteralToXlaFormat(
    const Literal& literal);

// Returns a deterministic encoded string representation of the proto message.
absl::StatusOr<std::string> GetProtoFingerprint(
    const tsl::protobuf::MessageLite&);

// Returns concatenated fingerprint of an HLO instruction without its backend
// config and its backend config's deterministic fingerprint.
template <typename ConfigType>
absl::StatusOr<std::string> FingerprintWithBackendConfig(
    const HloInstruction& hlo) {
  TF_ASSIGN_OR_RETURN(const auto config, hlo.backend_config<ConfigType>());
  TF_ASSIGN_OR_RETURN(const std::string fingerprint,
                      GetProtoFingerprint(config));
  return absl::StrCat(hlo.ToString(HloPrintOptions::Fingerprint()),
                      ", backend_config_fingerprint=", fingerprint);
}

struct InductionVariableFunctionalDependency {
  // The dependency may be via multiple levels of intermediate calls. At each
  // level, we need to know which parameters to evaluate, since not all of them
  // may be relevant. The while loop's body is not included here, since the
  // induction variable is implicitly the only dependency that is allowed.
  // The size of the value is always the same as the number of parameters in the
  // computation. We request a single element of inlined space, which will
  // automatically pick the optimum value (16, usually).
  absl::flat_hash_map<const HloComputation*,
                      absl::InlinedVector<bool, 1 /* chosen automatically */>>
      required_parameters;

  // The loop and its induction variable that the value depends on.
  const HloInstruction* loop;
  const HloInstruction* induction_var;
};

// Checks if `instr`'s value is a pure function of a while loop's induction
// variable. This supports instructions that are inside call, async or fusion
// instructions. The dependency can be through arbitrary non-side-effecting
// instructions.
// Currently, this does not support nested while loops. Only dependencies on the
// inner-most while loop will successfully be analyzed.
// Requires `while_loop_trip_count_annotator` to have been run on the loop.
std::optional<InductionVariableFunctionalDependency>
ResolveFunctionalDependencyOnInductionVariable(const HloInstruction* instr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
