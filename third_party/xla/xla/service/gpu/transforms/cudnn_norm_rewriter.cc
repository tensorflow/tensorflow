/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/cudnn_norm_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn.h"        // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn_version.h"
#endif

namespace xla {
namespace gpu {

namespace {

namespace m = match;

// Traverses the graph upward starting at instr and returns the
// first instruction that is not a convert, bitcast or reshape.
const HloInstruction* SkipUnaryOps(const HloInstruction* instr) {
  while (HloPredicateIsOp<HloOpcode::kConvert, HloOpcode::kBitcast,
                          HloOpcode::kReshape>(instr)) {
    instr = instr->operand(0);
  }
  return instr;
}

// Recursively traverses the graph downward starting at instr and stores in
// instrs the users that are not a convert, bitcast or reshape.
void SkipUnaryOpsTopDownRecursive(HloInstruction* instr,
                                  std::vector<HloInstruction*>& instrs) {
  if (HloPredicateIsOp<HloOpcode::kConvert, HloOpcode::kBitcast,
                       HloOpcode::kReshape>(instr)) {
    for (HloInstruction* user : instr->users()) {
      SkipUnaryOpsTopDownRecursive(user, instrs);
    }
  } else {
    instrs.push_back(instr);
  }
}

// Holds auxiliary information about individual layer norm patterns rewritten
// into a cuDNN Custom Call.
struct NormMetadata {
  // Transposes applied to the input and output of the forward layer norm to
  // order the normalization and non-normalization dimensions as required by
  // cuDNN. Nullptr if no transposes were inserted.
  HloInstruction *x_transpose, *y_transpose;
  // The reduction and non-reduction dimensions of the input into the forward
  // layer norm before the potential application of transposes and adjusted for
  // the removal of any degenerate dimensions in the input to the norm.
  std::vector<int64_t> norm_dims_adjusted, non_norm_dims_adjusted;
};

// Map from the instruction pointer of a layer norm Custom Call to its metadata.
using NormMetadataMap = absl::flat_hash_map<HloInstruction*, NormMetadata>;

// Captures multiple HloInstruction pointers and verifies that their target
// is identical.
//
// Example:
// Pattern cos(x) / sin(x) with cos and sin intended to operate on the same
// HloInstruction:
//  UniqueHloInstruction x;
//  bool m = Match(
//      instr, m::Divide(m::Cos(m::Op().WithPredicate(x.CaptureOrVerifyFn())),
//                       m::Sin(m::Op().WithPredicate(x.CaptureOrVerifyFn()))));
// m is true and x.Instr() returns an HloInstruction pointer to the operand of
// cosine and sine iff HloInstruction *instr points to a division of a cosine by
// a sine that operate on the same instruction.
class UniqueHloInstruction {
 public:
  UniqueHloInstruction()
      : is_set_(false),
        instr_(nullptr),
        capture_or_verify_([this](const HloInstruction* instr) -> bool {
          return CaptureOrVerify(const_cast<HloInstruction*>(instr));
        }) {}
  HloInstruction* Instr() const { return instr_; }
  void SetInstr(HloInstruction* instr) {
    is_set_ = true;
    instr_ = instr;
  }

  // Stores instr when invoked the first time. Otherwise, compares instr to the
  // stored value and sets the stored value to nullptr if the comparison fails.
  bool CaptureOrVerify(HloInstruction* instr) {
    if (is_set_ && instr != instr_) {
      instr_ = nullptr;
    }
    if (!is_set_) {
      is_set_ = true;
      instr_ = instr;
    }
    return instr_;
  }

  // Returns a std::function for capturing or verifying an instruction using
  // WithPredicate.
  std::function<bool(const HloInstruction*)> CaptureOrVerifyFn() const {
    return capture_or_verify_;
  }

 private:
  bool is_set_;
  HloInstruction* instr_;
  std::function<bool(const HloInstruction*)> capture_or_verify_;
};

// Returns an architecture-specific constant for the calculation of an upper
// bound for the size of the scratch space for layer norm kernels.
absl::StatusOr<int64_t> CConstant(
    se::CudaComputeCapability cuda_compute_capability) {
  if (cuda_compute_capability.major == se::CudaComputeCapability::kAmpere) {
    return 32 * 128;
  } else if (cuda_compute_capability.major ==
             se::CudaComputeCapability::kHopper) {
    return 32 * 144;
  }
  return xla::Internal("Norm kernels require Ampere or Hopper architecture.");
}

// Returns whether the element type of instr is compatible with layer norm
// kernels.
bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F16 || element_type == F32;
}

// Returns the dimensions associated with shape, adjusted for the removal of any
// degenerate dimensions in shape. Specifically, for each dimension d in
// dimensions, returns the new index of d if all dimensions of size 1 are
// removed from shape. If d has size 1, it is not included in the returned
// vector.
std::vector<int64_t> AdjustedDimensions(const Shape& shape,
                                        absl::Span<const int64_t> dimensions) {
  absl::flat_hash_map<int64_t, int64_t> dimension_map;
  for (int64_t dimension = 0, non_degen_dimension = 0;
       dimension < shape.dimensions().size(); ++dimension) {
    if (shape.dimensions(dimension) > 1) {
      dimension_map.insert({dimension, non_degen_dimension});
      non_degen_dimension++;
    }
  }
  std::vector<int64_t> adjusted_dimensions;
  for (int64_t dimension : dimensions) {
    auto non_degenerate_dimension = dimension_map.find(dimension);
    if (non_degenerate_dimension != dimension_map.end()) {
      adjusted_dimensions.emplace_back(non_degenerate_dimension->second);
    }
  }
  return adjusted_dimensions;
}

// Returns the dimensions of broadcast or reduction instructions, adjusted for
// the removal of any degenerate dimensions in the output or input.
std::vector<int64_t> AdjustedDimensions(const HloInstruction* instr) {
  Shape shape;
  if (HloPredicateIsOp<HloOpcode::kBroadcast>(instr)) {
    shape = instr->shape();
  } else if (HloPredicateIsOp<HloOpcode::kReduce>(instr)) {
    shape = instr->operand(0)->shape();
  } else {
    return {};
  }
  return AdjustedDimensions(shape, instr->dimensions());
}

// Returns whether the HLO Computation applied by instr calculates the sum of
// the elements. When provided, compares reduce_dims to the dimensions of the
// reduction.
bool AppliesAddReduce(const HloInstruction* instr,
                      absl::Span<const int64_t> reduce_dims = {}) {
  if (HloPredicateIsNotOp<HloOpcode::kReduce>(instr)) {
    return false;
  }

  // Verify the dimensions of the reduction.
  if (!reduce_dims.empty() && AdjustedDimensions(instr) != reduce_dims) {
    return false;
  }

  HloComputation* reduce_comp = instr->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return instr->operand_count() == 2 &&
         instr->operand(1)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsScalar(instr->operand(1)->shape()) &&
         instr->operand(1)->literal().GetAsDouble({}) == 0. &&
         HloPredicateIsOp<HloOpcode::kAdd>(reduce_comp_root) &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}

// Returns whether instr multiplies the result of a reduction by one over the
// number of reduced elements.
bool CalculatesExpectation(const HloInstruction* instr) {
  instr = SkipUnaryOps(instr);
  if (HloPredicateIsNotOp<HloOpcode::kMultiply>(instr)) {
    return false;
  }
  bool bcast_operand = instr->operand(0)->opcode() != HloOpcode::kBroadcast;
  const HloInstruction *broadcast = instr->operand(bcast_operand),
                       *reduce = SkipUnaryOps(instr->operand(!bcast_operand));
  if (HloPredicateIsNotOp<HloOpcode::kReduce>(reduce) ||
      HloPredicateIsNotOp<HloOpcode::kBroadcast>(broadcast) ||
      broadcast->operand(0)->opcode() != HloOpcode::kConstant) {
    return false;
  }

  float actual_r_nelems =
      broadcast->operand(0)->literal().GetAsDouble({}).value();
  int64_t nelems = 1;
  for (int64_t norm_dim : reduce->dimensions()) {
    nelems *= reduce->operand(0)->shape().dimensions()[norm_dim];
  }
  // The absolute of the difference between the actual scaling factor and the
  // reference value must not exceed a prescribed threshold.
  float r_nelems = 1. / static_cast<float>(nelems);
  float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
  return abs(actual_r_nelems - r_nelems) <
         ((actual_r_nelems + r_nelems) * numerical_epsilon);
}

// Returns whether target can be reached from instr by recursively traversing
// the graph across converts, bitcasts and reshapes.
bool FindTargetRecursive(
    const HloInstruction* instr, const HloInstruction* target,
    absl::flat_hash_set<const HloInstruction*>& visited_instrs,
    const HloInstruction* transpose) {
  visited_instrs.emplace(instr);
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape};
  if (instr == target) {
    return true;
  }
  // Look for target among the users of instr.
  for (HloInstruction* user : instr->users()) {
    if ((supported_ops.contains(user->opcode()) || user == transpose) &&
        !visited_instrs.contains(user)) {
      return FindTargetRecursive(user, target, visited_instrs, transpose);
    }
  }
  // Ascend the graph if target is not found and instr is a convert, bitcast
  // or reshape.
  if (supported_ops.contains(instr->opcode())) {
    return FindTargetRecursive(instr->operand(0), target, visited_instrs,
                               transpose);
  }
  return false;
}

bool FindTarget(const HloInstruction* custom_call, const HloInstruction* instr,
                const HloInstruction* target,
                const NormMetadataMap& norm_metadata) {
  absl::flat_hash_set<const HloInstruction*> visited_instrs;
  auto custom_call_metadata = norm_metadata.find(custom_call);
  if (custom_call_metadata == norm_metadata.end()) {
    return false;
  }
  return FindTargetRecursive(instr, target, visited_instrs,
                             custom_call_metadata->second.x_transpose);
}

// Maps the dimension numbers in dimensions from shape original_shape to shape
// reshaped_shape, assuming that the shapes are related through a strict
// reshape. Returns an empty vector if a dimension mapping is not found.
std::vector<int64_t> MapDimensions(const Shape& original_shape,
                                   const Shape& reshaped_shape,
                                   const absl::Span<const int64_t> dimensions) {
  auto dimension_product =
      [](const Shape& shape,
         absl::Span<const int64_t> product_dimensions) -> int64_t {
    int64_t product = 1;
    for (int64_t product_dimension : product_dimensions) {
      product *= shape.dimensions(product_dimension);
    }
    return product;
  };
  // Construct the dimension mapping.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> dimensions_map;
  std::vector<int64_t> original_dimensions, reshaped_dimensions;
  for (int64_t original_dimension = 0, reshaped_dimension = 0;
       original_dimension < original_shape.dimensions().size();
       ++original_dimension) {
    original_dimensions.push_back(original_dimension);
    while ((reshaped_dimensions.empty() ||
            dimension_product(reshaped_shape, reshaped_dimensions) <
                dimension_product(original_shape, original_dimensions)) &&
           reshaped_dimension < reshaped_shape.dimensions().size()) {
      reshaped_dimensions.emplace_back(reshaped_dimension++);
    }

    // Many-to-many dimension mappings are not supported.
    if (original_dimensions.size() > 1 && reshaped_dimensions.size() > 1) {
      return {};
    }

    if (dimension_product(original_shape, original_dimensions) ==
        dimension_product(reshaped_shape, reshaped_dimensions)) {
      std::vector<int64_t> original_dimensions_in_dimensions;
      std::set_intersection(
          original_dimensions.begin(), original_dimensions.end(),
          dimensions.begin(), dimensions.end(),
          std::back_inserter(original_dimensions_in_dimensions));
      // The unique mapping of dimensions requires either all or none of the
      // entries of original_dimensions to be an element of dimensions.
      if (!original_dimensions_in_dimensions.empty() &&
          original_dimensions_in_dimensions.size() !=
              original_dimensions.size()) {
        return {};
      }
      for (int64_t dimension : original_dimensions) {
        dimensions_map.insert({dimension, reshaped_dimensions});
      }
      original_dimensions.clear();
      reshaped_dimensions.clear();
    }
  }

  // Map the dimensions numbers to the reshaped shape.
  std::vector<int64_t> mapped_dimensions;
  for (int64_t dimension : dimensions) {
    auto mapped_dimension = dimensions_map.find(dimension);
    if (mapped_dimension == dimensions_map.end()) {
      return {};
    }
    mapped_dimensions.insert(mapped_dimensions.end(),
                             mapped_dimension->second.begin(),
                             mapped_dimension->second.end());
  }

  // Eliminate duplicates in the mapped dimension numbers.
  mapped_dimensions.erase(
      std::unique(mapped_dimensions.begin(), mapped_dimensions.end()),
      mapped_dimensions.end());
  return mapped_dimensions;
}

// Recursively traverses the graph across converts, bitcasts and reshapes,
// starting from instr, and returns the first addition-reduction identified.
// Returns nullptr if no addition-reduction is found.
HloInstruction* FindAddReduceRecursive(
    HloInstruction* instr, const Shape& orig_instr_shape,
    const absl::Span<const int64_t> reduce_dims,
    absl::flat_hash_set<HloInstruction*>& visited_instrs) {
  visited_instrs.emplace(instr);
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape};
  // Look for a reduction among the users of instr.
  for (HloInstruction* user : instr->users()) {
    if (HloPredicateIsOp<HloOpcode::kReduce>(user)) {
      std::vector<int64_t> mapped_reduce_dims =
          MapDimensions(orig_instr_shape, instr->shape(), reduce_dims);
      if (!mapped_reduce_dims.empty() &&
          AppliesAddReduce(user, mapped_reduce_dims)) {
        return user;
      }
    }
    if (supported_ops.contains(user->opcode()) &&
        !visited_instrs.contains(user)) {
      return FindAddReduceRecursive(user, orig_instr_shape, reduce_dims,
                                    visited_instrs);
    }
  }
  // Ascend the graph if the addition-reduction is not found and instr is a
  // convert, bitcast or reshape.
  if (supported_ops.contains(instr->opcode())) {
    return FindAddReduceRecursive(instr->mutable_operand(0), orig_instr_shape,
                                  reduce_dims, visited_instrs);
  }
  return nullptr;
}

HloInstruction* FindAddReduce(HloInstruction* instr,
                              const absl::Span<const int64_t> reduce_dims) {
  absl::flat_hash_set<HloInstruction*> visited_instrs;
  return FindAddReduceRecursive(instr, instr->shape(), reduce_dims,
                                visited_instrs);
}

// Type conversion from and to any of BF16, FP16 and FP32.
template <typename Pattern>
auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

// Bitcast or reshape adding or removing degenerate dimensions.
template <typename Pattern>
auto SupportedBitcastOrReshape(Pattern pattern) {
  auto supported_bitcast_or_reshape = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::Equal(
        ShapeUtil::DropDegenerateDimensions(instr->shape()),
        ShapeUtil::DropDegenerateDimensions(instr->operand(0)->shape()));
  };
  return m::AnyOf<HloInstruction>(
      m::Bitcast(pattern).WithPredicate(supported_bitcast_or_reshape),
      m::Reshape(pattern).WithPredicate(supported_bitcast_or_reshape));
}

// Matches pattern, SupportedConvert(pattern),
// SupportedBitcastOrReshape(pattern),
// SupportedConvert(SupportedBitcastOrReshape(pattern)) and
// SupportedBitcastOrReshape(SupportedConvert(pattern)).
template <typename Pattern>
auto OptionalSupportedTransform(Pattern pattern) {
  auto shared_subpattern = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(
      SupportedConvert(SupportedBitcastOrReshape(shared_subpattern)),
      SupportedBitcastOrReshape(SupportedConvert(shared_subpattern)),
      SupportedConvert(shared_subpattern),
      SupportedBitcastOrReshape(shared_subpattern), shared_subpattern);
}

// Bitcast or reshape with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern>
auto BitcastOrReshape(Pattern pattern) {
  return OptionalSupportedTransform(
      m::AnyOf<HloInstruction>(m::Bitcast(pattern), m::Reshape(pattern)));
}

// Transpose with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern>
auto Transpose(Pattern pattern) {
  return OptionalSupportedTransform(m::Transpose(pattern));
}

// Rsqrt with optional supported type conversion and/or addition or removal of
// degenerate dimensions.
template <typename Pattern>
auto Rsqrt(HloInstruction** rsqrt, Pattern pattern) {
  return OptionalSupportedTransform(m::Rsqrt(rsqrt, pattern));
}

// AddAnyOrder with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto AddAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::AddAnyOrder(pattern0, pattern1));
}

// Subtract with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto Subtract(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(pattern0, pattern1));
}

// Capturing subtract with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto Subtract(HloInstruction** subtract, Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(subtract, pattern0, pattern1));
}

// Multiply with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::MultiplyAnyOrder(pattern0, pattern1));
}

// Capturing multiply with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(HloInstruction** multiply, Pattern0 pattern0,
                      Pattern1 pattern1) {
  return OptionalSupportedTransform(
      m::MultiplyAnyOrder(multiply, pattern0, pattern1));
}

// Multiplication of pattern by itself with optional supported type conversion
// and/or addition or removal of degenerate dimensions.
template <typename Pattern>
auto Square(Pattern pattern) {
  return MultiplyAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}

// Multiplication of the square of pattern by pattern with optional supported
// type conversion and/or addition or removal of degenerate dimensions. The root
// instruction of pattern cannot be a multiplication.
template <typename Pattern>
auto Cube(Pattern pattern) {
  auto unique_cube = [](const HloInstruction* instr) -> bool {
    bool square_operand = instr->operand(0)->opcode() != HloOpcode::kMultiply;
    return instr->operand(!square_operand)->opcode() != HloOpcode::kMultiply &&
           instr->operand(square_operand)->operand(0) ==
               instr->operand(!square_operand);
  };
  return MultiplyAnyOrder(Square(pattern), pattern).WithPredicate(unique_cube);
}

// Addition-reduction of pattern with optional supported type conversion and/or
// addition or removal of degenerate dimensions.
template <typename Pattern>
auto AddReduce(Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}

// Capturing addition-reduction of pattern with optional supported type
// conversion and/or addition or removal of degenerate dimensions.
template <typename Pattern>
auto AddReduce(HloInstruction** reduction, Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(reduction, pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}

// Negated addition-reduction.
template <typename Pattern>
auto NegateAddReduce(HloInstruction** reduction, Pattern pattern) {
  return m::AnyOf<HloInstruction>(AddReduce(reduction, m::Negate(pattern)),
                                  m::Negate(AddReduce(reduction, pattern)));
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(UniqueHloInstruction* expectation, Pattern pattern) {
  auto shared_subpattern = OptionalSupportedTransform(
      m::MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          })
          .WithPredicate(expectation->CaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(UniqueHloInstruction* expectation, HloInstruction** reduce,
                 Pattern pattern) {
  auto shared_subpattern = OptionalSupportedTransform(
      m::MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()),
                          AddReduce(reduce, pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          })
          .WithPredicate(expectation->CaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Variance, expressed as expectation(X^2) - expectation(X)^2 or
// expectation((X - expectation(X))^2).
auto Variance(UniqueHloInstruction* variance, UniqueHloInstruction* expectation,
              UniqueHloInstruction* x) {
  return m::AnyOf<HloInstruction>(
      Subtract(
          Expectation(Square(OptionalSupportedTransform(
              m::Op().WithPredicate(x->CaptureOrVerifyFn())))),
          Square(Expectation(
              expectation, OptionalSupportedTransform(
                               m::Op().WithPredicate(x->CaptureOrVerifyFn())))))
          .WithPredicate(variance->CaptureOrVerifyFn()),
      Expectation(
          Square(Subtract(
              OptionalSupportedTransform(
                  m::Op().WithPredicate(x->CaptureOrVerifyFn())),
              Expectation(expectation,
                          OptionalSupportedTransform(
                              m::Op().WithPredicate(x->CaptureOrVerifyFn()))))))
          .WithPredicate(variance->CaptureOrVerifyFn()));
}

// Reciprocal of the square root of variance + epsilon with optional broadcast.
auto NormFactor(HloInstruction** norm_factor, UniqueHloInstruction* x,
                UniqueHloInstruction* variance,
                UniqueHloInstruction* expectation,
                UniqueHloInstruction* epsilon) {
  auto shared_subpattern = m::SharedSubpattern(Rsqrt(
      norm_factor, AddAnyOrder(Variance(variance, expectation, x),
                               m::Broadcast(m::ConstantScalar().WithPredicate(
                                   epsilon->CaptureOrVerifyFn())))));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Any order of p0 * p1 * p2.
template <typename P0, typename P1, typename P2>
auto MultiplyMultiplyAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, MultiplyAnyOrder(p1, p2)),
      MultiplyAnyOrder(p1, MultiplyAnyOrder(p0, p2)),
      MultiplyAnyOrder(p2, MultiplyAnyOrder(p0, p1)));
}

// Any order of p0 + p1 + p2.
template <typename P0, typename P1, typename P2>
auto AddAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(p0, AddAnyOrder(p1, p2)),
                                  AddAnyOrder(p1, AddAnyOrder(p0, p2)),
                                  AddAnyOrder(p2, AddAnyOrder(p0, p1)));
}

// Any order of p0 * (p1 + p2).
template <typename P0, typename P1, typename P2>
auto MultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, AddAnyOrder(p1, p2)),
      AddAnyOrder(MultiplyAnyOrder(p0, p1), MultiplyAnyOrder(p0, p2)));
}

// Any order of p0 - p1 + p2.
template <typename P0, typename P1, typename P2>
auto SubtractAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(Subtract(p0, p1), p2),
                                  AddAnyOrder(Subtract(p2, p1), p0),
                                  Subtract(AddAnyOrder(p0, p2), p1));
}

// Any order of (p0 - p1) * p2 * p3 + p4.
template <typename P0, typename P1, typename P2, typename P3, typename P4>
auto SubtractMultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2, P3 p3, P4 p4) {
  return m::AnyOf<HloInstruction>(
      SubtractAddAnyOrder(MultiplyMultiplyAnyOrder(p0, p2, p3),
                          MultiplyMultiplyAnyOrder(p1, p2, p3), p4),
      AddAnyOrder(MultiplyMultiplyAnyOrder(Subtract(p0, p1), p2, p3), p4));
}

// Expectation fused into a layer norm Custom Call.
auto FusedExpectation(UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(m::CustomCall({kCudnnNormCallTarget})
                             .WithPredicate(custom_call->CaptureOrVerifyFn()),
                         1));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}

// Expectation fused into a layer norm Custom Call.
auto FusedExpectation(UniqueHloInstruction* fused_expectation,
                      UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(m::CustomCall({kCudnnNormCallTarget})
                             .WithPredicate(custom_call->CaptureOrVerifyFn()),
                         1)
          .WithPredicate(fused_expectation->CaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}

// Norm factor fused into a layer norm Custom Call.
auto FusedNormFactor(UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(m::CustomCall({kCudnnNormCallTarget})
                             .WithPredicate(custom_call->CaptureOrVerifyFn()),
                         2));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}

// Norm factor fused into a layer norm Custom Call.
auto FusedNormFactor(UniqueHloInstruction* fused_norm_factor,
                     UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(m::CustomCall({kCudnnNormCallTarget})
                             .WithPredicate(custom_call->CaptureOrVerifyFn()),
                         2)
          .WithPredicate(fused_norm_factor->CaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}

// Derivative of the norm factor w.r.t. variance + epsilon,
// d(norm_factor)/d(variance + epsilon)
// = d((variance + epsilon)^-1/2)/d(variance + epsilon)
// = -1/2 * norm_factor^3.
// Forwards custom_call to FusedNormFactor for verification.
auto DNormFactor(UniqueHloInstruction* custom_call) {
  return MultiplyAnyOrder(m::Broadcast(m::ConstantScalar(-0.5)),
                          Cube(FusedNormFactor(custom_call)));
}

//  Zero-centered input of the layer norm, X - expectation(X). Verifies that
//  custom_call is a forward layer norm fusing X. Forwards custom_call to
//  FusedExpectation for verification.
auto XCenter(UniqueHloInstruction* x, UniqueHloInstruction* custom_call,
             const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_x =
      [x, custom_call, &norm_metadata](const HloInstruction* instr) -> bool {
    return x->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(0), norm_metadata)
            ? custom_call->Instr()->mutable_operand(0)
            : nullptr);
  };
  return Subtract(m::Op(), m::Broadcast(FusedExpectation(custom_call)))
      .WithPredicate(capture_or_verify_x);
}

// Zero-centered input of the layer norm, X - expectation(X). Captures X in x if
// custom_call is a forward layer norm fusing X. Forwards custom_call to
// FusedExpectation for comparison.
auto XCenter(UniqueHloInstruction* x_center, UniqueHloInstruction* x,
             UniqueHloInstruction* fused_expectation,
             UniqueHloInstruction* custom_call,
             const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_x =
      [x, custom_call, &norm_metadata](const HloInstruction* instr) -> bool {
    return x->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(0), norm_metadata)
            ? custom_call->Instr()->mutable_operand(0)
            : nullptr);
  };
  return Subtract(m::Op(), m::Broadcast(FusedExpectation(fused_expectation,
                                                         custom_call)))
      .WithPredicate(x_center->CaptureOrVerifyFn())
      .WithPredicate(capture_or_verify_x);
}

// Addition-reduction of the product of XCenter, the broadcasted scale and DY,
// XCenter * scale * DY. Captures the scale in scale if custom_call is a forward
// layer norm fusing the scale. Forwards custom_call to XCenter for comparison.
auto F0(UniqueHloInstruction* custom_call, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, UniqueHloInstruction* x,
        HloInstruction** reduce, const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_scale = [scale, custom_call, &norm_metadata](
                                     const HloInstruction* instr) -> bool {
    return scale->CaptureOrVerify(FindTarget(custom_call->Instr(), instr,
                                             custom_call->Instr()->operand(1),
                                             norm_metadata)
                                      ? custom_call->Instr()->mutable_operand(1)
                                      : nullptr);
  };
  return AddReduce(
      reduce, MultiplyMultiplyAnyOrder(
                  XCenter(x, custom_call, norm_metadata),
                  m::Broadcast(m::Op().WithPredicate(capture_or_verify_scale)),
                  m::Op().WithPredicate(dy->CaptureOrVerifyFn())));
}

// Product of XCenter and the scaled and broadcasted product of F0 and
// d(norm_factor)/d(variance + epsilon), XCenter * F0 * DNormFactor * 2 /
// nelems. Forwards custom_call to XCenter, F0 and DNormFactor for capture or
// verification.
auto F1(UniqueHloInstruction* x, UniqueHloInstruction* x_center,
        UniqueHloInstruction* fused_expectation,
        UniqueHloInstruction* custom_call, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, HloInstruction** reduce,
        const NormMetadataMap& norm_metadata) {
  auto broadcasts_two_over_nelems = [](const HloInstruction* instr) -> bool {
    const HloInstruction* multiply = SkipUnaryOps(instr->operand(0));
    bool bcast_operand =
        multiply->operand(0)->opcode() != HloOpcode::kBroadcast;

    // The captured scalar must be two over the number of elements in the
    // broadcasted dimensions.
    float actual_two_over_nelems = multiply->operand(bcast_operand)
                                       ->operand(0)
                                       ->literal()
                                       .GetAsDouble({})
                                       .value();
    int64_t nelems = 1;
    for (int i = 0; i < instr->shape().dimensions().size(); ++i) {
      if (!absl::c_linear_search(instr->dimensions(), i)) {
        nelems *= instr->shape().dimensions()[i];
      }
    }
    // The absolute of the difference between the actual scaling factor and the
    // reference value must not exceed a prescribed threshold.
    float two_over_nelems = 2. / static_cast<float>(nelems);
    float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
    return abs(actual_two_over_nelems - two_over_nelems) <
           ((actual_two_over_nelems + two_over_nelems) * numerical_epsilon);
  };

  return MultiplyAnyOrder(
      XCenter(x_center, x, fused_expectation, custom_call, norm_metadata),
      m::Broadcast(
          MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()),
                           MultiplyAnyOrder(DNormFactor(custom_call),
                                            F0(custom_call, scale, dy, x,
                                               reduce, norm_metadata))))
          .WithPredicate(broadcasts_two_over_nelems));
}

// Product of the norm factor, scale and DY, NormFactor * scale * DY. Captures
// the scale in scale if custom_call is a forward layer norm fusing the scale.
// Forwards custom_call to FusedNormFactor for comparison.
auto F2(UniqueHloInstruction* fused_norm_factor, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, UniqueHloInstruction* custom_call,
        const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_scale = [scale, custom_call, &norm_metadata](
                                     const HloInstruction* instr) -> bool {
    return scale->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(1), norm_metadata)
            ? custom_call->Instr()->mutable_operand(1)
            : nullptr);
  };
  return MultiplyAnyOrder(
      m::Broadcast(
          BitcastOrReshape(FusedNormFactor(fused_norm_factor, custom_call))),
      MultiplyAnyOrder(m::Broadcast().WithPredicate(capture_or_verify_scale),
                       m::Op().WithPredicate(dy->CaptureOrVerifyFn())));
}

class CudnnNormRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNormRewriterVisitor(
      const se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}

  absl::Status HandleAdd(HloInstruction* instr) override {
    TF_RETURN_IF_ERROR(MatchLayerNorm(instr));
    TF_RETURN_IF_ERROR(MatchLayerNormGradient(instr));
    return absl::OkStatus();
  }

  absl::Status HandleSubtract(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  // Matches and rewrites layer norm patterns,
  // Y = (X - expectation(X))/sqrt(variance(X) + epsilon) * scale + bias,
  // into Custom Calls to cuDNN.
  absl::Status MatchLayerNorm(HloInstruction* instr) {
    UniqueHloInstruction x, expectation, variance, epsilon;
    HloInstruction *scale, *bias, *reduce, *norm_factor, *broadcast_scale,
        *broadcast_bias;
    if (Match(
            instr,
            SubtractMultiplyAddAnyOrder(
                OptionalSupportedTransform(
                    m::Op().WithPredicate(x.CaptureOrVerifyFn())),
                Expectation(&expectation, &reduce,
                            OptionalSupportedTransform(
                                m::Op().WithPredicate(x.CaptureOrVerifyFn()))),
                NormFactor(&norm_factor, &x, &variance, &expectation, &epsilon),
                m::Broadcast(&broadcast_scale, m::Op(&scale)),
                m::Broadcast(&broadcast_bias, m::Op(&bias))))) {
#if CUDNN_VERSION < 8905
      // Layer norm kernels are available with cuDNN 8.9.5 and above.
      VLOG(1) << "Layer norm Custom Calls require cuDNN 8.9.5.";
      return absl::OkStatus();
#endif  // CUDNN_VERSION < 8905

      if (!instr->GetModule()
               ->config()
               .debug_options()
               .xla_gpu_enable_cudnn_layer_norm()) {
        VLOG(1) << "Layer norm Custom Calls disabled.";
        return absl::OkStatus();
      }

      // Layer norm kernels require Ampere or Hopper architectures.
      if (cuda_compute_capability_.major !=
              se::CudaComputeCapability::kAmpere &&
          cuda_compute_capability_.major !=
              se::CudaComputeCapability::kHopper) {
        VLOG(1) << "Layer norm Custom Calls require Ampere or Hopper "
                   "architectures.";
        return absl::OkStatus();
      }

      // Verify the uniqueness of the inputs.
      if (!x.Instr() || !expectation.Instr() || !variance.Instr() ||
          !epsilon.Instr()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }

      // Verify the input and output layouts.
      // TODO(philipphack): Consider supporting more general cases.
      if (!LayoutUtil::IsMonotonicWithDim0Major(x.Instr()->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(scale->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(bias->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout())) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }

      // Verify the element types. The element types of input and output and the
      // shapes of scale and bias must match. If a conversion to the type of the
      // input is the only user of the output, set the output to the conversion.
      // Similarly, to ensure the scale and bias have the same type, if the
      // scale/bias is a conversion from the type of the bias/scale, set the
      // scale/bias to the operand of the conversion. If scale and bias are type
      // conversions from the same type, set both to the operands of the
      // conversions.
      if (instr->user_count() == 1 &&
          instr->users()[0]->opcode() == HloOpcode::kConvert &&
          ShapeUtil::SameElementType(instr->users()[0]->shape(),
                                     x.Instr()->shape())) {
        instr = instr->users()[0];
      }
      if (HloPredicateIsOp<HloOpcode::kConvert>(scale) &&
          ShapeUtil::SameElementType(scale->operand(0)->shape(),
                                     bias->shape())) {
        scale = scale->mutable_operand(0);
      }
      if (HloPredicateIsOp<HloOpcode::kConvert>(bias) &&
          ShapeUtil::SameElementType(bias->operand(0)->shape(),
                                     scale->shape())) {
        bias = bias->mutable_operand(0);
      }
      if (HloPredicateIsOp<HloOpcode::kConvert>(scale) &&
          HloPredicateIsOp<HloOpcode::kConvert>(bias) &&
          ShapeUtil::SameElementType(scale->operand(0)->shape(),
                                     bias->operand(0)->shape())) {
        scale = scale->mutable_operand(0);
        bias = bias->mutable_operand(0);
      }
      if (!CompatibleElementType(instr) || !CompatibleElementType(scale) ||
          !CompatibleElementType(bias) ||
          !ShapeUtil::SameElementType(instr->shape(), x.Instr()->shape()) ||
          !ShapeUtil::Equal(scale->shape(), bias->shape())) {
        VLOG(1) << "Layer norm input types or shapes not supported.";
        return absl::OkStatus();
      }

      // Verify that the shapes of scale and bias are compatible with the
      // operation. The adjusted norm dimensions are the dimensions of the
      // reduction after removing any degenerate dimensions from the input of
      // the reduction.
      std::vector<int64_t> norm_dims(reduce->dimensions().begin(),
                                     reduce->dimensions().end());
      std::vector<int64_t> norm_dims_adjusted = AdjustedDimensions(reduce);
      if (norm_dims_adjusted.size() !=
          ShapeUtil::DropDegenerateDimensions(scale->shape())
              .dimensions()
              .size()) {
        VLOG(1) << "Layer norm input dimensions not supported.";
        return absl::OkStatus();
      }

      // Verify the broadcasts of scale and bias.
      if (!ShapeUtil::EqualIgnoringElementType(
              ShapeUtil::DropDegenerateDimensions(reduce->operand(0)->shape()),
              ShapeUtil::DropDegenerateDimensions(broadcast_scale->shape())) ||
          !ShapeUtil::EqualIgnoringElementType(
              ShapeUtil::DropDegenerateDimensions(reduce->operand(0)->shape()),
              ShapeUtil::DropDegenerateDimensions(broadcast_bias->shape())) ||
          norm_dims_adjusted != AdjustedDimensions(broadcast_scale) ||
          norm_dims_adjusted != AdjustedDimensions(broadcast_bias)) {
        VLOG(1) << "Layer norm operand broadcast not supported.";
        return absl::OkStatus();
      }

      // If necessary, transpose the input so that the dimensions not being
      // normalized are the leading dimensions.
      std::vector<int64_t> non_norm_dims;
      for (int64_t x_dim = 0; x_dim < x.Instr()->shape().dimensions().size();
           ++x_dim) {
        if (std::find(norm_dims.begin(), norm_dims.end(), x_dim) ==
            norm_dims.end()) {
          non_norm_dims.push_back(x_dim);
        }
      }
      std::vector<int64_t> non_norm_dims_adjusted =
          AdjustedDimensions(x.Instr()->shape(), non_norm_dims);

      std::vector<int64_t> x_transpose_order = non_norm_dims;
      x_transpose_order.insert(x_transpose_order.end(), norm_dims.begin(),
                               norm_dims.end());

      bool apply_transpose = false;
      for (int i = 0; i < x_transpose_order.size(); ++i) {
        if (x_transpose_order[i] != i) {
          apply_transpose = true;
          break;
        }
      }

      std::optional<HloInstruction*> x_transpose;
      // The transpose applied to the output is the inverse of the transpose
      // applied to the input.
      std::vector<int64_t> y_transpose_order(x_transpose_order.size());
      if (apply_transpose) {
        for (int k = 0; k < x_transpose_order.size(); ++k) {
          y_transpose_order[x_transpose_order[k]] = k;
        }
        TF_ASSIGN_OR_RETURN(x_transpose,
                            MakeTransposeHlo(x.Instr(), x_transpose_order));
      }

      // Combine the dimensions not normalized into the first dimension of the
      // input as required by cuDNN.
      std::vector<int64_t> reshaped_dims = {1};
      for (auto non_norm_dim : non_norm_dims) {
        reshaped_dims[0] *= x.Instr()->shape().dimensions(non_norm_dim);
      }
      for (auto norm_dim : norm_dims) {
        reshaped_dims.emplace_back(x.Instr()->shape().dimensions(norm_dim));
      }
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_dims.size() < 4) {
        reshaped_dims.push_back(1);
      }

      Shape reshaped_shape = ShapeUtil::MakeShape(
          x.Instr()->shape().element_type(), reshaped_dims);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * x_reshape,
          MakeReshapeHlo(reshaped_shape, x_transpose.value_or(x.Instr())));

      // Reshape the scale and bias. The first dimension corresponds to the
      // non-normalization dimension of the norm input and must have size 1.
      std::vector<int64_t> reshaped_scale_dims = reshaped_dims;
      reshaped_scale_dims[0] = 1;

      Shape scale_bias_shape = ShapeUtil::MakeShape(
          scale->shape().element_type(), reshaped_scale_dims);
      TF_ASSIGN_OR_RETURN(HloInstruction * scale_reshape,
                          MakeReshapeHlo(scale_bias_shape, scale));
      TF_ASSIGN_OR_RETURN(HloInstruction * bias_reshape,
                          MakeReshapeHlo(scale_bias_shape, bias));
      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_epsilon(
          epsilon.Instr()->literal().GetAsDouble({}).value());
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_INFER);
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);

      // Set the workspace size to its upper bound.
      // TODO(philipphack): Consider autotuning the norm kernels.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size =
          (2 * c_constant * (4 + 256)) + (2 * reshaped_dims[0] * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);

      // The output of the Custom Call is a tuple, the second element of which
      // describes the scratch space.
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {x_reshape->shape(), ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape, {x_reshape, scale_reshape, bias_reshape},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));

      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(custom_call, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * y_reshape,
          MakeReshapeHlo(x_transpose.value_or(instr)->shape(), gte));

      std::optional<HloInstruction*> y_transpose;
      if (apply_transpose) {
        TF_ASSIGN_OR_RETURN(y_transpose,
                            MakeTransposeHlo(y_reshape, y_transpose_order));
      }
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, y_transpose.value_or(y_reshape)));

      // Store metadata for potential use in the backward graph.
      norm_metadata_.insert(
          {custom_call,
           NormMetadata({x_transpose.value_or(nullptr),
                         y_transpose.value_or(nullptr), norm_dims_adjusted,
                         non_norm_dims_adjusted})});

      VLOG(1) << "Layer norm rewritten into Custom Call.";

      // The layer norm training graph separately contains the norm factor
      // divided by the sum of variance and epsilon.
      for (HloInstruction* user : norm_factor->users()) {
        if (HloPredicateIsOp<HloOpcode::kDivide>(user) &&
            user->operand_index(norm_factor) == 0) {
          TF_ASSIGN_OR_RETURN(bool changed,
                              MatchNormFactor(user, custom_call, variance,
                                              expectation, epsilon));
          if (changed) {
            break;
          }
        }
      }
    }

    return absl::OkStatus();
  }

  // The layer norm training graph separately contains the expectation as well
  // as the norm factor and its cube, (variance + epsilon)^-1/2 and (variance +
  // epsilon)^-3/2. When identified in the graph, these quantities are fused
  // into the layer norm Custom Call.
  absl::StatusOr<bool> MatchNormFactor(HloInstruction* instr,
                                       HloInstruction* custom_call,
                                       UniqueHloInstruction& variance,
                                       UniqueHloInstruction& expectation,
                                       UniqueHloInstruction& epsilon) {
    HloInstruction* gte = custom_call->users()[0];
    if (Match(instr,
              m::Divide(m::Op(),
                        AddAnyOrder(
                            m::Op().WithPredicate(variance.CaptureOrVerifyFn()),
                            m::Broadcast(m::ConstantScalar().WithPredicate(
                                epsilon.CaptureOrVerifyFn())))))) {
      // Verify the uniqueness of the operands.
      if (!variance.Instr() || !epsilon.Instr()) {
        VLOG(1) << "Layer norm operands not unique.";
        return false;
      }

      // Verify the element types.
      if (!CompatibleElementType(instr) ||
          !CompatibleElementType(expectation.Instr())) {
        VLOG(1) << "Layer norm input types not compatible.";
        return false;
      }

      // Retrieve metadata of the forward layer norm.
      auto norm_metadata = norm_metadata_.extract(custom_call);
      if (!norm_metadata) {
        VLOG(1) << "Unable to retrieve norm metadata of forward Custom Call.";
        return false;
      }

      // The shape of the expectation and norm factor return values of the
      // Custom Call is [nelems, 1, 1, 1], where nelems is the
      // number of elements in the expectation and norm factor shapes.
      auto make_compatible_shape = [](Shape shape) -> Shape {
        return ShapeUtil::MakeShape(shape.element_type(),
                                    {ShapeUtil::ElementsIn(shape), 1, 1, 1});
      };

      Shape expectation_shape =
          make_compatible_shape(expectation.Instr()->shape());
      Shape norm_factor_shape = make_compatible_shape(instr->shape());

      // The augmented Custom Call additionally returns the expectation and the
      // norm factor.
      std::vector<Shape> tuple_shapes = custom_call->shape().tuple_shapes();
      tuple_shapes.insert(tuple_shapes.begin() + 1,
                          {expectation_shape, norm_factor_shape});

      Shape custom_call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

      HloInstruction* new_custom_call = instr->AddInstruction(
          custom_call->CloneWithNewShape(custom_call_shape));

      TF_ASSIGN_OR_RETURN(
          GpuBackendConfig gpu_backend_config,
          custom_call->backend_config<xla::gpu::GpuBackendConfig>());
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_TRAIN);

      // Update the workspace size.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) + 32;
      backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
          workspace_size);
      TF_RETURN_IF_ERROR(
          new_custom_call->set_backend_config(gpu_backend_config));

      auto replace_with_new_cc = [new_custom_call, this](
                                     HloInstruction* old_instr,
                                     int tuple_index) -> absl::Status {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * new_gte,
            MakeGetTupleElementHlo(new_custom_call, tuple_index));
        HloInstruction* new_instr = new_gte;
        if (!ShapeUtil::Equal(new_gte->shape(), old_instr->shape())) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), new_gte));
        }
        if (HloPredicateIsNotOp<HloOpcode::kDivide>(old_instr)) {
          // Replace the result of the layer norm or the expectation.
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        } else {
          // Replace the norm factor, (variance + epsilon)^-1/2.
          TF_RETURN_IF_ERROR(
              ReplaceInstruction(old_instr->mutable_operand(0), new_instr));
          // Also replace the norm factor to the power of 3, (variance +
          // epsilon)^-1/2 / (variance + epsilon) = ((variance +
          // epsilon)^-1/2)^3.
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply0,
              MakeBinaryHlo(HloOpcode::kMultiply, new_instr, new_instr));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply1,
              MakeBinaryHlo(HloOpcode::kMultiply, new_multiply0, new_instr));
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_multiply1));
        }
        return absl::OkStatus();
      };

      // Replace the result of the original Custom Call as well as the
      // expectation and the norm factor with the augmented Custom Call.
      TF_RETURN_IF_ERROR(replace_with_new_cc(gte, 0));
      TF_RETURN_IF_ERROR(replace_with_new_cc(expectation.Instr(), 1));
      TF_RETURN_IF_ERROR(replace_with_new_cc(instr, 2));

      // Update the Custom Call associated with the metadata of the forward
      // norm.
      norm_metadata.key() = new_custom_call;
      norm_metadata_.insert(std::move(norm_metadata));

      VLOG(1)
          << "Expectation and norm factor fused into layer norm Custom Call.";
    }

    return true;
  }

  // Matches and rewrites the backward graph of layer norm patterns into Custom
  // Calls to cuDNN when the associated forward graph has been rewritten into a
  // cuDNN Custom Call. The gradients are
  //   DX = F1 + F2 - AddReduce(F1 + F2) / nelems,
  //   Dscale = AddReduce(DY * XCenter * NormFactor),
  //   Dbias = AddReduce(DY),
  // with
  //   F0 = XCenter * scale * DY,
  //   F1 =  XCenter * F0 * DNormFactor * 2 / nelems,
  //   F2 = NormFactor * scale * DY,
  //   XCenter = X - expectation(X),
  //   NormFactor = (variance(X) + epsilon)^-1/2 and
  //   DNormFactor = -1/2 * NormFactor^3.
  absl::Status MatchLayerNormGradient(HloInstruction* instr) {
    UniqueHloInstruction fwd_custom_call, x, x_center, scale, dy,
        fused_expectation, fused_norm_factor;
    HloInstruction *broadcast, *scalar, *dscale, *dbias, *reduce0, *reduce1,
        *reduce2, *reduce3;
    if (Match(instr,
              AddAddAnyOrder(
                  m::Broadcast(
                      &broadcast,
                      MultiplyAddAnyOrder(
                          m::Broadcast(m::ConstantScalar(&scalar)),
                          NegateAddReduce(&reduce0,
                                          F1(&x, &x_center, &fused_expectation,
                                             &fwd_custom_call, &scale, &dy,
                                             &reduce2, norm_metadata_)),
                          NegateAddReduce(
                              &reduce1, F2(&fused_norm_factor, &scale, &dy,
                                           &fwd_custom_call, norm_metadata_)))),
                  F2(&fused_norm_factor, &scale, &dy, &fwd_custom_call,
                     norm_metadata_),
                  F1(&x, &x_center, &fused_expectation, &fwd_custom_call,
                     &scale, &dy, &reduce3, norm_metadata_)))) {
      // Skip initial convert, if present.
      if (instr->user_count() == 1 &&
          instr->users()[0]->opcode() == HloOpcode::kConvert &&
          CompatibleElementType(instr->users()[0])) {
        instr = instr->users()[0];
      }

      // Verify the uniqueness of the captured Custom Call and inputs.
      if (!fwd_custom_call.Instr() || !x.Instr() || !dy.Instr() ||
          !x_center.Instr() || !scale.Instr() || !fused_expectation.Instr() ||
          !fused_norm_factor.Instr()) {
        VLOG(1) << "Layer norm gradient inputs not unique.";
        return absl::OkStatus();
      }

      // Retrieve metadata of the forward layer norm.
      auto norm_metadata = norm_metadata_.find(fwd_custom_call.Instr());
      if (norm_metadata == norm_metadata_.end()) {
        VLOG(1) << "Unable to retrieve norm metadata of forward Custom Call.";
        return absl::OkStatus();
      }

      // Verify the dimensions of reductions in the backward graph.
      if (AdjustedDimensions(reduce0) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce1) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce2) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce3) !=
              norm_metadata->second.norm_dims_adjusted) {
        VLOG(1) << "Unexpected reductions dimensions in layer norm gradient.";
        return absl::OkStatus();
      }

      // The captured scalar must be one over the number of elements in the
      // broadcasted dimensions.
      float actual_r_nelems = scalar->literal().GetAsDouble({}).value();
      int64_t nelems = 1;
      for (int i = 0; i < broadcast->shape().dimensions().size(); ++i) {
        if (!absl::c_linear_search(broadcast->dimensions(), i)) {
          nelems *= broadcast->shape().dimensions()[i];
        }
      }
      // The absolute of the difference between the actual scaling factor and
      // the reference value must not exceed a prescribed threshold.
      float r_nelems = 1. / static_cast<float>(nelems);
      float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
      if (!(abs(actual_r_nelems - r_nelems) <
            ((actual_r_nelems + r_nelems) * numerical_epsilon))) {
        VLOG(1)
            << "Layer norm backward broadcast operand outside expected range.";
        return absl::OkStatus();
      }

      // Identify Dscale = AddReduce(DY * XCenter * norm factor) with factor0
      // and factor1 intended to be XCenter and DY or DY and XCenter.
      auto find_dscale =
          [&fused_norm_factor, &norm_metadata](
              const UniqueHloInstruction& factor0,
              const UniqueHloInstruction& factor1) -> HloInstruction* {
        for (HloInstruction* factor0_user : factor0.Instr()->users()) {
          std::vector<HloInstruction*> users;
          SkipUnaryOpsTopDownRecursive(factor0_user, users);
          // One of the users of factor0 must be a chained multiplication by the
          // fused norm factor and factor1.
          for (HloInstruction* user : users) {
            if (Match(user,
                      MultiplyAnyOrder(
                          m::Op(), MultiplyAnyOrder(
                                       m::Broadcast(BitcastOrReshape(m::Op().Is(
                                           fused_norm_factor.Instr()))),
                                       m::Op().Is(factor1.Instr()))))) {
              // Dscale is an addition-reduction of the product.
              for (HloInstruction* multiply_user : user->users()) {
                if (AppliesAddReduce(
                        multiply_user,
                        norm_metadata->second.non_norm_dims_adjusted)) {
                  return multiply_user;
                }
              }
            }
          }
        }
        return nullptr;
      };
      if (!(dscale = find_dscale(x_center, dy)) &&
          !(dscale = find_dscale(dy, x_center))) {
        VLOG(1) << "Unable to identify Dscale in graph.";
        return absl::OkStatus();
      }

      // Find Dbias, i.e. an addition-reduction of DY, starting from DY.
      // Rewriting proceeds without fusing Dbias if unsuccessful.
      dbias = FindAddReduce(dy.Instr(),
                            norm_metadata->second.non_norm_dims_adjusted);

      // Verify the input and output layouts.
      // TODO(philipphack): Consider supporting more general cases.
      if (!LayoutUtil::IsMonotonicWithDim0Major(dy.Instr()->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(dscale->shape().layout()) ||
          (dbias &&
           !LayoutUtil::IsMonotonicWithDim0Major(dbias->shape().layout()))) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }

      // The types of X and DX must match.
      if (x.Instr()->shape().element_type() != instr->shape().element_type()) {
        VLOG(1) << "The types of X and DX must match.";
        return absl::OkStatus();
      }

      // The types and shapes of scale, Dscale and Dbias (if present) must
      // match.
      if (!ShapeUtil::Equal(
              ShapeUtil::DropDegenerateDimensions(scale.Instr()->shape()),
              ShapeUtil::DropDegenerateDimensions(dscale->shape())) ||
          (dbias &&
           !ShapeUtil::Equal(
               ShapeUtil::DropDegenerateDimensions(scale.Instr()->shape()),
               ShapeUtil::DropDegenerateDimensions(dbias->shape())))) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }

      // Verify the element types.
      if (!CompatibleElementType(dy.Instr())) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }

      // cuDNN requires the byte size of the element type of X to be at least
      // that of DY and scale.
      if (ShapeUtil::ByteSizeOfPrimitiveType(
              x.Instr()->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(
                  dy.Instr()->shape().element_type()) ||
          ShapeUtil::ByteSizeOfPrimitiveType(
              x.Instr()->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(
                  scale.Instr()->shape().element_type())) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }

      // Transpose DY applying the stored transpose order of X from the forward
      // graph.
      HloInstruction* transposed_dy = dy.Instr();
      if (norm_metadata->second.x_transpose) {
        TF_ASSIGN_OR_RETURN(
            transposed_dy,
            MakeTransposeHlo(dy.Instr(),
                             norm_metadata->second.x_transpose->dimensions()));
      }
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_dy,
                          MakeReshapeHlo(x.Instr()->shape(), transposed_dy));

      Shape dx_shape = ShapeUtil::MakeShape(instr->shape().element_type(),
                                            x.Instr()->shape().dimensions());

      Shape dscale_dbias_shape = ShapeUtil::MakeShape(
          dscale->shape().element_type(), scale.Instr()->shape().dimensions());

      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_BWD);
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);

      // Set the workspace size to its upper bound.
      // TODO(philipphack): Consider autotuning the norm kernels.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size =
          (2 * c_constant * (4 + 256)) +
          (2 * x.Instr()->shape().dimensions(0) * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);

      // The output of the Custom Call is a tuple. The output shape of Dscale
      // and Dbias is that of scale.
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {dx_shape, dscale_dbias_shape, dscale_dbias_shape,
           ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape,
              {x.Instr(), scale.Instr(), reshaped_dy, fused_expectation.Instr(),
               fused_norm_factor.Instr()},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));

      auto replace_with_cc = [custom_call, norm_metadata, transposed_dy, this](
                                 HloInstruction* old_instr,
                                 int tuple_index) -> absl::Status {
        TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                            MakeGetTupleElementHlo(custom_call, tuple_index));
        HloInstruction* new_instr;
        // Transpose DX applying the stored transpose order of Y from the
        // forward graph.
        if (tuple_index == 0 && norm_metadata->second.y_transpose) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(transposed_dy->shape(), gte));
          TF_ASSIGN_OR_RETURN(
              new_instr,
              MakeTransposeHlo(
                  new_instr, norm_metadata->second.y_transpose->dimensions()));
        } else {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), gte));
        }
        TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        return absl::OkStatus();
      };

      TF_RETURN_IF_ERROR(replace_with_cc(instr, 0));
      TF_RETURN_IF_ERROR(replace_with_cc(dscale, 1));
      if (dbias) {
        TF_RETURN_IF_ERROR(replace_with_cc(dbias, 2));
      }
      VLOG(1) << "Gradients w.r.t. x"
              << (dbias ? ", scale and bias" : " and scale")
              << " rewritten into layer norm backward Custom Call.";
    }

    return absl::OkStatus();
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
  NormMetadataMap norm_metadata_;
};

absl::StatusOr<bool> RunOnComputation(
    HloComputation* computation,
    se::CudaComputeCapability cuda_compute_capability) {
  CudnnNormRewriterVisitor visitor(cuda_compute_capability);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

CudnnNormRewriter::CudnnNormRewriter(
    se::CudaComputeCapability cuda_compute_capability)
    : cuda_compute_capability_(cuda_compute_capability) {}

absl::StatusOr<bool> CudnnNormRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
