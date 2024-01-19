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

#include "xla/service/gpu/cudnn_norm_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/dnn.pb.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn.h"        // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn_version.h"
#endif

namespace xla {
namespace gpu {

namespace {

namespace m = match;

// Returns an architecture-specific constant for the calculation of an upper
// bound for the size of the scratch space for layer norm kernels.
absl::StatusOr<int64_t> CConstant(
    se::CudaComputeCapability cuda_compute_capability) {
  if (cuda_compute_capability.major == se::CudaComputeCapability::AMPERE) {
    return 32 * 128;
  } else if (cuda_compute_capability.major ==
             se::CudaComputeCapability::HOPPER) {
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

// Returns whether the HLO Computation applied by instr calculates the sum of
// the elements.
bool AppliesAddReduce(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kReduce) {
    return false;
  }
  HloComputation* reduce_comp = instr->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return instr->operand_count() == 2 &&
         instr->operand(1)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsScalar(instr->operand(1)->shape()) &&
         instr->operand(1)->literal().GetAsDouble({}) == 0. &&
         reduce_comp_root->opcode() == HloOpcode::kAdd &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}

// Returns whether instr multiplies the result of a reduction by one over the
// number of reduced elements.
bool CalculatesExpectation(const HloInstruction* instr) {
  auto skip_convert_and_reshape =
      [](const HloInstruction* instr) -> const HloInstruction* {
    while (instr->opcode() == HloOpcode::kConvert ||
           instr->opcode() == HloOpcode::kReshape) {
      instr = instr->operand(0);
    }
    return instr;
  };

  instr = skip_convert_and_reshape(instr);
  if (instr->opcode() != HloOpcode::kMultiply) {
    return false;
  }
  bool bcast_operand = instr->operand(0)->opcode() != HloOpcode::kBroadcast;
  const HloInstruction *broadcast = instr->operand(bcast_operand),
                       *reduce = instr->operand(!bcast_operand);
  reduce = skip_convert_and_reshape(reduce);
  if (reduce->opcode() != HloOpcode::kReduce ||
      broadcast->opcode() != HloOpcode::kBroadcast ||
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

// Type conversion from and to any of BF16, FP16 and FP32.
template <typename Pattern>
auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

// Reshape adding or removing degenerate dimensions.
template <typename Pattern>
auto SupportedReshape(Pattern pattern) {
  auto supported_reshape = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::Equal(
        ShapeUtil::DropDegenerateDimensions(instr->shape()),
        ShapeUtil::DropDegenerateDimensions(instr->operand(0)->shape()));
  };
  return m::Reshape(pattern).WithPredicate(supported_reshape);
}

// Matches pattern, SupportedConvert(pattern), SupportedReshape(pattern),
// SupportedConvert(SupportedReshape(pattern)) and
// SupportedReshape(SupportedConvert(pattern)).
template <typename Pattern>
auto OptionalConvertAndOrReshape(Pattern pattern) {
  auto shared_subpattern = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(
      SupportedConvert(SupportedReshape(shared_subpattern)),
      SupportedReshape(SupportedConvert(shared_subpattern)),
      SupportedConvert(shared_subpattern), SupportedReshape(shared_subpattern),
      shared_subpattern);
}

// Rsqrt with optional convert and/or reshape.
template <typename Pattern>
auto Rsqrt(HloInstruction** rsqrt, Pattern pattern) {
  return OptionalConvertAndOrReshape(m::Rsqrt(rsqrt, pattern));
}

// AddAnyOrder with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto AddAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::AddAnyOrder(pattern0, pattern1));
}

// Subtract with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto Subtract(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::Subtract(pattern0, pattern1));
}

// Capturing subtract with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto Subtract(HloInstruction** subtract, Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::Subtract(subtract, pattern0, pattern1));
}

// Multiply with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::MultiplyAnyOrder(pattern0, pattern1));
}

// Capturing multiply with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(HloInstruction** multiply, Pattern0 pattern0,
                      Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(
      m::MultiplyAnyOrder(multiply, pattern0, pattern1));
}

// Multiplication of pattern by itself with optional convert and/or reshape.
template <typename Pattern>
auto Square(Pattern pattern) {
  return MultiplyAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}

// Addition-reduction of pattern with optional convert and/or reshape and
// constant 0 scalar.
template <typename Pattern>
auto AddReduce(Pattern pattern) {
  return OptionalConvertAndOrReshape(
      m::Reduce(pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}

// Capturing addition-reduction of pattern with optional convert and/or reshape
// and constant 0 scalar.
template <typename Pattern>
auto AddReduce(HloInstruction** reduction, Pattern pattern) {
  return OptionalConvertAndOrReshape(
      m::Reduce(reduction, pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
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
auto Expectation(HloInstruction** expectation, Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(expectation, m::Broadcast(m::ConstantScalar()),
                       AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(HloInstruction** expectation, HloInstruction** reduce,
                 Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(expectation, m::Broadcast(m::ConstantScalar()),
                       AddReduce(reduce, pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Variance, expressed as expectation(X^2) - expectation(X)^2 or
// expectation((X - expectation(X))^2). The simultaneous capture of input0 and
// input1 allows the caller to verify that they are identical.
auto Variance(HloInstruction** expectation, HloInstruction** input0,
              HloInstruction** input1) {
  return m::AnyOf<HloInstruction>(
      Subtract(Expectation(Square(m::Op(input0))),
               Square(Expectation(expectation, m::Op(input1)))),
      Expectation(Square(
          Subtract(m::Op(input0), Expectation(expectation, m::Op(input1))))));
}

// Variance, expressed as expectation(X^2) - expectation(X)^2 or
// expectation((X - expectation(X))^2). The simultaneous capture of input0 and
// input1 allows the caller to verify that they are identical.
auto Variance(HloInstruction** variance, HloInstruction** expectation,
              HloInstruction** input0, HloInstruction** input1) {
  return m::AnyOf<HloInstruction>(
      Subtract(variance, Expectation(Square(m::Op(input0))),
               Square(Expectation(expectation, m::Op(input1)))),
      Expectation(variance,
                  Square(Subtract(m::Op(input0),
                                  Expectation(expectation, m::Op(input1))))));
}

// Reciprocal of the square root of variance + epsilon with optional broadcast.
// The simultaneous capture of input0 and input1 allows the caller to verify
// that they are identical.
auto NormFactor(HloInstruction** norm_factor, HloInstruction** input0,
                HloInstruction** input1, HloInstruction** variance,
                HloInstruction** expectation, HloInstruction** epsilon) {
  auto shared_subpattern = m::SharedSubpattern(Rsqrt(
      norm_factor, AddAnyOrder(Variance(variance, expectation, input0, input1),
                               m::Broadcast(m::ConstantScalar(epsilon)))));
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

class CudnnNormRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNormRewriterVisitor(
      const se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}

  absl::Status HandleAdd(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  absl::Status HandleSubtract(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  // Matches and rewrites layer norm patterns,
  // (X - expectation(X))/(variance(X) + epsilon)^1/2 * scale + bias,
  // into Custom Calls to cuDNN.
  absl::Status MatchLayerNorm(HloInstruction* instr) {
    HloInstruction *input, *input0, *input1, *input2, *scale, *bias, *epsilon,
        *expectation, *expectation0, *reduce, *norm_factor, *variance,
        *broadcast_scale, *broadcast_bias;
    if (Match(instr, SubtractMultiplyAddAnyOrder(
                         m::Op(&input),
                         Expectation(&expectation, &reduce, m::Op(&input0)),
                         NormFactor(&norm_factor, &input1, &input2, &variance,
                                    &expectation0, &epsilon),
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
      if (cuda_compute_capability_.major != se::CudaComputeCapability::AMPERE &&
          cuda_compute_capability_.major != se::CudaComputeCapability::HOPPER) {
        VLOG(1) << "Layer norm Custom Calls require Ampere or Hopper "
                   "architectures.";
        return absl::OkStatus();
      }

      // Verify the uniqueness of the inputs.
      auto is_input = [input](HloInstruction* inputx) -> bool {
        return inputx->unique_id() == input->unique_id() ||
               (inputx->opcode() == HloOpcode::kConvert &&
                inputx->operand(0)->unique_id() == input->unique_id());
      };
      if (!is_input(input0) || !is_input(input1) || !is_input(input2) ||
          expectation->unique_id() != expectation0->unique_id()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }

      // Skip initial convert, if present.
      if (input->opcode() == HloOpcode::kConvert) {
        input = input->mutable_operand(0);
      }

      // Verify the input and output layouts.
      // TODO(philipphack): Consider supporting more general cases.
      if (!LayoutUtil::IsMonotonicWithDim0Major(input->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(scale->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(bias->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout())) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }

      // Verify the element types. The types and shapes of the scale and bias
      // must match.
      if (!CompatibleElementType(input) || !CompatibleElementType(instr) ||
          !CompatibleElementType(scale) || !CompatibleElementType(bias) ||
          !ShapeUtil::Equal(scale->shape(), bias->shape())) {
        VLOG(1) << "Layer norm input types or shapes not supported.";
        return absl::OkStatus();
      }

      // Verify that the shapes of scale and bias are compatible with the
      // operation.
      std::vector<int64_t> norm_dims(reduce->dimensions().begin(),
                                     reduce->dimensions().end());
      if (norm_dims.size() != scale->shape().dimensions_size()) {
        VLOG(1) << "Layer norm input dimensions not supported.";
        return absl::OkStatus();
      }
      for (int i = 0; i < norm_dims.size(); ++i) {
        if (input->shape().dimensions(norm_dims[i]) !=
            scale->shape().dimensions(i)) {
          VLOG(1) << "Layer norm input dimensions not supported.";
          return absl::OkStatus();
        }
      }

      // Verify the broadcasts of scale and bias.
      if (!ShapeUtil::EqualIgnoringElementType(reduce->operand(0)->shape(),
                                               broadcast_scale->shape()) ||
          !ShapeUtil::EqualIgnoringElementType(reduce->operand(0)->shape(),
                                               broadcast_bias->shape()) ||
          reduce->dimensions() != broadcast_scale->dimensions() ||
          reduce->dimensions() != broadcast_bias->dimensions()) {
        VLOG(1) << "Layer norm operand broadcast not supported.";
        return absl::OkStatus();
      }

      // If necessary, transpose the input so that the dimensions not being
      // normalized are the leading dimensions.
      std::vector<int64_t> non_norm_dims;
      for (int64_t input_dim = 0; input_dim < input->shape().rank();
           ++input_dim) {
        if (std::find(norm_dims.begin(), norm_dims.end(), input_dim) ==
            norm_dims.end()) {
          non_norm_dims.emplace_back(input_dim);
        }
      }
      std::vector<int64_t> transpose_order = non_norm_dims;
      transpose_order.insert(transpose_order.end(), norm_dims.begin(),
                             norm_dims.end());

      bool apply_transpose = false;
      for (int i = 0; i < transpose_order.size(); ++i) {
        if (transpose_order[i] != i) {
          apply_transpose = true;
          break;
        }
      }

      std::optional<HloInstruction*> transpose;
      std::vector<int64_t> inverse_transpose_order(transpose_order.size());
      if (apply_transpose) {
        for (int k = 0; k < transpose_order.size(); ++k) {
          inverse_transpose_order[transpose_order[k]] = k;
        }
        TF_ASSIGN_OR_RETURN(transpose,
                            MakeTransposeHlo(input, transpose_order));
      }

      // Combine the dimensions not normalized into the first dimension of the
      // input as required by cuDNN.
      std::vector<int64_t> reshaped_dims = {1};
      for (auto non_norm_dim : non_norm_dims) {
        reshaped_dims[0] *= input->shape().dimensions(non_norm_dim);
      }
      for (auto norm_dim : norm_dims) {
        reshaped_dims.emplace_back(input->shape().dimensions(norm_dim));
      }
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_dims.size() < 4) {
        reshaped_dims.emplace_back(1);
      }

      Shape reshaped_shape =
          ShapeUtil::MakeShape(input->shape().element_type(), reshaped_dims);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * reshape,
          MakeReshapeHlo(reshaped_shape, transpose.value_or(input)));

      // Reshape the scale and bias.
      std::vector<int64_t> reshaped_scale_dims(reshaped_dims.begin() + 1,
                                               reshaped_dims.end());
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_scale_dims.size() < 4) {
        reshaped_scale_dims.emplace_back(1);
      }
      Shape scale_bias_shape = ShapeUtil::MakeShape(
          scale->shape().element_type(), reshaped_scale_dims);
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_scale,
                          MakeReshapeHlo(scale_bias_shape, scale));
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_bias,
                          MakeReshapeHlo(scale_bias_shape, bias));
      GpuBackendConfig gpu_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_config.mutable_cudnn_norm_backend_config();
      backend_config.set_epsilon(epsilon->literal().GetAsDouble({}).value());
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      algorithm->set_is_cudnn_frontend(true);

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
          {reshape->shape(), ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape, {reshape, reshaped_scale, reshaped_bias},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(custom_call, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * inverse_reshape,
          MakeReshapeHlo(transpose.value_or(instr)->shape(), gte));

      if (!apply_transpose) {
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, inverse_reshape));
      } else {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * inverse_transpose,
            MakeTransposeHlo(inverse_reshape, inverse_transpose_order));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, inverse_transpose));
      }

      VLOG(1) << "Layer norm rewritten into Custom Call.";

      // The layer norm training graph separately contains the norm factor
      // divided by the sum of variance and epsilon.
      for (HloInstruction* user : norm_factor->users()) {
        if (user->opcode() == HloOpcode::kDivide &&
            user->operand_index(norm_factor) == 0) {
          TF_RETURN_IF_ERROR(MatchNormFactor(user, custom_call, variance,
                                             expectation, epsilon));
        }
      }
    }

    return absl::OkStatus();
  }

  // The layer norm training graph separately contains the expectation as well
  // as the norm factor and its cube, (variance + epsilon)^-1/2 and (variance +
  // epsilon)^-3/2. When identified in the graph, these quantities are fused
  // into the layer norm Custom Call.
  absl::Status MatchNormFactor(HloInstruction* instr,
                               HloInstruction* custom_call,
                               HloInstruction* variance,
                               HloInstruction* expectation,
                               HloInstruction* epsilon) {
    HloInstruction *variance0, *epsilon0, *gte = custom_call->users()[0];
    if (Match(instr,
              m::Divide(m::Op(), AddAnyOrder(m::Op(&variance0),
                                             m::Broadcast(m::ConstantScalar(
                                                 &epsilon0)))))) {
      // Verify the uniqueness of the operands.
      if (variance->unique_id() != variance0->unique_id() ||
          epsilon->unique_id() != epsilon0->unique_id()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }

      // Verify the element types.
      if (!CompatibleElementType(instr) ||
          !CompatibleElementType(expectation)) {
        VLOG(1) << "Layer norm input types not compatible.";
        return absl::OkStatus();
      }

      // The shape of the expectation and norm factor return values of the
      // Custom Call is [nelems, 1, 1, 1], where nelems is the
      // number of elements in the expectation and norm factor shapes.
      auto make_compatible_shape = [](Shape shape) -> Shape {
        return ShapeUtil::MakeShape(shape.element_type(),
                                    {ShapeUtil::ElementsIn(shape), 1, 1, 1});
      };

      Shape expectation_shape = make_compatible_shape(expectation->shape());
      Shape norm_factor_shape = make_compatible_shape(instr->shape());

      // The augmented Custom Call additionally returns the expectation and the
      // norm factor.
      std::vector<Shape> tuple_shapes = custom_call->shape().tuple_shapes();
      tuple_shapes.insert(tuple_shapes.begin() + 1,
                          {expectation_shape, norm_factor_shape});

      Shape custom_call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

      HloInstruction* new_custom_call = instr->AddInstruction(
          custom_call->CloneWithNewShape(custom_call_shape));

      // Update the workspace size.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) + 32;
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          custom_call->backend_config<GpuBackendConfig>());
      CudnnNormBackendConfig& backend_config =
          *gpu_config.mutable_cudnn_norm_backend_config();
      backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
          workspace_size);
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

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
        if (old_instr->opcode() != HloOpcode::kDivide) {
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
      TF_RETURN_IF_ERROR(replace_with_new_cc(expectation, 1));
      TF_RETURN_IF_ERROR(replace_with_new_cc(instr, 2));

      VLOG(1)
          << "Expectation and norm factor fused into layer norm Custom Call.";
    }
    return absl::OkStatus();
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
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
