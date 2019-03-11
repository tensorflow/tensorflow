#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

static bool IsAllFloatValue(const HloInstruction* inst, const double value) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAllFloat(value);
}

bool IsFloatType(const HloInstruction* inst) {
  return ShapeUtil::ElementIsFloating(inst->shape());
}

bool IsTruncatedNormal(const HloInstruction* inst) {
  return inst->metadata().op_type() == "TruncatedNormal";
}

bool IsRandomNormal(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kRng &&
         inst->random_distribution() == RandomDistribution::RNG_NORMAL;
}

bool IsRandomUniform(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kRng &&
         inst->random_distribution() == RandomDistribution::RNG_UNIFORM;
}

bool IsConstantZero(const HloInstruction* inst) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAll(0);
}

bool IsConstantHalf(const HloInstruction* inst) {
  return IsAllFloatValue(inst, 0.5);
}

bool IsConstantOne(const HloInstruction* inst) {
  return IsAllFloatValue(inst, 1.0);
}

bool IsExternalPadding(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kPad) {
    return false;
  }

  const PaddingConfig& cfg(inst->padding_config());
  for (auto& d : cfg.dimensions()) {
    if (d.interior_padding() > 0) return false;
  }
  return true;
}

bool IsAveragePool(const HloInstruction* inst) {
  return inst->metadata().op_type() == "AvgPool";
}

bool Is2DMaxPool(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReduceWindow) {
    return false;
  }

  if (inst->metadata().op_type() == "MaxPool") {
    const Window& window(inst->window());
    unsigned reduction_dims = 0;
    for (int64 i = 0; i < window.dimensions_size(); i++) {
      auto& d = window.dimensions(i);
      if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
          d.padding_high() != 0) {
        reduction_dims++;
      }
    }
    return inst->window().dimensions_size() == 4 && reduction_dims == 2;
  }
  return false;
}

bool Is2DMaxPoolGrad(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kSelectAndScatter) {
    return false;
  }

  if (inst->metadata().op_type() == "MaxPoolGrad") {
    const Window& window(inst->window());
    unsigned reduction_dims = 0;
    for (int64 i = 0; i < window.dimensions_size(); i++) {
      auto& d = window.dimensions(i);
      if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
          d.padding_high() != 0) {
        reduction_dims++;
      }
    }
    return inst->window().dimensions_size() == 4 && reduction_dims == 2;
  }
  return false;
}

bool Is2DReductionWindow(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReduceWindow) {
    return false;
  }

  const Window& window(inst->window());
  int reduction_count = 0;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    if (window.dimensions(i).size() != 1 ||
        window.dimensions(i).stride() != 1 ||
        window.dimensions(i).padding_low() != 0 ||
        window.dimensions(i).padding_high() != 0) {
      reduction_count++;
    }
  }
  return reduction_count == 2;
}
bool IsScalar(const HloInstruction* inst) {
  return ShapeUtil::IsScalar(inst->shape());
}

bool IsScalarConstant(const HloInstruction* inst) {
  return IsScalar(inst) && inst->IsConstant();
}

bool IsScalarIntegerConstant(const HloInstruction* inst) {
  return IsScalar(inst) && inst->IsConstant() &&
         ShapeUtil::ElementIsIntegral(inst->shape());
}

bool IsConvFilterTranspose(const HloInstruction* inst) {
  // If this reverse feeds a convolution and it is reversing the
  // spatial dimensions of the convolution, then we can use the
  // special 'reverse spatial dimensions' feature of the convolution
  // to achieve the reverse
  if (inst->users().size() != 1) return false;
  const std::vector<int64>& rev(inst->dimensions());

  HloInstruction* conv = inst->users()[0];
  if (conv->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  const ConvolutionDimensionNumbers& d(conv->convolution_dimension_numbers());

  if (rev.size() != d.kernel_spatial_dimensions_size()) return false;
  for (int64 i = 0; i < rev.size(); i++) {
    if (d.kernel_spatial_dimensions(i) != rev[i]) return false;
  }

  return true;
}

bool IsBiasReduce(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }
  if (root->opcode() != HloOpcode::kAdd) {
    return false;
  }

  if (inst->shape().rank() != 1) return false;

  const std::vector<int64>& dims(inst->dimensions());
  if (dims.size() != inst->operand(0)->shape().rank() - 1) {
    return false;
  }
  return true;
}

bool IsOutputFeed(const HloInstruction* inst) {
  HloInstruction* root = inst->parent()->root_instruction();
  if (inst == root) return true;
  if (inst->user_count() != 1) return false;
  if (inst->users()[0] == root) return true;
  return false;
}

bool IsTfReluGradOp(const HloInstruction* inst) {
  const std::string& tf_core_op = inst->metadata().op_type();
  return tf_core_op == "ReluGrad";
}

bool IsTrueParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool Is1DVector(const HloInstruction* inst) {
  return inst->shape().rank() == 1;
}

bool IsExpandingReshape(const HloInstruction* inst) {
  return ShapeUtil::TrueRank(inst->shape()) == 1;
}

bool IsF16(const HloInstruction* inst) {
  return inst->shape().element_type() == PrimitiveType::F16;
}

bool IsF32(const HloInstruction* inst) {
  return inst->shape().element_type() == PrimitiveType::F32;
}

bool IsF32ToF16Convert(const HloInstruction* inst) {
  return IsF16(inst) && IsF32(inst->operand(0));
}

bool IsF16ToF32Convert(const HloInstruction* inst) {
  return IsF32(inst) && IsF16(inst->operand(0));
}

bool IsPopOpsConvolution(const HloInstruction* inst) {
  if (IsPopOpsFusion(inst, "depthwise_conv")) return true;
  if (IsPopOpsFusion(inst, "conv_with_reverse")) return true;
  if (IsPopOpsFusion(inst, "depthwise_filter")) return true;
  return false;
}

bool IsPopOpsConvolutionWithReverse(const HloInstruction* inst) {
  return (IsPopOpsFusion(inst, "conv_with_reverse"));
}

bool IsOpWithWindowNoBaseDilation(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return !window_util::HasBaseDilation(inst->window());
    default:
      return false;
  }
}

bool IsOpWithWindowNoStride(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return !window_util::HasStride(inst->window());
    default:
      return false;
  }
}

bool IsScalarConstantNegativeInfinity(const HloInstruction* inst) {
  return IsScalarConstant(inst) &&
         IsAllFloatValue(inst, -std::numeric_limits<double>::infinity());
}

bool IsScalarConstantOne(const HloInstruction* inst) {
  return IsScalarConstant(inst) && IsAllFloatValue(inst, 0);
}

bool IsPaddingReduceWindow(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReduceWindow) return false;
  // it's an identity, which means the window is of dim 1x...x1 and root of
  // to_apply computation is a parameter num 1
  const auto window = inst->window();
  for (const auto dim : window.dimensions()) {
    if (dim.size() != 1) return false;
  }
  const auto* root = inst->to_apply()->root_instruction();
  if (root->opcode() != HloOpcode::kParameter) return false;
  return root->parameter_number() == 1;
}

bool IsBiasAdd(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kAdd) {
    return false;
  }
  const auto& op_shape = inst->operand(0)->shape();
  const auto& bias_shape = inst->operand(1)->shape();
  if (op_shape.rank() != bias_shape.rank()) {
    return false;
  }

  // Go through the bias shape, if the dimension size is 1, then the dimension
  // of the op doesn't matter, otherwise they have to match.
  for (int64 i = 0; i < bias_shape.rank(); i++) {
    int64 bias_dim = ShapeUtil::GetDimension(bias_shape, i);
    if (bias_dim != 1) {
      int64 op_dim = ShapeUtil::GetDimension(op_shape, i);
      if (bias_dim != op_dim) {
        return false;
      }
    }
  }
  return true;
}

bool IsAddOrSubtract(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAdd ||
         inst->opcode() == HloOpcode::kSubtract;
}

bool IsPopOpsBiasAdd(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "matmul_biasadd") ||
         IsPopOpsFusion(inst, "conv_biasadd");
}

bool IsPopOpsElementwise(const HloInstruction* inst) {
  return IsPopOpsBiasAdd(inst) || IsPopOpsFusion(inst, "scaled_inplace") ||
         inst->IsElementwise() || IsPoplibsCustomOpElementwise(inst);
}

bool IsPopOpsElementwiseBinary(const HloInstruction* inst) {
  // Scaled inplace is a special case because it has 3 operands but the 3rd one
  // is always constant - we consider it a binary op.
  return (IsPopOpsElementwise(inst) && inst->operand_count() == 2) ||
         IsPopOpsFusion(inst, "scaled_inplace");
}

bool IsNormInference(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormInference ||
         IsPoplibsCustomOp(inst, PoplibsOp::Popnn,
                           PoplibsOp::GroupNormInference);
}

bool IsNormTraining(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormTraining ||
         IsPoplibsCustomOp(inst, PoplibsOp::Popnn,
                           PoplibsOp::GroupNormTraining);
}

bool IsNormInferenceOrTraining(const HloInstruction* inst) {
  return IsNormTraining(inst) || IsNormInference(inst);
}

bool IsNormGradient(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormGrad ||
         IsPoplibsCustomOp(inst, PoplibsOp::Popnn, PoplibsOp::GroupNormGrad);
}

bool IsGTEIndex0(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGetTupleElement &&
         inst->tuple_index() == 0;
}

bool IsGTEIndex1(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGetTupleElement &&
         inst->tuple_index() == 1;
}

bool IsGTEIndex2(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGetTupleElement &&
         inst->tuple_index() == 2;
}

bool IsNonLinearity(const HloInstruction* inst) {
  return !IsNonLinearityGradient(inst) &&
         (IsPopOpsFusion(inst, "relu") || IsPopOpsFusion(inst, "sigmoid"));
}

bool IsNonLinearityGradient(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "relugrad") ||
         IsPopOpsFusion(inst, "sigmoidgrad");
}

}  // namespace poplarplugin
}  // namespace xla
