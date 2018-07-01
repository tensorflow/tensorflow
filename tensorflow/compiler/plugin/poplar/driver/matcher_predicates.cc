#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

static bool IsPoplibsFusion(const HloInstruction *inst,
                            const std::string &type) {
  const HloComputation *comp = inst->to_apply();
  if (comp->name().substr(0, 8) == "_pop_op_") {
    auto end = comp->name().find('.');
    std::string name = comp->name().substr(8, end - 8);
    return name == type;
  }
  return false;
}

bool IsFloatType(const HloInstruction *inst) {
  return ShapeUtil::ElementIsFloating(inst->shape());
}

bool IsTruncatedNormalWhile(const HloInstruction *inst) {
  return inst->while_condition()->name().substr(0, 16) == "truncated_normal";
}

bool IsRandomNormal(const HloInstruction *inst) {
  return inst->random_distribution() == RandomDistribution::RNG_NORMAL;
}

bool IsRandomUniform(const HloInstruction *inst) {
  return inst->random_distribution() == RandomDistribution::RNG_UNIFORM;
}

bool IsConstantZero(const HloInstruction *inst) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAll(0);
}

bool IsConstantHalf(const HloInstruction *inst) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAllFloat(0.5);
}

bool IsConstantOne(const HloInstruction *inst) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAllFloat(1.0);
}

bool IsPoplarConvolution(const HloInstruction *inst) {
  if (inst->to_apply()->name().substr(0, 17) == "pop_backprop_conv")
    return true;
  if (inst->to_apply()->name().substr(0, 15) == "pop_convolution") return true;
  if (inst->to_apply()->name().substr(0, 14) == "pop_depth_conv") return true;
  return false;
}

bool IsExternalPadding(const HloInstruction *inst) {
  const PaddingConfig &cfg(inst->padding_config());
  for (auto &d : cfg.dimensions()) {
    if (d.interior_padding() > 0) return false;
  }
  return true;
}

bool IsAveragePool(const HloInstruction *inst) {
  return inst->metadata().op_type() == "AvgPool";
}

bool Is2DReductionWindow(const HloInstruction *inst) {
  const Window &window(inst->window());
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

bool IsScalarConstant(const HloInstruction *inst) {
  return ShapeUtil::IsScalar(inst->shape());
}

bool IsConvFilterTranspose(const HloInstruction *inst) {
  // If this reverse feeds a convolution and it is reversing the
  // spatial dimensions of the convolution, then we can use the
  // special 'reverse spatial dimensions' feature of the convolution
  // to achieve the reverse
  if (inst->users().size() != 1) return false;
  const std::vector<int64> &rev(inst->dimensions());

  HloInstruction *conv = inst->users()[0];
  const ConvolutionDimensionNumbers &d(conv->convolution_dimension_numbers());

  if (rev.size() != d.kernel_spatial_dimensions_size()) return false;
  for (int64 i = 0; i < rev.size(); i++) {
    if (d.kernel_spatial_dimensions(i) != rev[i]) return false;
  }

  return true;
}

bool IsBiasReduce(const HloInstruction *inst) {
  HloInstruction *root(inst->to_apply()->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }
  if (root->opcode() != HloOpcode::kAdd) {
    return false;
  }

  if (ShapeUtil::Rank(inst->shape()) != 1) return false;

  if (ShapeUtil::Rank(inst->operand(0)->shape()) != 4) return false;

  const std::vector<int64> &dims(inst->dimensions());
  if (dims.size() != ShapeUtil::Rank(inst->operand(0)->shape()) - 1) {
    return false;
  }
  for (int64 d : dims) {
    if (d == ShapeUtil::Rank(inst->operand(0)->shape()) - 1) return false;
  }
  return true;
}

bool IsOutputFeed(const HloInstruction *inst) {
  HloInstruction *root = inst->parent()->root_instruction();
  if (inst == root) return true;
  if (inst->user_count() != 1) return false;
  if (inst->users()[0] == root) return true;
  return false;
}

bool IsTfReluGradOp(const HloInstruction *inst) {
  const std::string &tf_core_op = inst->metadata().op_type();
  return tf_core_op == "ReluGrad";
}

bool IsTrueParameter(const HloInstruction *inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool IsFusedReverseInputConv(const HloInstruction *inst) {
  return IsPoplibsFusion(inst, "conv_with_reverse");
}

bool IsFusedDepthwiseConv(const HloInstruction *inst) {
  return IsPoplibsFusion(inst, "depthwise_conv");
}

bool Is1DVector(const HloInstruction *inst) {
  return ShapeUtil::Rank(inst->shape()) == 1;
}

bool IsF16(const HloInstruction *inst) {
  return inst->shape().element_type() == PrimitiveType::F16;
}

bool IsF32(const HloInstruction *inst) {
  return inst->shape().element_type() == PrimitiveType::F32;
}

bool IsF32ToF16Convert(const HloInstruction *inst) {
  return IsF16(inst) && IsF32(inst->operand(0));
}

bool IsF16ToF32Convert(const HloInstruction *inst) {
  return IsF32(inst) && IsF16(inst->operand(0));
}

bool IsPopOpsConvolution(const HloInstruction *inst) {
  if (inst->to_apply()->name().substr(0, 22) == "_pop_op_depthwise_conv")
    return true;
  if (inst->to_apply()->name().substr(0, 25) == "_pop_op_conv_with_reverse")
    return true;
  if (inst->to_apply()->name().substr(0, 24) == "_pop_op_depthwise_filter")
    return true;
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
