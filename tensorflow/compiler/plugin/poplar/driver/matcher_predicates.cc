#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"


namespace xla {
namespace poplarplugin {

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
  return !ShapeUtil::HasZeroElements(inst->shape()) &&
         inst->literal().IsAll(0);
}

bool IsConstantHalf(const HloInstruction *inst) {
  return !ShapeUtil::HasZeroElements(inst->shape()) &&
         inst->literal().IsAllFloat(0.5);
}

bool IsPoplarConvolution(const HloInstruction *inst) {
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
  for (int64 i=0; i<window.dimensions_size(); i++) {
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

bool IsConvFilterSpatialReverse(const HloInstruction *inst) {
  // If this reverse feeds a convolution and it is reversing the
  // spatial dimensions of the convolution, then we can use the
  // special 'reverse spatial dimensions' feature of the convolution
  // to achieve the reverse
  if (inst->users().size() != 1) return false;
  const std::vector<int64>& rev(inst->dimensions());

  HloInstruction* conv = inst->users()[0];
  const ConvolutionDimensionNumbers& d(conv->convolution_dimension_numbers());

  if (rev.size() != d.kernel_spatial_dimensions_size()) return false;
  for (int64 i = 0; i < rev.size(); i++) {
    if (d.kernel_spatial_dimensions(i) != rev[i]) return false;
  }

  return true;
}

bool IsBiasReduce(const HloInstruction *inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }
  if (root->opcode() != HloOpcode::kAdd) {
      return false;
  }

  if (ShapeUtil::Rank(inst->shape()) != 1) return false;

  if (ShapeUtil::Rank(inst->operand(0)->shape()) != 4) return false;

  const std::vector<int64>& dims(inst->dimensions());
  if (dims.size() != ShapeUtil::Rank(inst->operand(0)->shape()) - 1) {
    return false;
  }
  for (int64 d : dims) {
    if (d == ShapeUtil::Rank(inst->operand(0)->shape()) - 1) return false;
  }
  return true;
}

bool IsOutputFeed(const HloInstruction *inst) {
  HloInstruction* root = inst->parent()->root_instruction();
  if (inst == root) return true;
  if (inst->user_count() != 1) return false;
  if (inst->users()[0] == root) return true;
  return false;
}

bool IsForwardConvolution(const HloInstruction *inst) {
  const std::string& tf_core_op = inst->metadata().op_type();
  return (tf_core_op == "Conv2D" ||
          tf_core_op == "Conv3D" ||
          tf_core_op == "DepthwiseConv2dNative");
}

bool IsGradientConvolution(const HloInstruction *inst) {
  const std::string& tf_core_op = inst->metadata().op_type();
  return (tf_core_op == "Conv2DBackpropInput" ||
          tf_core_op == "Conv3DBackpropInputV2" ||
          tf_core_op == "DepthwiseConv2dNativeBackpropInput");
}

bool IsWeightUpdateConvolution(const HloInstruction *inst) {
  const std::string& tf_core_op = inst->metadata().op_type();
  return (tf_core_op == "Conv2DBackpropFilter" ||
          tf_core_op == "Conv3DBackpropFilterV2" ||
          tf_core_op == "DepthwiseConv2dNativeBackpropFilter");
}

bool IsForwardMatMul(const HloInstruction* inst) {
  const HloInstruction* lh = inst->operand(0);
  const HloInstruction* rh = inst->operand(1);
  return (lh->opcode() != HloOpcode::kTranspose &&
          rh->opcode() != HloOpcode::kTranspose);
}

bool IsGradientMatMul(const HloInstruction* inst) {
  const HloInstruction* lh = inst->operand(0);
  const HloInstruction* rh = inst->operand(1);
  return (lh->opcode() != HloOpcode::kTranspose &&
          rh->opcode() == HloOpcode::kTranspose);
}

bool IsWeightUpdateMatMul(const HloInstruction* inst) {
  const HloInstruction* lh = inst->operand(0);
  const HloInstruction* rh = inst->operand(1);
  return (lh->opcode() == HloOpcode::kTranspose &&
          rh->opcode() != HloOpcode::kTranspose);
}

}
}

#endif

