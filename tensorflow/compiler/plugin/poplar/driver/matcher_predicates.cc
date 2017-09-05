#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace poplarplugin {

bool IsTruncatedNormalWhile(HloInstruction *inst) {
  return inst->while_condition()->name().substr(0, 16) == "truncated_normal";
}

bool IsRandomBernoulli(HloInstruction *inst) {
  return inst->random_distribution() == RandomDistribution::RNG_BERNOULLI;
}

bool IsRandomNormal(HloInstruction *inst) {
  return inst->random_distribution() == RandomDistribution::RNG_NORMAL;
}

bool IsRandomUniform(HloInstruction *inst) {
  return inst->random_distribution() == RandomDistribution::RNG_UNIFORM;
}

bool IsConstantZero(HloInstruction *inst) {
  return !ShapeUtil::HasZeroElements(inst->shape()) &&
         inst->literal().IsAll(0);
}

bool IsConstantHalf(HloInstruction *inst) {
  return !ShapeUtil::HasZeroElements(inst->shape()) &&
         inst->literal().IsAllFloat(0.5);
}

bool IsPoplarConvolution(HloInstruction *inst) {
  if (inst->to_apply()->name().substr(0, 15) == "pop_convolution") return true;
  if (inst->to_apply()->name().substr(0, 14) == "pop_depth_conv") return true;
  return false;
}

bool IsExternalPadding(HloInstruction *inst) {
  const PaddingConfig &cfg(inst->padding_config());
  for (auto &d : cfg.dimensions()) {
    if (d.interior_padding() > 0) return false;
  }
  return true;
}

bool IsAveragePool(HloInstruction *inst) {
  return inst->metadata().op_type() == "AvgPool";
}

bool IsReductionWindowNYXC(HloInstruction *inst) {
  const Window &window(inst->window());
  if (window.dimensions(0).size() != 1 ||
      window.dimensions(0).stride() != 1 ||
      window.dimensions(0).padding_low() != 0 ||
      window.dimensions(0).padding_high() != 0 ||
      window.dimensions(3).size() != 1 ||
      window.dimensions(3).stride() != 1 ||
      window.dimensions(3).padding_low() != 0 ||
      window.dimensions(3).padding_high() != 0) {
    return false;
  }
  return true;
}

bool IsScalarConstant(HloInstruction *inst) {
  return ShapeUtil::IsScalar(inst->shape());
}

bool IsDepthwisePadding(HloInstruction *inst) {
  if (inst->users().size() != 1) return false;
  HloInstruction *conv = inst->users()[0];
  const Shape &conv_shape(conv->shape());
  if (conv_shape.dimensions().size() != 4) return false;
  int64 conv_c = conv->shape().dimensions(2);

  const PaddingConfig &cfg(inst->padding_config());
  if (cfg.dimensions().size() != 4) return false;

  for (unsigned int i = 0; i < 4; i++) {
    if (cfg.dimensions(i).edge_padding_low() != 0) return false;
    if (cfg.dimensions(i).edge_padding_high() != 0) return false;
  }

  if (cfg.dimensions(0).interior_padding() != 0) return false;
  if (cfg.dimensions(1).interior_padding() != 0) return false;
  if (cfg.dimensions(2).interior_padding() != conv_c) return false;
  if (cfg.dimensions(3).interior_padding() != 0) return false;

  return true;
}

}
}

#endif

