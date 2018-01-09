#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsTruncatedNormalWhile(const HloInstruction *inst);
bool IsRandomNormal(const HloInstruction *inst);
bool IsRandomUniform(const HloInstruction *inst);
bool IsConstantZero(const HloInstruction *inst);
bool IsConstantHalf(const HloInstruction *inst);
bool IsPoplarConvolution(const HloInstruction *inst);
bool IsExternalPadding(const HloInstruction *inst);
bool IsAveragePool(const HloInstruction *inst);
bool Is2DReductionWindow(const HloInstruction *inst);
bool IsScalarConstant(const HloInstruction *inst);
bool IsDepthwisePadding(const HloInstruction *inst);
bool IsConvFilterSpatialReverse(const HloInstruction *inst);
bool IsBiasReduce(const HloInstruction *inst);
bool IsOutputFeed(const HloInstruction *inst);
bool IsForwardConvolution(const HloInstruction *inst);
bool IsGradientConvolution(const HloInstruction *inst);
bool IsWeightUpdateConvolution(const HloInstruction *inst);
bool IsForwardMatMul(const HloInstruction* inst);
bool IsGradientMatMul(const HloInstruction* inst);
bool IsWeightUpdateMatMul(const HloInstruction* inst);

}
}

#endif

