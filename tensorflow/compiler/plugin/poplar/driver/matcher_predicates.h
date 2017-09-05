#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsTruncatedNormalWhile(HloInstruction *inst);
bool IsRandomBernoulli(HloInstruction *inst);
bool IsRandomNormal(HloInstruction *inst);
bool IsRandomUniform(HloInstruction *inst);
bool IsConstantZero(HloInstruction *inst);
bool IsConstantHalf(HloInstruction *inst);
bool IsPoplarConvolution(HloInstruction *inst);
bool IsExternalPadding(HloInstruction *inst);
bool IsAveragePool(HloInstruction *inst);
bool IsReductionWindowNYXC(HloInstruction *inst);
bool IsScalarConstant(HloInstruction *inst);
bool IsDepthwisePadding(HloInstruction *inst);

}
}

#endif

