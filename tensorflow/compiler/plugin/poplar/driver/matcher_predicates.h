#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsFloatType(const HloInstruction*);
bool IsTruncatedNormalWhile(const HloInstruction*);
bool IsRandomNormal(const HloInstruction*);
bool IsRandomUniform(const HloInstruction*);
bool IsConstantZero(const HloInstruction*);
bool IsConstantHalf(const HloInstruction*);
bool IsConstantOne(const HloInstruction*);
bool IsPoplarConvolution(const HloInstruction*);
bool IsExternalPadding(const HloInstruction*);
bool IsAveragePool(const HloInstruction*);
bool Is2DReductionWindow(const HloInstruction*);
bool IsScalarConstant(const HloInstruction*);
bool IsConvFilterTranspose(const HloInstruction*);
bool IsBiasReduce(const HloInstruction*);
bool IsOutputFeed(const HloInstruction*);
bool IsTfReluGradOp(const HloInstruction*);
bool IsTrueParameter(const HloInstruction*);
bool IsFusedReverseInputConv(const HloInstruction*);
bool IsFusedDepthwiseConv(const HloInstruction*);
bool Is1DVector(const HloInstruction*);
bool IsF16(const HloInstruction*);
bool IsF32(const HloInstruction*);
bool IsF32ToF16Convert(const HloInstruction*);
bool IsF16ToF32Convert(const HloInstruction*);
bool IsPopOpsConvolution(const HloInstruction*);

}  // namespace poplarplugin
}  // namespace xla

#endif
