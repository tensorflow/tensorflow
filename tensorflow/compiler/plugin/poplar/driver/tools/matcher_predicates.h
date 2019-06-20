#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsRandomNormal(const HloInstruction*);
bool IsRandomUniform(const HloInstruction*);
bool IsCompareEqual(const HloInstruction*);
bool IsConstantZero(const HloInstruction*);
bool IsExternalPadding(const HloInstruction*);
bool IsScalar(const HloInstruction*);
bool IsScalarConstant(const HloInstruction*);
bool IsScalarIntegerConstant(const HloInstruction*);
bool IsConvFilterTranspose(const HloInstruction*);
bool IsBiasReduce(const HloInstruction*);
bool IsOutputFeed(const HloInstruction*);
bool Is1DVector(const HloInstruction*);
bool IsExpandingReshape(const HloInstruction*);
bool IsF16(const HloInstruction*);
bool IsF32(const HloInstruction*);
bool IsF32ToF16Convert(const HloInstruction*);
bool IsF16ToF32Convert(const HloInstruction*);
bool IsPopOpsConvolution(const HloInstruction*);
bool IsPopOpsConvolutionWithReverse(const HloInstruction*);
bool IsOpWithWindowNoBaseDilation(const HloInstruction*);
bool IsOpWithWindowNoStride(const HloInstruction*);
bool IsPaddingReduceWindow(const HloInstruction*);
bool IsAddOrSubtract(const HloInstruction*);
bool IsBiasAdd(const HloInstruction*);
bool IsPopOpsBiasAdd(const xla::HloInstruction*);
bool IsPopOpsElementwise(const xla::HloInstruction*);
bool IsPopOpsElementwiseBinary(const xla::HloInstruction*);
bool IsNormInference(const xla::HloInstruction*);
bool IsNormTraining(const xla::HloInstruction*);
bool IsNormInferenceOrTraining(const xla::HloInstruction*);
bool IsNormGradient(const xla::HloInstruction*);
bool IsNonLinearityGradient(const xla::HloInstruction*);
bool IsNonLinearity(const xla::HloInstruction*);
}  // namespace poplarplugin
}  // namespace xla

#endif
