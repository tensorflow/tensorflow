#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_MATCHER_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

struct CompilerAnnotations;

bool IsFloatType(const HloInstruction*, const CompilerAnnotations&);
bool IsTruncatedNormalWhile(const HloInstruction*, const CompilerAnnotations&);
bool IsRandomNormal(const HloInstruction*, const CompilerAnnotations&);
bool IsRandomUniform(const HloInstruction*, const CompilerAnnotations&);
bool IsConstantZero(const HloInstruction*, const CompilerAnnotations&);
bool IsConstantHalf(const HloInstruction*, const CompilerAnnotations&);
bool IsConstantOne(const HloInstruction*, const CompilerAnnotations&);
bool IsPoplarConvolution(const HloInstruction*, const CompilerAnnotations&);
bool IsExternalPadding(const HloInstruction*, const CompilerAnnotations&);
bool IsAveragePool(const HloInstruction*, const CompilerAnnotations&);
bool Is2DReductionWindow(const HloInstruction*, const CompilerAnnotations&);
bool IsScalarConstant(const HloInstruction*, const CompilerAnnotations&);
bool IsConvFilterTranspose(const HloInstruction*, const CompilerAnnotations&);
bool IsBiasReduce(const HloInstruction*, const CompilerAnnotations&);
bool IsOutputFeed(const HloInstruction*, const CompilerAnnotations&);
bool IsForward(const HloInstruction*, const CompilerAnnotations&);
bool IsBackpropInput(const HloInstruction*, const CompilerAnnotations&);
bool IsBackpropFilter(const HloInstruction*, const CompilerAnnotations&);
bool IsTfReluGradOp(const HloInstruction*, const CompilerAnnotations&);
bool IsTrueParameter(const HloInstruction*, const CompilerAnnotations&);
bool IsFusedReverseInputConv(const HloInstruction*, const CompilerAnnotations&);
bool IsFusedDepthwiseConv(const HloInstruction*, const CompilerAnnotations&);
bool Is1DVector(const HloInstruction*, const CompilerAnnotations&);

}  // namespace poplarplugin
}  // namespace xla

#endif
