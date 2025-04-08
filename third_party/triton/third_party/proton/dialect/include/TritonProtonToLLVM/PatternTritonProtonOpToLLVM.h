#ifndef TRITON_CONVERSION_TRITONPROTON_TO_LLVM_PATTERNS_TRITON_PROTON_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONPROTON_TO_LLVM_PATTERNS_TRITON_PROTON_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
class TargetInfoBase;
namespace proton {
void populateRecordOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);
} // namespace proton
} // namespace mlir::triton

#endif
