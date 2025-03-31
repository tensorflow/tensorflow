#ifndef TRITON_CONVERSION_FMA_DOT_UTILITY_H
#define TRITON_CONVERSION_FMA_DOT_UTILITY_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

/// Abstract interface for scalar multiplication of Value vectors.
///
/// Enable generation of hardware specific code in different backends.
class FMAVectorMultiplier {
public:
  /// \returns scalar product of two arrays, plus c: a·b + c
  virtual Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                                Value c) = 0;

  virtual ~FMAVectorMultiplier() = default;
};

/// Implements a framework for FMA dot conversion to llvm.
///
/// This function implements architecture independent part of FMA dot
/// conversion and calls "multiplier" object, which is defined by caller
/// and implements architecture dependant part of conversion.
LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier);

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_FMA_DOT_UTILITY_H
