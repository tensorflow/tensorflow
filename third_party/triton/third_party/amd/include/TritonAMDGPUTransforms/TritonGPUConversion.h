//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonGPUTypeConverter : public TypeConverter {
public:
  TritonGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                         int numCTAs);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

class TritonGPUConversionTarget : public ConversionTarget {

public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx,
                                     TritonGPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_TRITONGPUCONVERSION_H_
