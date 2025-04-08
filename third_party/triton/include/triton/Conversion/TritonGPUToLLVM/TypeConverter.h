#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx,
                               const LowerToLLVMOptions &option,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);
  TritonGPUToLLVMTypeConverter(MLIRContext *ctx,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type convertTritonTensorType(RankedTensorType type,
                               const TargetInfoBase &targetInfo);
  Type convertMemDescType(triton::gpu::MemDescType type,
                          const TargetInfoBase &targetInfo);
  Type convertAsyncTokenType(triton::gpu::AsyncTokenType type);

  template <typename... T> void convertFP8Type() {
    (addConversion([&](T type) -> std::optional<Type> {
       return IntegerType::get(type.getContext(), 8);
     }),
     ...);
  }
};

#endif
