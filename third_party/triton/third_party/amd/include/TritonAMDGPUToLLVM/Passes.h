#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "TritonAMDGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton::AMD {
/// @brief Creates pass that keep LDS consumption within specified limits.
/// @param arch target architecture name, for example "gfx940"
/// @param customLDSLimit defines LDS size available for one thread block
/// zero value tells pass that whole LDS is available on a device
/// @return created pass
std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(StringRef arch, int32_t customLDSLimit = 0);
} // namespace mlir::triton::AMD

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef targetArch, bool ftz);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz);
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUInsertInstructionSchedHintsPass(StringRef variant);
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPULowerInstructionSchedHintsPass(StringRef arch,
                                                 int32_t numStages);

#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_
