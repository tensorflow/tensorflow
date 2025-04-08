#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>
#include <optional>
#include <string>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(const std::string &target, int numWarps,
                                   int threadsPerWarp = 32, int numCTAs = 1);

} // namespace triton
} // namespace mlir

#endif
