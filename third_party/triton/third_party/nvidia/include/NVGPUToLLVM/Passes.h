#ifndef NVGPU_CONVERSION_PASSES_H
#define NVGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "nvidia/include/NVGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
