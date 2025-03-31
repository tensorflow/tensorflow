#ifndef TRITON_CONVERSION_PASSES_H
#define TRITON_CONVERSION_PASSES_H

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
