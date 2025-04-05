#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "llvm/TargetParser/TargetParser.h"

namespace mlir::triton::AMD {

ISAFamily deduceISAFamily(llvm::StringRef arch) {
  llvm::AMDGPU::GPUKind kind = llvm::AMDGPU::parseArchAMDGCN(arch);

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for how to categorize
  // the following target gfx architectures.

  // CDNA ISA cases
  switch (kind) {
  case llvm::AMDGPU::GK_GFX950:
    return ISAFamily::CDNA4;
  case llvm::AMDGPU::GK_GFX942:
    return ISAFamily::CDNA3;
  case llvm::AMDGPU::GK_GFX90A:
    return ISAFamily::CDNA2;
  case llvm::AMDGPU::GK_GFX908:
    return ISAFamily::CDNA1;
  default:
    break;
  }

  // RNDA ISA cases
  if (kind >= llvm::AMDGPU::GK_GFX1100 && kind <= llvm::AMDGPU::GK_GFX1201)
    return ISAFamily::RDNA3;
  if (kind >= llvm::AMDGPU::GK_GFX1030 && kind <= llvm::AMDGPU::GK_GFX1036)
    return ISAFamily::RDNA2;
  if (kind >= llvm::AMDGPU::GK_GFX1010 && kind <= llvm::AMDGPU::GK_GFX1013)
    return ISAFamily::RDNA1;

  return ISAFamily::Unknown;
}

bool supportsVDot(llvm::StringRef arch) {
  switch (deduceISAFamily(arch)) {
  case AMD::ISAFamily::CDNA1:
  case AMD::ISAFamily::CDNA2:
  case AMD::ISAFamily::CDNA3:
  case AMD::ISAFamily::CDNA4:
  case AMD::ISAFamily::RDNA2:
  case AMD::ISAFamily::RDNA3:
    return true;
  default:
    break;
  }
  return false;
}

} // namespace mlir::triton::AMD
